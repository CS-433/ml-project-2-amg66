import argparse
import time

import os
import logging
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from dataset.data_loader import create_loader
from timm.data import resolve_data_config

from dataset.dataset import *
# from dataset.data_loader import *
from timm.models import create_model, safe_model_name
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler

from f1_loss import F1_Loss, f1_score
from focal_loss import FocalLoss
from arguments import parse_args

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

def main():

    setup_default_logging()
    args, args_text = parse_args()
    random_seed(args.seed, 0)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=0.0,
        global_pool=args.gp,
        checkpoint_path=args.initial_checkpoint)

    data_config = resolve_data_config(vars(args), model=model, verbose=True)

    # move model to GPU, enable channels last layout if set
    model.cuda()

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    _logger.info('Scheduled epochs: {}'.format(num_epochs))


    # dataset_dir = ['/media/data/mu/ML2/data/Diabetes/images', '/media/data/mu/ML2/data/HIV/images']
    loader_train, loader_eval, num_train, num_eval = create_loader(args.data_dir, batch_size=args.batch_size)

    # setup loss function
    if args.focal_loss:
        train_loss_fn = FocalLoss(gamma=2, alpha=0.25)
        print("using the focal loss")
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    best_metric = None
    best_epoch = None

    exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"),safe_model_name(args.model),str(data_config['input_size'][-1])])
    output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
    saver = CheckpointSaver(
        model=model, optimizer=optimizer, args=args, model_ema=model_ema,
        checkpoint_dir=output_dir, recovery_dir=output_dir)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    try:
        for epoch in range(num_epochs):
            print(f'epoch: {epoch}')
            start = time.time()

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, output_dir=output_dir, model_ema=model_ema)

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, num_eval)

            if model_ema is not None and not args.model_ema_force_cpu:
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics['top1'])

            if output_dir is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics['top1']
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            end = time.time()
            print(f'Time for epoch-{epoch} :', end-start)
    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args, lr_scheduler=None, output_dir=None, model_ema=None):
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)

        output = model(input)
        loss = loss_fn(output, target)

        losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward(create_graph=second_order)
        optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        # print('batch time:', time.time() - end)

        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            _logger.info(
                'Train: {} [{:>4d}/{}]  ''Loss: {loss.val:#.4g} ({loss.avg:#.3g})  ''LR: {lr:.3e}'
                .format(epoch, batch_idx, len(loader), loss=losses_m, lr=lr))

            if args.save_images and output_dir:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                    padding=0,
                    normalize=True)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()


    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, sample_num, log_suffix=''):
    print('-----sample num', sample_num)
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    output_all = torch.zeros((sample_num, 2)).cuda()
    target_all = torch.zeros(sample_num).cuda()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            [acc1] = accuracy(output, target, topk=(1,))

            output_all[batch_idx*args.batch_size:batch_idx*args.batch_size+output.size(0),:] = output.detach()
            target_all[batch_idx*args.batch_size:batch_idx*args.batch_size+output.size(0)] = target.detach()

            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            # print('batch time:', time.time() - end)
            end = time.time()

            # if batch_idx % args.log_interval == 0 or last_batch:
            if batch_idx % args.log_interval == 0:
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '.format(
                        log_name, batch_idx, last_idx, 
                        loss=losses_m, top1=top1_m))

            if last_batch:
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'F1score: {f1:.3f}'.format(
                        log_name, batch_idx, last_idx,
                        loss=losses_m, top1=top1_m, f1 = f1_score(target_all, output_all)))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg)])

    return metrics

def test(model, loader, loss_fn, args, sample_num, log_suffix=''):
    print('-----sample num', sample_num)
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    output_all = torch.zeros((sample_num, 2)).cuda()
    target_all = torch.zeros(sample_num).cuda()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            [acc1] = accuracy(output, target, topk=(1,))

            output_all[batch_idx*args.batch_size:batch_idx*args.batch_size+output.size(0),:] = output.detach()
            target_all[batch_idx*args.batch_size:batch_idx*args.batch_size+output.size(0)] = target.detach()

            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            # print('batch time:', time.time() - end)
            end = time.time()

            # if batch_idx % args.log_interval == 0 or last_batch:
            if batch_idx % args.log_interval == 0:
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '.format(
                        log_name, batch_idx, last_idx, 
                        loss=losses_m, top1=top1_m))

            if last_batch:
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'F1score: {f1:.3f}'.format(
                        log_name, batch_idx, last_idx,
                        loss=losses_m, top1=top1_m, f1 = f1_score(target_all, output_all)))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg)])

    return metrics

if __name__ == '__main__':
    main()
