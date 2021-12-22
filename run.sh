export CUDA_VISIBLE_DEVICES=1

python train.py --A_des 'train resnet with StepLR' \
                --model resnet50 --sched step --decay-epochs 50\
                --data_dir /home/project/data2/our_data_processed \
                --dataset_type all &> logs/res_step_lr.out && \

python train.py --A_des 'train resnet with crop data' \
                --model resnet50 \
                --data_dir /home/project/data2/our_data_cropped \
                --dataset_type all &> logs/res_crop.out && \

python train.py --A_des 'train resnet with mask data' \
                --model resnet50 \
                --data_dir /home/project/data2/our_data_masked \
                --dataset_type all &> logs/res_mask.out && \

python train.py --A_des 'efficientnetb2 with mask data' \
                --model efficientnet_b2 -b 64 --epochs 300 \
                --data_dir /home/project/data2/our_data_masked \
                --dataset_type all &> logs/eff_mask.out  && \

python train.py --A_des 'train the resnet with extra checkpoint initialization' \
                --initial-checkpoint '/home/project/ml-project2-TB/output/train/20211220-105317-resnet50-320/checkpoint-156.pth.tar' \
                --model resnet50 \
                --data_dir /home/project/data2/our_data_processed \
                --dataset_type all &> logs/res_extral_initial.out && \

python train.py --A_des 'train the resnet with extra checkpoint initialization and small lr on HIV' \
                --initial-checkpoint '/home/project/ml-project2-TB/output/train/20211220-105317-resnet50-320/checkpoint-156.pth.tar' \
                --model resnet50 --lr 0.001 \
                --data_dir /home/project/data2/our_data_processed \
                --dataset_type H &> logs/res_extral_initial_smallLR_HIV.out && \

python train.py --A_des 'train the resnet with extra checkpoint initialization and small lr on Dia' \
                --initial-checkpoint '/home/project/ml-project2-TB/output/train/20211220-105317-resnet50-320/checkpoint-156.pth.tar' \
                --model resnet50 --lr 0.001 \
                --data_dir /home/project/data2/our_data_processed \
                --dataset_type D &> logs/res_extral_initial_smallLR_Dia.out && \

python train.py --A_des 'train the resnet with extra checkpoint initialization and small lr and f1 loss on HIV' \
                --initial-checkpoint '/home/project/ml-project2-TB/output/train/20211220-105317-resnet50-320/checkpoint-156.pth.tar' \
                --model resnet50 --lr 0.001 --f1-loss \
                --data_dir /home/project/data2/our_data_processed \
                --dataset_type H &> logs/res_extral_initial_smallLR_f1_HIV.out