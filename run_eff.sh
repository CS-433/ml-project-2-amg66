export CUDA_VISIBLE_DEVICES=0
python train.py --model efficientnet_b2 -b 64 --sched step --epochs 450 --decay-epochs 2.4 \
                --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 \
                --weight-decay 1e-5 --drop 0.3 --drop-connect 0.2 --lr .016 \
                --data_dir /media/data/mu/ML2/data2/our_data_processed \
                --dataset_type D &> logs/eff_D.out && \

python train.py --model efficientnet_b2 -b 64 --sched step --epochs 450 --decay-epochs 2.4 \
                --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 \
                --weight-decay 1e-5 --drop 0.3 --drop-connect 0.2 --lr .016 \
                --data_dir /media/data/mu/ML2/data2/our_data_processed \
                --dataset_type H &> logs/eff_H.out && \

python train.py --model efficientnet_b2 -b 64 --sched step --epochs 450 --decay-epochs 2.4 \
                --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 \
                --weight-decay 1e-5 --drop 0.3 --drop-connect 0.2 --lr .016 \
                --data_dir /media/data/mu/ML2/data2/our_data_processed \
                --dataset_type all &> logs/eff_all.out