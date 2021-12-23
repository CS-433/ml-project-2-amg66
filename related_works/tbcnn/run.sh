export CUDA_VISIBLE_DEVICES=0
python train.py --method 1 --type H  &> out_m1H.out && \
python train.py --method 2 --type H  &> out_m2H.out && \
python train.py --method 1 --type all &> out_m1A.out && \
python train.py --method 2 --type all  &> out_m2A.out
# python train.py --method 1 --type D &> out_m1.out && \
# python train.py --method 2 --type D  &> out_m2.out