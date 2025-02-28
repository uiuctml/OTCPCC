CUDA_VISIBLE_DEVICES=0 python pretrain.py \
 --timestamp ottest-cifar10 --dataset CIFAR10 \
 --cpcc fft  --num_workers 4 \
 --seeds 1 --train 1

CUDA_VISIBLE_DEVICES=0 python pretrain.py \
 --timestamp ottest-cifar100 --dataset CIFAR100 \
 --cpcc fft  --num_workers 4 \
 --seeds 1 --train 1

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 pretrain.py \
 --timestamp ottest-breeds --dataset BREEDS --breeds_setting living17\
 --cpcc fft  --num_workers 12 \
 --seeds 1 --train 1

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 pretrain.py \
 --timestamp ottest-inat --dataset INAT \
 --cpcc fft  --num_workers 12 \
 --seeds 1 --train 1