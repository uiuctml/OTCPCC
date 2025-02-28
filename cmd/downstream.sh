CUDA_VISIBLE_DEVICES=0 python downstream.py \                                    
 --timestamp ottest --dataset CIFAR100 \
 --cpcc fft --num_workers 0 \
 --seeds 1