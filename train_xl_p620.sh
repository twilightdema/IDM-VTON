# CUDA_VISIBLE_DEVICES=0 accelerate launch train_xl.py --gradient_checkpointing --mixed_precision=fp16 --use_8bit_adam --output_dir=result --train_batch_size=1 --data_dir=/storage/datasets/zalando-hd-resized

python3 train_xl.py --gradient_checkpointing --mixed_precision=fp16 --use_8bit_adam --output_dir=result --train_batch_size=1 --data_dir=/home/nok/Downloads/zalando-hd-resized
