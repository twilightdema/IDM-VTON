# CUDA_VISIBLE_DEVICES=0 accelerate launch train_sd15.py --gradient_checkpointing --use_8bit_adam --output_dir=result_sd15 --train_batch_size=3 --data_dir=/storage/datasets/zalando-hd-resized

python3 train_sd15.py --gradient_checkpointing --mixed_precision=fp16 --use_8bit_adam --output_dir=result_sd15 --train_batch_size=2 --data_dir=/storage/datasets/zalando-hd-resized
