CUDA_VISIBLE_DEVICES=0 accelerate launch train_xl.py --gradient_checkpointing --use_8bit_adam --output_dir=result --train_batch_size=6 --data_dir=/home/nok/Downloads/zalando-hd-resized
