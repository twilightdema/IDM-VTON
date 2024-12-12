#VITON-HD
##paired setting
python3 test1_inpaint_sd15.py --pretrained_model_name_or_path "result_sd15/checkpoint-best-train-loss" \
    --width 384 --height 512 --num_inference_steps 30 \
    --output_dir "test1_results_sd15" --data_dir "/home/nok/Downloads/zalando-hd-resized" \
    --seed 42 --test_batch_size 1 --guidance_scale 2.0
