#VITON-HD
##paired setting
python3 inference_sd15.py --pretrained_model_name_or_path "result_sd15/checkpoint-best-train-loss" \
    --width 384 --height 512 --num_inference_steps 30 \
    --output_dir "inference_result_sd15" --data_dir "/storage/datasets/zalando-hd-resized" \
    --seed 42 --test_batch_size 2 --guidance_scale 2.0

# accelerate launch inference_sd15.py --pretrained_model_name_or_path "result_sd15/checkpoint-best-train-loss" \
#     --width 384 --height 512 --num_inference_steps 30 \
#     --output_dir "inference_result_sd15" --data_dir "/home/nok/Downloads/zalando-hd-resized" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0


# ##unpaired setting
# accelerate launch inference.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/Dataset/zalando" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0



# #DressCode
# ##upper_body
# accelerate launch inference_dc.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/DressCode" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0 --category "upper_body"

# ##lower_body
# accelerate launch inference_dc.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/DressCode" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0 --category "lower_body"

# ##dresses
# accelerate launch inference_dc.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/DressCode" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0 --category "dresses"