python inference.py --cfg 7.5 --total_noise_levels 900 --steps 45 \
--model_path ckpts/venhancer_paper.pt \
--start_frame 0 --max_frame_num 48 \
--up_scale 4 --target_fps 24 --noise_aug 200 \
--input_path inputs/sample.mp4 \
--prompt 'Clown fish swimming through the coral reef.' \
--save_dir 'results/'