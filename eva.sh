CUDA_VISIBLE_DEVICES=0 nohup python3 evaluate.py --configs './cfgs/vox2_img.ini' --input_dir '/Light_distangle/Data/occface/data/voxceleb_pics' --output_dir './faceresults/deface/voxceleb_pics' --ckpt_dir './ckpts/voxceleb_pics' >result.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 evaluate.py --configs './cfgs/por_img.ini' --input_dir '/Light_distangle/Data/occface/data/porpics' --output_dir './faceresults/deface/porpics' --ckpt_dir './ckpts/porpics' >result.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 evaluate.py --configs './cfgs/celeba.ini' --input_dir '/Light_distangle/Data/occface/data/celeba512' --output_dir './faceresults/deface/celeba512' --ckpt_dir './ckpts/celeba512' >result.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 evaluate.py --configs './cfgs/vox2_seq.ini' --input_dir '/Light_distangle/Data/occface/data/voxceleb_seq' --output_dir './faceresults/deface/voxceleb_seq' --ckpt_dir './ckpts/voxcebe_seq' >result.txt 2>&1 &