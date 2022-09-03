CUDA_VISIBLE_DEVICES=0 python train.py --config configs/day2golden_fg.yaml  --save_name day2golden_fg
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/day2golden_fg.yaml  --save_name day2golden_fg_resume --resume outputs/day2golden_fg/ckpts
