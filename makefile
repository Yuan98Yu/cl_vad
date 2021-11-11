train:
	CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 \
	train.py \
	--net s3d --model infonce --moco-k 2048 \
	--dataset shtech --seq_len 32 --ds 1 --batch_size 16 \
	--epochs 100000 --schedule 250 280 -j 16 \
	--lr 3e-3 \
	--save_freq 1000 \
	--resume ./log/train/epoch89999.pth.tar \
	--log_path ./log/train