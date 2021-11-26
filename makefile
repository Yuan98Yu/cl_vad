train_fe:
	CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 \
	train_feature_extractor.py \
	--net s3d --model infonce --moco-k 2048 \
	--dataset shtech --seq_len 32 --ds 1 --batch_size 16 \
	--epochs 100000 --schedule 250 280 -j 16 \
	--lr 3e-3 \
	--save_freq 1000 \
	--resume ./log/train/epoch89999.pth.tar \
	--log_path ./log/train

train_cl:
	CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 \
	train_one_class_classifier.py \
	--model lincls --network s3d --train_what last --objective soft-boundary \
	--nu 0.1 --R 0 \
	--dataset shtech --seq_len 32 --ds 1 --batch_size 16 \
	--epochs 300 --schedule 60 160 250 \
	--lr 3e-3 --warm_up_n_epochs 10 \
	--save_freq 10 \
	--pretrain ./log/train/epoch98000.pth.tar \
	--cfg_path ./log/train_cl/config.json \
	--log_path ./log/train_cl

test:
	python predict.py \
	--resume ./log/train_cl/epoch0.pth.tar \
	--cfg_path ./log/train_cl/config.json \
	--log_path ./log/test \
	--gpu 0