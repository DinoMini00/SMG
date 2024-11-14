CUDA_VISIBLE_DEVICES=1 python3 src/train.py \
	--algorithm soda  \
	--seed 0 \
	--eval_mode all \
	--domain_name walker \
	--task_name walk
