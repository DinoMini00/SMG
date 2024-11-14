CUDA_VISIBLE_DEVICES=2 python3 src/train.py \
	--algorithm drq \
	--seed 0 \
	--eval_mode all \
	--domain_name walker \
	--task_name walk 
