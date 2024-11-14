CUDA_VISIBLE_DEVICES=3 python3 src/train.py \
	--algorithm sac  \
	--seed 0 \
	--eval_mode all \
	--domain_name walker \
	--task_name walk 
