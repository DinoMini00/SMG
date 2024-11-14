CUDA_VISIBLE_DEVICES=1 python3 src/train.py \
	--algorithm srm  \
	--seed 0 \
	--eval_mode all \
	--domain_name walker \
	--task_name walk \
	--action_repeat 2