CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
	--algorithm sgsac  \
	--seed 0 \
	--eval_mode all \
	--domain_name walker \
	--task_name walk \
	--sgqn_quantile 0.98 \
	--action_repeat 2
