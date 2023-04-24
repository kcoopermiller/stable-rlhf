## Training Process

1. Supervised fine-tuning of the base stableLM-3b model:
   `torchrun --nnodes 1  --nproc_per_node 8 supervised_finetuning.py --model_path=StabilityAI/stablelm-base-alpha-3b --streaming --no_gradient_checkpointing --learning_rate 1e-5 --max_steps 5000 --output_dir ./stable-se`

2. Reward modeling using the fine-tuned stable-3b-se:
   `torchrun --nnodes 1  --nproc_per_node 8 reward_modeling.py --model_name=<SE_MODEL>`

3. RL fine-tuning of stable-3b-se with the stable-3b-se-rm reward model:
   `accelerate launch --multi_gpu --num_machines 1  --num_processes 8 rl_training.py --log_with=wandb --model_name=<SE_MODEL> --reward_model_name=<SE_RM_MODEL> --adafactor=False --tokenizer_name=StabilityAI/stablelm-base-alpha-3b --save_freq=100 --output_max_length=128 --batch_size=8 --gradient_accumulation_steps=8 --batched_gen=True --ppo_epochs=4 --seed=0 --learning_rate=1.4e-5 --early_stopping=True --output_dir=stable-se-rl-finetune-128-8-8-1.4e-5_adam`
