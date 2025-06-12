#!/bin/bash
#SBATCH -J testlavicot           # 作业名是 test
#SBATCH -p HGX              # 提交到 cpu 分区
#SBATCH -N 1                # 使用一个节点
#SBATCH --cpus-per-task=64   # 每个进程占用一个 cpu 核心
#SBATCH -t 1-00:00:00              # 任务最大运行时间是 5 分钟
#SBATCH --qos=lv0b
#SBATCH -o log_testlavicot.out         # 将屏幕的输出结果保存到当前文件夹的 test.out
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
source /scratch/zhouziheng/miniconda3/bin/activate
conda activate lavicot
which python
pwd

cd /scratch/zhouziheng/lavicot
pwd

python scripts/train.py \
  --config experiments/configs/default.yaml \
  --base_output_dir experiments/results/ \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --resume_checkpoint_path null \
  --dataset_config_name metamath \
  --num_train_epochs 100 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --max_length 512 \
  --warmup_steps 1000 \
  --save_steps 5000 \
  --eval_steps 5000 \
  --seen_token_weight 1.0 \
  --unseen_token_weight 1.0 \
  --num_rounds 2 \
  --stochastic_rounds true \
  --first_round_question_only true \
  --use_previous_prefix true \
  --proportion_min 0.1 \
  --proportion_max 0.9 \
  --prefix_generator.layer_selection_mode all \
  --prefix_generator.max_iterations 7 \
  --prefix_generator.gradient_steps 4 \
  --prefix_generator.shared_weight_for_all_layers true \
  --prefix_generator.use_hooks_during_prefix_update false
