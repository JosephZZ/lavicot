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
  --config experiments/configs/default.yaml
