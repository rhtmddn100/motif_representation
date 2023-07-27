#!/bin/bash

#SBATCH --job-name=mbmr       # Submit a job named "example"
#SBATCH --partition=a6000        # 계산노드 종류 선택: a6000 or a100
#SBATCH --gres=gpu:0             # Use 1 GPU
#SBATCH --time=3-24:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=192G             # cpu memory size
#SBATCH --cpus-per-task=16        # cpu 개수
#SBATCH --output=log_USPTO-479k.txt         # 스크립트 실행 결과 std output을 저장할 파일 이름

ml purge
ml load cuda/10.2                # 필요한 쿠다 버전 로드
eval "$(conda shell.bash hook)"  # Initialize Conda Environment
conda activate mbmr             # Activate your conda environment

srun python src/merging_operation_learning.py \
    --dataset USPTO-479k \
    --num_iters 500 \
    --num_workers 4
