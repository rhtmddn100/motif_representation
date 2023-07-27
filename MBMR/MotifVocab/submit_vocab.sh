#!/bin/bash

#SBATCH --job-name=mbmr       # Submit a job named "example"
#SBATCH --partition=a6000        # 계산노드 종류 선택: a6000 or a100
#SBATCH --gres=gpu:0             # Use 1 GPU
#SBATCH --time=3-24:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=48000             # cpu memory size
#SBATCH --cpus-per-task=12        # cpu 개수
#SBATCH --output=log_uspto_vocab.txt         # 스크립트 실행 결과 std output을 저장할 파일 이름

ml purge
ml load cuda/10.2                # 필요한 쿠다 버전 로드
eval "$(conda shell.bash hook)"  # Initialize Conda Environment
conda activate mbmr             # Activate your conda environment

python src/motif_vocab_construction.py \
    --dataset USPTO-479k \
    --num_operations 1000 \
    --num_workers 16