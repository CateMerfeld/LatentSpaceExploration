#!/bin/bash
#SBATCH --job-name=mnist-6-8-encoder
#SBATCH --partition=short
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"
MPLBACKEND=Agg /home/cmdunham/micromamba/envs/latent_space_exploration/bin/python mnist_encoder_experiment.py
