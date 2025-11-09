#! /bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=0

model_index=$1
ckpt_index=$2

python test/inference.py \
    --model_path="data/outputs/model_${model_index}/checkpoint-${ckpt_index}" \
    --test_file="data/iwslt2017-en-de/test.jsonl" \
    --output_file="data/results/model_${model_index}/checkpoint-${ckpt_index}/inference_results.jsonl" \
    --batch_size=256


python test/evaluation.py \
    --test_file="data/results/model_${model_index}/checkpoint-${ckpt_index}/inference_results.jsonl" \
    --output_file="data/results/model_${model_index}/checkpoint-${ckpt_index}/evaluation_results.jsonl"