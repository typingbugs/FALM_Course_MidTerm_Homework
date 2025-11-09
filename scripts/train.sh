#! /bin/bash
set -ex

log_dir=logs
mkdir -p $log_dir

python -m train.train \
    --config train_configs/train_2.yaml \
    2>&1 | tee ${log_dir}/train_2.log