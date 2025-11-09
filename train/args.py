from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from typing import Optional
from dataclasses import dataclass, field
import argparse
import yaml

@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
    )
    cache_dir: str = field(
        default=None,
    )
    num_proc: int = field(
        default=20,
    )

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
    )


def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    training_parser = HfArgumentParser((
        Seq2SeqTrainingArguments,
        DataArguments,
        ModelArguments
    ))
    training_args, data_args, model_args = training_parser.parse_dict(config)
    return training_args, data_args, model_args
