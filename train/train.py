import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import logging
from datasets import Dataset
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorForSeq2Seq, AutoConfig
from train.args import get_train_args

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_model(model_args):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_config(config=config)
    tokenizer.model_max_length = config.max_position_embeddings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    model.config.vocab_size = len(tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def load_and_preprocess_dataset(data_args, tokenizer):
    dataset = {"train": {"de": [], "en": []}, "validation": {"de": [], "en": []}}
    for split in dataset.keys():
        data_file = os.path.join(data_args.data_path, f"{split}.jsonl")
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line.strip())
                dataset[split]["de"].append(example["de"])
                dataset[split]["en"].append(example["en"])
    dataset_train = Dataset.from_dict(dataset["train"])
    dataset_validation = Dataset.from_dict(dataset["validation"])

    def preprocess_function(example):
        prompt = f"Translate German to English.\nGerman: {example['de']}\nEnglish:"
        full_text = f"{prompt} {example['en']}"
        model_inputs = tokenizer(full_text, truncation=True, max_length=tokenizer.model_max_length, padding=False)
        input_ids = model_inputs["input_ids"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        start_of_target = len(prompt_ids)
        labels = [-100] * start_of_target + input_ids[start_of_target:]
        model_inputs["labels"] = labels
        return model_inputs

    processed_train = dataset_train.map(
        preprocess_function,
        num_proc=data_args.num_proc,
        remove_columns=['en', 'de'],
        load_from_cache_file=False,
        cache_file_name=os.path.join(data_args.cache_dir, "processed_train.arrow"),
        desc="Running tokenizer on train dataset",
    )

    processed_validation = dataset_validation.map(
        preprocess_function,
        num_proc=data_args.num_proc,
        remove_columns=['en', 'de'],
        load_from_cache_file=False,
        cache_file_name=os.path.join(data_args.cache_dir, "processed_validation.arrow"),
        desc="Running tokenizer on validation dataset",
    )

    return processed_train, processed_validation


def train(training_args, data_args, model_args):
    logger.info(f"loading model from {model_args.model_name_or_path}")
    model, tokenizer = load_model(model_args)

    logger.info(f"loading and preprocessing dataset from {data_args.data_path}")
    train_dataset, val_dataset = load_and_preprocess_dataset(data_args, tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    logger.info(f"start training")

    bleu_metric = evaluate.load("bleu")
    def computer_metric(eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        bleu_result = bleu_metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
        return {"bleu": bleu_result["bleu"]}

    trainer = Trainer(
        model=model,
        args=training_args, 
        train_dataset=train_dataset if training_args.do_train else None, 
        eval_dataset=val_dataset if training_args.do_eval else None, 
        data_collator=data_collator,
        compute_metrics=computer_metric,
    )
    trainer.train()

if __name__ == "__main__":
    training_args, data_args, model_args = get_train_args()
    train(training_args, data_args, model_args)
    