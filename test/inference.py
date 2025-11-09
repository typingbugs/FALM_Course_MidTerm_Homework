import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype=torch.float16
    )
    model.eval()
    model.to('cuda')
    return model, tokenizer


def format_input_for_inference(german_text):
    prompt = f"Translate German to English.\nGerman: {german_text}\nEnglish:"
    return prompt


def load_test_data(test_file_path):
    test_data = []
    prompts = []
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            test_data.append({
                "input": data["de"],
                "reference": data["en"]
            })
            prompts.append(format_input_for_inference(data["de"]))

    return test_data, prompts

def save_results(results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def inference(model_path, test_file, output_file, batch_size=8, max_length=256):
    logger.info(f"Loading model and tokenizer from {model_path}")
    model, tokenizer = load_model(model_path)

    logger.info(f"Loading test data from {test_file}")
    test_data, prompts = load_test_data(test_file)

    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size), desc="Inference"):
            test_batch = test_data[i:i+batch_size]
            prompt_batch = prompts[i:i+batch_size]    

            # Tokenize
            tokenized_inputs = tokenizer(
                prompt_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            generate_config = GenerationConfig(
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Generate
            generated_ids = model.generate(
                input_ids=tokenized_inputs.input_ids.to(model.device),
                attention_mask=tokenized_inputs.attention_mask.to(model.device),
                generation_config=generate_config
            )

            for j in range(len(test_batch)):
                generated_text = tokenizer.decode(generated_ids[j], skip_special_tokens=True)
                prompt_text = prompt_batch[j]
                if prompt_text in generated_text:
                    generated_text = generated_text.split(prompt_text, 1)[1].strip()
                test_data[i + j]["prediction"] = generated_text
    
    save_results(test_data, output_file)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for MyModel translation")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--test_file", type=str, default="test/test.jsonl", help="Path to the test data file (JSONL format)")
    parser.add_argument("--output_file", type=str, default="results/inference_results.jsonl", help="Path to save the inference results (JSONL format)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    args = parser.parse_args()

    inference(
        model_path=args.model_path, 
        test_file=args.test_file,
        output_file=args.output_file,
        batch_size=args.batch_size
    )