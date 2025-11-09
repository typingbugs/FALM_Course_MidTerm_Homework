import json
from evaluate import load
import os
from tqdm import tqdm

bleu = load("bleu")

def compute_metrics(references, predictions):
    bleu_result = bleu.compute(predictions=predictions, references=references)
    return {"bleu": bleu_result["bleu"]}


def save_results(results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def compute_bleu_metric(test_file, output_file):
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            test_data.append(data)
    for i, data in tqdm(enumerate(test_data), total=len(test_data), desc="Computing BLEU scores"):
        bleu_score = compute_metrics(
            references=[[data["reference"]]],
            predictions=[data["prediction"]]
        )["bleu"]
        test_data[i]["bleu"] = bleu_score

    average_bleu = sum(item["bleu"] for item in test_data) / len(test_data)
    save_results(test_data, output_file)
    print(f"Average BLEU score: {average_bleu:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test file with references.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the results with predictions and BLEU scores.")
    args = parser.parse_args()

    compute_bleu_metric(args.test_file, args.output_file)
    

    