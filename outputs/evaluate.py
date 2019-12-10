from typing import List
from bleu import *
import math
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
import torch
import pandas as pd
import fasttext
import numpy as np
from pathlib import Path
from tqdm import tqdm


# LM for perplexity
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

# classifier for acc
fasttext_model = fasttext.load_model(f"../fasttext/{DATASET}_model.bin")

def get_perplexity(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss=model(tensor_input, labels=tensor_input)
    return math.exp(loss[0].item())


def evaluate(hyp: List[str], ref: List[str]):
    print("Evaluating BLEU...")
    bleu, _ = corpus_bleu(hyp, [[x] for x in ref])
    bleu = bleu[0]
    print("\nBLEU: \n", bleu)

    print("Evaluating PPL (hyp)...")
    ppls = [get_perplexity(s) for s in tqdm(hyp)]
    ppl = pd.DataFrame(ppls).describe()
    print("\nPPL: \n", ppl)

    print("Evaluating PPL (ref)...")
    ppls_ref = [get_perplexity(s) for s in tqdm(ref)]
    ppl_ref = pd.DataFrame(ppls_ref).describe()
    print("\nPPL (ref): \n", ppl_ref)

    print("Evaluating ACC...")
    labels = ["__label__pos" for _ in range(500)] + ["__label__neg" for _ in range(500)]
    pred_human = [fasttext_model.predict(l)[0][0] for l in ref]
    pred_model = [fasttext_model.predict(l)[0][0] for l in hyp]
    human_correct = [1 if pred == true else 0 for pred,true in zip(pred_human, labels)]
    model_correct = [1 if pred == true else 0 for pred,true in zip(pred_model, labels)]
    model_same_as_human = [1 if pred == true else 0 for pred,true in zip(pred_model, pred_human)]
    human_acc = np.sum(human_correct) / len(human_correct)
    model_acc = np.sum(model_correct) / len(model_correct)
    model_same_as_human_acc = np.sum(model_same_as_human) / len(model_same_as_human)

    print("\nACC1 (pred on human = labels): \n", human_acc)
    print("\nACC2 (pred on model = labels): \n", model_acc)
    print("\nACC3 (pred on model = pred on human): \n", model_same_as_human_acc)

DATASET = "yelp"

if __name__ == '__main__':
    hyp_dict = {
        "yelp": "my-model-yelp1.txt",
        "amazon": "my-model-amazon.txt",
        "imagecaption": "my-model-captions.txt"
    }
    ref_dict = {
        "yelp": "human-yelp.txt",
        "amazon": "human-amazon.txt",
        "imagecaption": "human-captions.txt"
    }
    hyp = Path().read_text(hyp_dict[DATASET]).split("\n")
    hyp = [l for l in hyp if l != ""]
    ref = Path(ref_dict[DATASET]).read_text().split("\n")
    ref = [l for l in ref if l != ""]
    evaluate(hyp, ref)