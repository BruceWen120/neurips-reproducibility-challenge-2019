from typing import List
from bleu import *
import math
import pandas as pd
import fasttext
import numpy as np
from pathlib import Path
from tqdm import tqdm
from subprocess import PIPE, run

# LM for perplexity
model = None
tokenizer = None


def out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout

def get_perplexity(text: List[str], dataset: str):
    tmp_corpus = Path("test_corpus_tmp.txt")
    tmp_corpus.write_text("\n".join(text))
    output = out(f"/usr/share/srilm/bin/i686-m64/ngram -lm  "
                f"../srilm/{dataset}.corpus.lm -ppl  test_corpus_tmp.txt")

    ppl = float(output.split("ppl=")[1].split("ppl1")[0])
    ppl1 = float(output.split("ppl1=")[1])

    return ppl

def get_gpt2_perplexity(sentence):
    global model
    if model is None:
        from transformers import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
        import torch
        model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        model.eval()
        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss=model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss[0].item())


def evaluate(hyp: List[str], ref: List[str], dataset: str):
    print("Evaluating BLEU...")
    bleu, _ = corpus_bleu(hyp, [[x] for x in ref])
    bleu = bleu[0]
    print("\nBLEU: \n", bleu)

    # print("Evaluating PPL (hyp)...")
    # ppls = [get_gpt2_perplexity(s) for s in tqdm(hyp)]
    # ppl = pd.DataFrame(ppls).describe()
    # print("\nPPL: \n", ppl)
    #
    # print("Evaluating PPL (ref)...")
    # ppls_ref = [get_gpt2_perplexity(s) for s in tqdm(ref)]
    # ppl_ref = pd.DataFrame(ppls_ref).describe()
    # print("\nPPL (ref): \n", ppl_ref)

    print("Evaluating PPL (hyp)...")
    ppl = get_perplexity(hyp, dataset)
    print("\nPPL: \n", ppl)

    print("Evaluating PPL (ref)...")
    ppl_ref = get_perplexity(ref, dataset)
    print("\nPPL: \n", ppl_ref)

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

    # classifier for acc
    DATASET = "yelp"
    fasttext_model = fasttext.load_model(f"../fasttext/{DATASET}_model.bin")

    hyp = Path(hyp_dict[DATASET]).read_text().split("\n")
    hyp = [l for l in hyp if l != ""]
    ref = Path(ref_dict[DATASET]).read_text().split("\n")
    ref = [l for l in ref if l != ""]
    evaluate(hyp, ref, DATASET)