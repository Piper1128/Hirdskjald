"""
finetune.py — VALGFRIT: LoRA-finpuds af skjalden (4-bit QLoRA på SnakModel-7B).

Base-modellen taler allerede fremragende skjald-dansk, så dette er finpuds: det
låser persona og saga-adfærd fastere. Læser data/skald_sft.jsonl og gemmer en adapter.

    python finetune.py --epochs 3 --out adapter
    python app.py --adapter adapter

Kræver CUDA + bitsandbytes. Samme LoRA-idé som SkjaldSvar, nu i 4-bit på en 7B.
"""
import argparse
import json
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skald.persona import BASE_MODEL, SYSTEM_PROMPT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode(tok, instr, resp, max_len):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instr}]
    ptext = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ftext = tok.apply_chat_template(
        msgs + [{"role": "assistant", "content": resp}],
        add_generation_prompt=False, tokenize=False)
    prompt_ids = tok(ptext, add_special_tokens=False)["input_ids"]
    full_ids = tok(ftext, add_special_tokens=False)["input_ids"]
    labels = list(full_ids)
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100
    return full_ids[:max_len], labels[:max_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/skald_sft.jsonl")
    ap.add_argument("--out", default="adapter")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        sys.exit("QLoRA kræver CUDA.")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    print(f"Indlæser {BASE_MODEL} i 4-bit ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ex = [encode(tok, r["instruction"], r["response"], args.max_len)
          for r in load_rows(args.data)]
    print(f"{len(ex)} træningseksempler fra {args.data}\n")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    rng = random.Random(0)
    model.train()
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        order = list(range(len(ex)))
        rng.shuffle(order)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for j, idx in enumerate(order):
            ids, lab = ex[idx]
            loss = model(input_ids=torch.tensor([ids], device=device),
                         labels=torch.tensor([lab], device=device)).loss
            (loss / args.grad_accum).backward()
            tot += loss.item()
            if (j + 1) % args.grad_accum == 0 or (j + 1) == len(order):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        print(f"epoch {ep}/{args.epochs}  loss {tot/len(order):.3f}  ({time.time()-t0:.0f}s)")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\nGemte adapter i {args.out}/ — kør: python app.py --adapter {args.out}")


if __name__ == "__main__":
    main()
