"""
finetune.py — VALGFRIT: LoRA-finpuds af skjalden (4-bit QLoRA på SnakModel-7B).

Træner på data/sft_faner.jsonl: eksempler i PRÆCIS det format de to faner sender —
samme system-prompt med identitet, samme anker-vekslinger, samme saga-/kapitel-bid og
samme prefill-seed (se skald/sft.py). Tab kun på det skjalden selv skriver efter seed'et.

    python finetune.py --epochs 3 --out adapter
    python app.py --adapter adapter

Kræver CUDA + bitsandbytes. Samme LoRA-idé som SkjaldSvar, nu i 4-bit på en 7B.
"""
import argparse
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skald.persona import BASE_MODEL
from skald import sft

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_faner.jsonl")
    ap.add_argument("--out", default="adapter")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)  # LoRA-A initialiseres tilfældigt -> reproducerbar kørsel

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

    rows = sft.load_rows(args.data)
    ex, skipped = [], []
    for r in rows:
        e = sft.encode_example(tok, r, args.max_len)
        if e is None:
            skipped.append(r["fane"])
        else:
            ex.append(e)
    per_fane = {f: sum(1 for r in rows if r["fane"] == f) for f in sft.FANER}
    n_tok = sum(len(i) for i, _ in ex)
    n_lab = sum(sum(1 for t in lab if t != -100) for _, lab in ex)
    print(f"{len(ex)} træningseksempler fra {args.data}  {per_fane}")
    print(f"  {n_tok:,} tokens i alt, heraf {n_lab:,} med tab (svar + EOS)")
    if skipped:
        print(f"  SPRUNGET OVER (længere end --max-len {args.max_len}): {skipped}")
    print()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    rng = random.Random(args.seed)
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
