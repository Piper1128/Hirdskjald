"""Indlæser SnakModel-7B i 4-bit (QLoRA-kompatibelt) og genererer skjald-svar.

Kører helt lokalt på GPU'en — intet internet efter modellen er hentet.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

from .persona import BASE_MODEL


class Skald:
    """Den vågne skjald: model + tokenizer, klar til at svare."""

    def __init__(self, adapter_dir=None):
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, device_map="auto")
        if adapter_dir and PeftModel is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def reply(self, messages, max_new_tokens=320, temperature=0.75, top_p=0.9, seed=""):
        text = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        text = text + seed  # 'prefill': tvinger svarets begyndelse -> låser stemme/spor
        enc = self.tok(text, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, repetition_penalty=1.1,
            no_repeat_ngram_size=4,  # bryder "en valkyrie, en valkyrie…"-løkker
            pad_token_id=self.tok.eos_token_id)
        gen = self.tok.decode(
            out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return (seed + gen).strip()
