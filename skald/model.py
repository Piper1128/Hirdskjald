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
            pad_token_id=self.tok.eos_token_id)
        gen = self.tok.decode(
            out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return (seed + gen).strip()

    @torch.no_grad()
    def extract_facts(self, user_text):
        """Fisk nye kendsgerninger om brugeren ud af hans besked (til hukommelsen).

        Anti-hallucination: et udtrukket faktum accepteres kun hvis det er bygget af
        brugerens EGNE ord — bringer det mange nye ord ind, er det lore modellen fandt
        på, og vi kasserer det.
        """
        from .persona import extract_messages
        raw = self.reply(extract_messages(user_text),
                         max_new_tokens=100, temperature=0.15, top_p=0.9)

        def content(s):
            return {w.lower().strip(".,!?:;\"'()") for w in s.split() if len(w) > 3}

        user_words = content(user_text)
        facts = []
        for line in raw.splitlines():
            s = line.strip(" -•*·\t")
            if not s or s.upper() == "INTET" or not (2 < len(s) < 160):
                continue
            new_words = content(s) - user_words
            if len(new_words) <= 2:  # næsten kun brugerens egne ord -> ægte (lore afvises)
                facts.append(s)
        return facts
