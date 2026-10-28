"""SFT-eksempler i køretidens format: én række = én anmodning fra en af de to faner + svaret.

Række-felter (data/sft_faner.jsonl):
    fane        "saga" | "grundet_saga" | "din_saga"
    identitet   {"navn","hjemstavn","historie"} — skjaldens faste identitet i eksemplet
    svar        teksten modellen skal generere EFTER prefill-seed'et
    saga:          emne
    grundet_saga:  emne, kilde, passages [{"title","text"}]
    din_saga:      navn, story, nye_bedrifter, nye_noter, foerste, alle_bedrifter

Prompten bygges med de samme persona.*_request-funktioner som app.py kalder og tokeniseres
med samme fmt.encode_prompt som Skald.reply. Tab beregnes kun på svar + EOS; seed'et er en
del af prompten (det gives også i køretid), så det maskeres.
"""
import json

from . import persona
from .fmt import encode_prompt

FANER = ("saga", "grundet_saga", "din_saga")


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def request_for(row):
    """(beskeder, seed) for en række — samme kald som app.py laver for den fane."""
    fane = row["fane"]
    if fane not in FANER:
        raise ValueError(f"ukendt fane: {fane!r}")
    persona.set_identity(row["identitet"])
    if fane == "saga":
        return persona.saga_request(row["emne"])
    if fane == "grundet_saga":
        return persona.grounded_saga_request(row["emne"], row["passages"], row["kilde"])
    return persona.din_saga_request(
        row["navn"], row.get("story", ""), row.get("nye_bedrifter", []),
        row.get("nye_noter", []), row.get("foerste", False), row.get("alle_bedrifter"))


def encode_example(tok, row, max_len):
    """(input_ids, labels) — eller None hvis eksemplet ikke kan være i max_len.

    Vi klipper ikke: et afklippet eksempel mister EOS (modellen lærer ikke at stoppe) eller
    hele svaret (intet tab). Hellere springe over og sige det.
    """
    msgs, seed = request_for(row)
    prompt_ids = encode_prompt(tok, msgs, seed)
    resp_ids = list(tok(row["svar"], add_special_tokens=False)["input_ids"]) + [tok.eos_token_id]
    ids = prompt_ids + resp_ids
    if len(ids) > max_len:
        return None
    labels = [-100] * len(prompt_ids) + resp_ids
    return ids, labels
