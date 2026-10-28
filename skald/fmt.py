"""Prompt -> token-id'er. Én funktion, brugt af BÅDE generering (model.py) og træning (sft.py).

Køretid: chat-skabelonen med add_generation_prompt=True, plus et prefill-seed bagefter, så
svarets første ord er givet. Tokeniseres med tokenizerens standard special tokens (som
tok(text) altid har gjort i Skald.reply). Trænes adapteren på en anden ramme end denne,
lærer den noget den aldrig ser — derfor findes funktionen kun ét sted.

Ingen torch-import her: testene kører uden GPU og uden model.
"""


def encode_prompt(tok, messages, seed=""):
    """Returnér prompt-token-id'erne præcis som modellen ser dem før første genererede token."""
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return list(tok(text + seed)["input_ids"])
