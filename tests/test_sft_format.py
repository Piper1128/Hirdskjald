"""
test_sft_format.py — træningsprompten SKAL være den prompt køretiden sender.

Fund 2026-10-28: finetune.py trænede på [bar SYSTEM_PROMPT, Q&A], mens køretid sendte
system+identitet, anker-vekslinger, saga-/kapitel-bid og et prefill-seed. Adapteren
lærte en ramme den aldrig så. Denne test fanger det, hvis de to glider fra hinanden igen.

Kører uden GPU, uden model og uden transformers: en lille falsk tokenizer gør
chat-skabelon og tekst -> id'er deterministisk og synlig.

    python tests/test_sft_format.py      (eller: pytest tests/)
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skald import persona, sft  # noqa: E402
from skald.fmt import encode_prompt  # noqa: E402
from skald.knowledge import medieval_source  # noqa: E402

DATA = os.path.join(ROOT, "data", "sft_faner.jsonl")


class FakeTok:
    """Tegn-niveau tokenizer med BOS/EOS og en synlig chat-skabelon."""
    bos_token_id, eos_token_id = 1, 2

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert tokenize is False
        text = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        return text + ("<|assistant|>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=True):
        ids = [ord(c) + 10 for c in text]
        return {"input_ids": ([self.bos_token_id] if add_special_tokens else []) + ids}


def runtime_request(row):
    """(beskeder, seed) som app.py laver dem for rækkens fane."""
    persona.set_identity(row["identitet"])
    if row["fane"] == "saga":
        return persona.saga_request(row["emne"])
    if row["fane"] == "grundet_saga":
        return persona.grounded_saga_request(
            row["emne"], row["passages"], medieval_source(row["emne"]))
    return persona.din_saga_request(
        row["navn"], row["story"], row["nye_bedrifter"], row["nye_noter"],
        row["foerste"], row["alle_bedrifter"])


def test_training_prompt_is_runtime_prompt():
    tok = FakeTok()
    rows = sft.load_rows(DATA)
    assert {r["fane"] for r in rows} == set(sft.FANER), "hver fane skal have eksempler"
    for i, row in enumerate(rows):
        persona.set_identity(None)  # sft skal selv sætte rækkens identitet
        ids, labels = sft.encode_example(tok, row, max_len=10**9)
        msgs, seed = runtime_request(row)
        expected = encode_prompt(tok, msgs, seed)
        n = len(expected)
        assert ids[:n] == expected, f"række {i} ({row['fane']}): træningsprompt != køretidsprompt"
        assert all(t == -100 for t in labels[:n]), f"række {i}: prompt/seed må ikke give tab"
        resp = tok(row["svar"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        assert ids[n:] == resp and labels[n:] == resp, f"række {i}: tab skal være svar + EOS"
        # Prompten skal bære det køretiden bærer: identitet og anker-vekslinger.
        system = msgs[0]["content"]
        assert row["identitet"]["navn"] in system, f"række {i}: identitet mangler i system"
        assert msgs[1:1 + len(persona._ANCHOR)] == persona._ANCHOR, f"række {i}: anker mangler"


def test_too_long_example_is_skipped_not_clipped():
    tok = FakeTok()
    row = sft.load_rows(DATA)[0]
    assert sft.encode_example(tok, row, max_len=50) is None


def _calls(path):
    tree = ast.parse(open(os.path.join(ROOT, path), encoding="utf-8").read())
    return {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
            for n in ast.walk(tree) if isinstance(n, ast.Call)}


def test_app_and_model_use_the_shared_builders():
    """app.py og Skald.reply skal gå gennem de delte funktioner, ikke egne kopier."""
    app = _calls("app.py")
    for f in ("saga_request", "grounded_saga_request", "din_saga_request"):
        assert f in app, f"app.py kalder ikke persona.{f}"
    for f in ("saga_messages", "grounded_saga_messages", "din_saga_chapter_messages"):
        assert f not in app, f"app.py bygger selv beskeder med {f} — brug *_request"
    model = _calls(os.path.join("skald", "model.py"))
    assert "encode_prompt" in model and "apply_chat_template" not in model, \
        "Skald.reply skal tokenisere via fmt.encode_prompt"
    ft = _calls("finetune.py")
    assert "encode_example" in ft and "apply_chat_template" not in ft, \
        "finetune.py skal kode eksempler via sft.encode_example"


if __name__ == "__main__":
    test_training_prompt_is_runtime_prompt()
    test_too_long_example_is_skipped_not_clipped()
    test_app_and_model_use_the_shared_builders()
    print("OK — træningsprompt == køretidsprompt for alle eksempler")
