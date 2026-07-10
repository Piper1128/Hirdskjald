"""
Hirdskjald — din personlige vikinge-skjald, kørende helt lokalt på din GPU.

    python app.py                 # start serveren, browseren åbner selv
    python app.py --adapter adapter   # med et valgfrit LoRA-finpuds-lag

Snak med skjalden, bed om en saga, eller lad ham digte DIN egen saga. Han husker
dig mellem sessioner i memory.json (lokalt, offline) — og lærer nyt om dig undervejs.
"""
import argparse
import os
import re
import sys

from flask import Flask, request, jsonify, render_template, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skald.model import Skald
from skald.memory import Memory
from skald import persona

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
mem = Memory(os.path.join(HERE, "memory.json"))
skald = None  # indlæses ved opstart (main)

# "jeg hedder X" / "mit navn er X" / "kald mig X"
_NAME_RE = re.compile(
    r"\b(?:jeg hedder|mit navn er|kald mig)\s+"
    r"([^\s,.!?]{2,25}(?:\s+[A-ZÆØÅ][^\s,.!?]{1,24}){0,2})",
    re.IGNORECASE)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(HERE, "skjald.ico", mimetype="image/x-icon")


@app.route("/api/chat", methods=["POST"])
def chat():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="tom besked"), 400
    msgs = persona.chat_messages(text, mem.context(), mem.recent())
    svar = skald.reply(msgs, max_new_tokens=384)
    mem.add_turn(text, svar)

    # Lær nyt om herren af hans egne ord (navne-fangst + LLM-udtræk)
    learned = []
    m = _NAME_RE.search(text)
    if m and not mem.data["navn"]:
        mem.set_navn(m.group(1).strip(" .,!?"))
        learned.append("dit navn: " + mem.data["navn"])
    # Udtræk KUN fra udsagn, ikke fra rene spørgsmål (undgår forurening)
    if not text.rstrip().endswith("?"):
        for f in skald.extract_facts(text):
            mm = re.match(r"\s*(?:navn|tilnavn|hedder)\s*[:\- ]\s*(.+)", f, re.IGNORECASE)
            if mm and not mem.data["navn"]:
                mem.set_navn(mm.group(1).strip(" .,"))
                learned.append("dit navn: " + mem.data["navn"])
                continue
            n = len(mem.data["fakta"])
            mem.add_fakta(f)
            if len(mem.data["fakta"]) > n:
                learned.append(f)
    return jsonify(svar=svar, learned=learned)


@app.route("/api/saga", methods=["POST"])
def saga():
    emne = (request.get_json(silent=True) or {}).get("emne", "").strip()
    # Generisk saga = om EMNET, ikke om herren (personlig hukommelse kun i Snak/Min saga).
    msgs = persona.saga_messages(emne)
    return jsonify(svar=skald.reply(msgs, max_new_tokens=900))


@app.route("/api/minsaga", methods=["POST"])
def minsaga():
    msgs = persona.personal_saga_messages(
        mem.data["navn"], mem.data["bedrifter"], mem.context())
    return jsonify(svar=skald.reply(msgs, max_new_tokens=900))


@app.route("/api/memory", methods=["GET", "POST"])
def memory():
    if request.method == "POST":
        d = request.get_json(silent=True) or {}
        if "navn" in d:
            mem.set_navn(d["navn"])
        if d.get("bedrift"):
            mem.add_bedrift(d["bedrift"])
        if d.get("fakta"):
            mem.add_fakta(d["fakta"])
        if d.get("remove_bedrift"):
            mem.remove_bedrift(d["remove_bedrift"])
        if d.get("remove_fakta"):
            mem.remove_fakta(d["remove_fakta"])
        if d.get("clear"):
            mem.clear()
    return jsonify(navn=mem.data["navn"], bedrifter=mem.data["bedrifter"],
                   fakta=mem.data["fakta"], historik=mem.recent(8))


def main():
    global skald
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="valgfri LoRA-adapter (finpuds)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true",
                    help="åbn ikke browseren automatisk")
    args = ap.parse_args()
    print("Vækker Hirdskjald (indlæser SnakModel-7B i 4-bit)... det tager et øjeblik.")
    skald = Skald(adapter_dir=args.adapter)
    url = f"http://127.0.0.1:{args.port}"
    print(f"\nSkjalden er vågen. Åbn  {url}\n")
    if not args.no_browser:
        # Browseren åbnes lige efter serveren er bundet (modellen er alt indlæst nu).
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
