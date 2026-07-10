"""
Hirdskjald — din personlige vikinge-skjald, kørende helt lokalt på din GPU.

    python app.py                 # start serveren, åbn http://127.0.0.1:5000
    python app.py --adapter adapter   # med et valgfrit LoRA-finpuds-lag

Snak med skjalden, bed om en saga, eller lad ham digte DIN egen saga. Han husker
dig mellem sessioner i memory.json (lokalt, offline).
"""
import argparse
import os
import sys

from flask import Flask, request, jsonify, render_template

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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="tom besked"), 400
    msgs = persona.chat_messages(text, mem.context(), mem.recent())
    svar = skald.reply(msgs)
    mem.add_turn(text, svar)
    return jsonify(svar=svar)


@app.route("/api/saga", methods=["POST"])
def saga():
    emne = (request.get_json(silent=True) or {}).get("emne", "").strip()
    # En generisk saga handler om EMNET, ikke om herren — så ingen hukommelse her
    # (personlig hukommelse bruges i Snak og Min saga).
    msgs = persona.saga_messages(emne)
    return jsonify(svar=skald.reply(msgs, max_new_tokens=420))


@app.route("/api/minsaga", methods=["POST"])
def minsaga():
    msgs = persona.personal_saga_messages(
        mem.data["navn"], mem.data["bedrifter"], mem.context())
    return jsonify(svar=skald.reply(msgs, max_new_tokens=480))


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
        if d.get("clear"):
            mem.clear()
    return jsonify(navn=mem.data["navn"], bedrifter=mem.data["bedrifter"],
                   fakta=mem.data["fakta"], historik=mem.recent(8))


def main():
    global skald
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="valgfri LoRA-adapter (finpuds)")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    print("Vækker Hirdskjald (indlæser SnakModel-7B i 4-bit)... det tager et øjeblik.")
    skald = Skald(adapter_dir=args.adapter)
    print(f"\nSkjalden er vågen. Åbn  http://127.0.0.1:{args.port}\n")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
