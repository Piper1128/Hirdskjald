"""
Hirdskjald — din personlige vikinge-skjald, kørende helt lokalt på din GPU.

    python app.py                      # start serveren, browseren åbner selv
    python app.py --adapter adapter    # med et valgfrit LoRA-finpuds-lag

To faner, to formål:
  * Saga-fortælling — bed om en saga; skjalden ringer den rigtige ind med 1-2 sokratiske
    spørgsmål og kvæder så den ægte saga, grundet i viden-basen, i sin faste persona.
  * Din saga — giv navn, bedrifter og fri tekst; skjalden bygger din egen saga op
    kapitel for kapitel over tid, mens han lærer din historie bedre.
"""
import argparse
import os
import random
import re
import sys

from flask import Flask, request, jsonify, render_template, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skald.model import Skald
from skald.memory import Memory
from skald.knowledge import Knowledge, medieval_source
from skald.embeddings import Embedder
from skald import persona

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
mem = Memory(os.path.join(HERE, "memory.json"))
knowledge = None  # bygges ved opstart (main) med semantiske embeddings
skald = None      # indlæses ved opstart (main)
_RNG = random.Random()

# Emne nævnt direkte ("fortæl en saga om Ragnarok") -> spring det sokratiske over
_EMNE_RE = re.compile(r"\b(?:saga|kvad|vers|kvæd|digt|fortæl|hør)\b.*?\b(?:om|over)\s+(.+)",
                      re.IGNORECASE)
# "ny saga"/"forfra" bryder ud af en igangværende sokratisk runde
_NYSAGA_RE = re.compile(r"\bny saga\b|forfra|start (?:om )?igen|noget andet", re.IGNORECASE)


def _ensure_identity():
    """Giv skjalden en fast identitet (navn/hjemstavn/forhistorie) hvis han ikke har en endnu."""
    if not mem.data.get("skjald"):
        mem.set_skjald(persona.new_identity(_RNG))
    persona.set_identity(mem.data["skjald"])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(HERE, "skjald.ico", mimetype="image/x-icon")


# =============================== Fane 1: Saga-fortælling ===============================

def _emne(text):
    m = _EMNE_RE.search(text)
    return m.group(1).strip(" .!?\"'") if m else ""


def _tell_saga(query):
    """Hent den bedst matchende ÆGTE saga fra viden-basen og kvæd den — grundet + annonceret."""
    passages = knowledge.retrieve(query, k=4) if knowledge else []
    if not passages:
        msgs = persona.saga_messages(query)
        svar = skald.reply(msgs, max_new_tokens=900, temperature=0.8,
                           seed=f"Hør nu, herre — hør sagaen om {query}: ")
        return jsonify(svar=svar)
    primary = passages[0][1]["title"]
    skrift = medieval_source(primary)
    chunks = [c for _, c in passages if c["title"] == primary][:3] or [passages[0][1]]
    msgs = persona.grounded_saga_messages(primary, chunks, skrift)
    svar = skald.reply(
        msgs, max_new_tokens=900, temperature=0.8,
        seed=f"Hør nu, herre — jeg kvæder sagaen om {primary}, som den kendes fra {skrift}: ")
    return jsonify(svar=svar, skrift=skrift, kilder=[primary])


@app.route("/api/fortael", methods=["POST"])
def fortael():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="tom besked"), 400
    _ensure_identity()

    if _NYSAGA_RE.search(text):
        mem.flow_reset()

    flow = mem.data["saga_flow"]

    # 1) Emne nævnt direkte -> kvæd den saga med det samme
    emne = _emne(text)
    if emne and not flow["aktiv"]:
        mem.flow_reset()
        return _tell_saga(emne)

    # 2) Midt i en sokratisk runde -> dette er herrens svar
    if flow["aktiv"]:
        mem.flow_answer(text)
        svar_liste = mem.data["saga_flow"]["svar"]
        if mem.data["saga_flow"]["runde"] < 2 and len("".join(svar_liste).split()) < 3:
            mem.flow_set_runde(2)
            return jsonify(svar=persona.socratic_question(2, svar_liste), sporger=True)
        query = " ".join(svar_liste)
        mem.flow_reset()
        return _tell_saga(query)

    # 3) Bar saga-anmodning uden emne -> begynd sokratisk
    mem.flow_begin()
    return jsonify(svar=persona.socratic_question(1, []), sporger=True)


# =============================== Fane 2: Din saga ===============================

def _titel():
    return (mem.data["navn"] + "s saga") if mem.data["navn"] else "Din saga"


def _dinsaga_state():
    return jsonify(navn=mem.data["navn"], titel=_titel(),
                   bedrifter=mem.data["bedrifter"], fakta=mem.data["fakta"],
                   kapitler=mem.data["min_saga"]["kapitler"])


@app.route("/api/dinsaga", methods=["GET", "POST"])
def dinsaga():
    if request.method == "GET":
        return _dinsaga_state()

    d = request.get_json(silent=True) or {}
    handling = d.get("handling")

    if handling == "gem":  # gem navn / bedrift / fri-tekst-info
        if d.get("navn"):
            mem.set_navn(d["navn"].strip())
        if d.get("bedrift"):
            mem.add_bedrift(d["bedrift"].strip())
        if d.get("info"):
            mem.add_fakta(d["info"].strip())
        return _dinsaga_state()

    if handling == "nulstil":
        mem.clear_min_saga()
        return _dinsaga_state()

    if handling == "kvaed":  # kvæd næste kapitel af din saga
        _ensure_identity()
        navn = mem.data["navn"] or "den navnløse"
        ms = mem.data["min_saga"]
        nye_bedrifter = mem.data["bedrifter"][ms["brugt_bedrifter"]:]
        nye_noter = mem.data["fakta"][ms["brugt_fakta"]:]
        if not ms["kapitler"] and not mem.data["navn"] and not nye_bedrifter and not nye_noter:
            return jsonify(fejl="Fortæl mig først hvem du er, herre — dit navn og en dåd — "
                                "så kan jeg kvæde.")
        foerste = not ms["kapitler"]
        story = "\n\n".join(ms["kapitler"])[-1200:]  # kort "sagaen indtil nu" til sammenhæng
        msgs = persona.din_saga_chapter_messages(
            navn, story, nye_bedrifter, nye_noter, foerste, mem.data["bedrifter"])
        seed = (f"Hør, herre! Her tager {navn}s saga sin begyndelse. " if foerste
                else f"Og videre gik {navn}s saga. ")
        kapitel = skald.reply(msgs, max_new_tokens=460, temperature=0.8, seed=seed)
        mem.min_saga_add(kapitel)
        return jsonify(kapitel=kapitel, kapitler=mem.data["min_saga"]["kapitler"], titel=_titel())

    return jsonify(error="ukendt handling"), 400


def main():
    global skald, knowledge
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="valgfri LoRA-adapter (finpuds)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true", help="åbn ikke browseren automatisk")
    args = ap.parse_args()
    print("Vækker Hirdskjald (indlæser SnakModel-7B i 4-bit)... det tager et øjeblik.")
    skald = Skald(adapter_dir=args.adapter)
    kpath = os.path.join(HERE, "data", "knowledge.jsonl")
    if os.path.exists(kpath):
        print("Bygger viden-base (semantiske embeddings)...")
        knowledge = Knowledge(kpath, embedder=Embedder())
        print(f"Viden-base: {knowledge.N} stykker (semantisk RAG aktiv).")
    url = f"http://127.0.0.1:{args.port}"
    print(f"\nSkjalden er vågen. Åbn  {url}\n")
    if not args.no_browser:
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
