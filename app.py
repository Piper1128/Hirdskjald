"""
Hirdskjald — din personlige vikinge-skjald, kørende helt lokalt på din GPU.

    python app.py                 # start serveren, browseren åbner selv
    python app.py --adapter adapter   # med et valgfrit LoRA-finpuds-lag

Snak med skjalden, bed om en saga, eller lad ham digte DIN egen saga. Han husker
dig mellem sessioner i memory.json (lokalt, offline) — og lærer nyt om dig undervejs.
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
skald = None  # indlæses ved opstart (main)

# "jeg hedder X" / "mit navn er X" / "kald mig X" (navnet skal være stort-begyndende,
# så vi ikke opsluger " og jeg ..." — derfor case-sensitivt navn, kun triggeren tåler stort/lille)
_NAME_RE = re.compile(
    r"\b(?:[Jj]eg hedder|[Mm]it navn er|[Kk]ald mig)\s+"
    r"([A-ZÆØÅ][^\s,.!?]{1,24}(?:\s+[A-ZÆØÅ][^\s,.!?]{1,24}){0,2})")

# Intent-genkendelse til ét-vindues styring (siger man hvad man vil, gør skjalden det)
_FORGET_RE = re.compile(r"\bglem alt\b|\bglem mig\b|start forfra|nulstil", re.IGNORECASE)
_CONV_SAGA_RE = re.compile(
    r"vores samtale|vores snak|det vi (har )?talt|denne samtale|vores stund|af (vores|denne) snak",
    re.IGNORECASE)
_SELF_SAGA_RE = re.compile(r"\bom mig\b|mig selv|min (egen )?saga|mit eftermæle", re.IGNORECASE)
_SAGA_RE = re.compile(
    r"\ben saga\b|\bet kvad\b|\bet vers\b|\bkvæd\b|\bdigt\b|fortæl .{0,15}\bsaga\b|lav .{0,15}\bsaga\b",
    re.IGNORECASE)
_EMNE_RE = re.compile(r"\b(?:saga|kvad|vers|kvæd|digt|fortæl)\b.*?\b(?:om|over)\s+(.+)", re.IGNORECASE)
# Herren deler sin dag / sine følelser (godt eller skidt) -> mød ham med passende lore
_MOOD_RE = re.compile(
    r"\b(møgdag|lortedag|skidedag)\b|haft en .{0,25}\bdag\b|alt (gik|går) galt|orker ikke|"
    r"jeg (er|føler mig|har det) (så |rigtig |virkelig |ret )?"
    r"(ked|trist|nedtrykt|deprimeret|stresset|udkørt|udmattet|træt|frustreret|skuffet|ensom|"
    r"vred|sur|rasende|bange|glad|lykkelig|stolt|begejstret|lettet|elendig)|"
    r"\bjeg (græder|savner|hader)\b",
    re.IGNORECASE)


# Herren rykker ved rollen eller spoerger til nutids-ting (AI, kode, maskiner) -> skjalden afviser
# I-ROLLE og grundes IKKE (at hente Wikipedia til "er du en AI?" fik modellen til at bryde rollen).
_META_RE = re.compile(
    r"\b(?:ai|kunstig intelligens|sprogmodel|sprog-model|chatbot|chat-bot|bot|robot|"
    r"computer|maskine|program(?:met|mering)?|programmere|kode|kodning|python|javascript|"
    r"internet(?:tet)?|hjemmeside|app|wifi|elektricitet|telefon|smartphone)\b|"
    r"bryd(?:er)? rollen|hold op med at (?:spille|lade som)|lade som om du|"
    r"er du (?:egentlig |mon |virkelig |bare |en |et )*"
    r"(?:menneske|person|rigtig|ægte|levende|kunstig|et program)|"
    r"hvem (?:har )?(?:lavet|skabt|programmeret|kodet) dig",
    re.IGNORECASE)
# Hilsen eller "hvem er du / hvad kan du" -> fast i-rolle intro (ingen grunding, intet opdigt).
# Uden denne gik "hvem er du og hvad kan du" i grundet chat og opdigtede en falsk identitet.
_INTRO_RE = re.compile(
    r"\bhvem (?:er du|du er)\b|\bhvad (?:kan|laver) du\b|hvad kan du (?:hjælpe|gøre)|"
    r"introducer dig|præsent(?:er|ér) dig|hvad er (?:det her|dette)|"
    r"^(?:hej|halløj|goddag|hejsa|hallo|yo)\b",
    re.IGNORECASE)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(HERE, "skjald.ico", mimetype="image/x-icon")


@app.route("/api/nysamtale", methods=["POST"])
def nysamtale():
    """Ryd KUN samtalen (historik + resumé); behold hvem-du-er (navn/bedrifter/fakta)."""
    mem.clear_conversation()
    return jsonify(ok=True)


_RNG = random.Random()


def _ensure_identity():
    """Giv skjalden en fast identitet (navn/hjemstavn/forhistorie) hvis han ikke har en endnu.

    Fast pr. samtale, huskes i memory.json — så DIN skjald er din. Skifter kun ved 'glem alt'.
    """
    if not mem.data.get("skjald"):
        mem.set_skjald(persona.new_identity(_RNG))
    persona.set_identity(mem.data["skjald"])


# Interaktiv saga: start over flere ture + stop-ord der bryder ud af saga-tilstanden.
_ISAGA_RE = re.compile(
    r"saga (?:over|på|i) flere (?:ture|gange|dele|kapitler)|"
    r"(?:fortløbende|interaktiv|sammenhængende) saga|saga vi (?:bygger|laver|digter) sammen|"
    r"saga (?:jeg|vi) (?:kan )?styr|fortæl .{0,20}saga sammen|start en (?:fortløbende )?saga|"
    r"en saga jeg kan (?:styre|påvirke|guide)",
    re.IGNORECASE)
_STOP_SAGA_RE = re.compile(
    r"\bafslut\b|stop sagaen|slut(?:t)? sagaen|færdig med sagaen|luk sagaen|"
    r"det er nok|ikke mere saga|\bhold op\b",
    re.IGNORECASE)


def _maybe_summarize():
    """Fold ældre ture ind i et kort, EKSTRAKTIVT resumé (herrens egne emner), så konteksten
    aldrig sprænger 4096. Ekstraktivt — ikke LLM-genereret — fordi en 7B hallucinerer resuméer
    (den opdigtede "Herre: Jens Jensen"). En liste af hans egne ord kan ikke hallucineres."""
    if not mem.needs_summary():
        return
    turns, end = mem.to_summarize()
    olds = [(t.get("user") or "").strip()[:80] for t in turns]
    olds = [u for u in olds if u]
    if not olds:
        mem.set_resume(mem.data["resume"], end)
        return
    prev = mem.data["resume"]
    combined = "; ".join(([prev] if prev else []) + olds)
    mem.set_resume(combined[-900:], end)  # hold resuméet kort


@app.route("/api/chat", methods=["POST"])
def chat():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="tom besked"), 400

    _ensure_identity()  # skjalden bærer en fast identitet (navn/hjemstavn) gennem samtalen

    # --- Intent-styring: skjalden gør det du beder om ---
    if _FORGET_RE.search(text):
        mem.clear()
        persona.set_identity(None)  # glem alt -> en ny skjald (nyt navn/hjemstavn) træder frem
        return jsonify(svar="Det er glemt, herre. Jeg kender dig ikke mere — sæt dig og fortæl på ny.")

    # Rører nogen ved rollen eller nutiden -> afvis I-ROLLE, uden grunding (kilder ville forvirre)
    # --- Interaktiv saga i gang: herrens ord STYRER næste del (kun 'afslut sagaen' bryder ud) ---
    if mem.saga_active():
        if _STOP_SAGA_RE.search(text):
            msgs = persona.saga_end_messages(mem.data["saga"]["historie"], mem.context())
            svar = skald.reply(msgs, max_new_tokens=560, temperature=0.8,
                               seed="Og således fik sagaen sin ende. ")
            mem.saga_stop()
            mem.add_turn(text, svar)
            return jsonify(svar=svar)
        msgs = persona.saga_continue_messages(mem.data["saga"]["historie"], text, mem.context())
        svar = skald.reply(msgs, max_new_tokens=520, temperature=0.85,
                           seed="Og videre gik sagaen. ")
        mem.saga_advance(svar)
        mem.add_turn(text, svar)
        return jsonify(svar=svar, saga=True)

    if _META_RE.search(text):
        msgs = persona.modern_deflect_messages(text, mem.context())
        svar = skald.reply(msgs, max_new_tokens=200, temperature=0.65, seed="Nej, herre — ")
        mem.add_turn(text, svar)
        return jsonify(svar=svar)

    # --- Start en INTERAKTIV saga over flere ture (skjalden vælger optakt, du styrer siden) ---
    if _ISAGA_RE.search(text):
        msgs = persona.saga_start_messages(mem.context())
        svar = skald.reply(msgs, max_new_tokens=520, temperature=0.85,
                           seed="Hør, herre, en ny saga tager sin begyndelse. ")
        mem.saga_begin(svar)
        mem.add_turn(text, svar)
        return jsonify(svar=svar, saga=True)

    saga_req = bool(_SAGA_RE.search(text))

    if saga_req and _CONV_SAGA_RE.search(text):          # saga af vores samtale
        hist = mem.data["historik"]
        if not hist:
            return jsonify(svar="Vi har endnu ikke talt sammen, herre. Fortæl mig om dig, "
                                "så væver jeg din saga af vore ord.")
        msgs = persona.conversation_saga_messages(
            mem.data["navn"], hist, mem.data["bedrifter"], mem.context())
        svar = skald.reply(msgs, max_new_tokens=900, temperature=0.85)
        mem.add_turn(text, svar)
        return jsonify(svar=svar)

    if saga_req and _SELF_SAGA_RE.search(text):          # din egen saga
        msgs = persona.personal_saga_messages(
            mem.data["navn"], mem.data["bedrifter"], mem.context())
        svar = skald.reply(msgs, max_new_tokens=900)
        mem.add_turn(text, svar)
        return jsonify(svar=svar)

    if saga_req:                                          # saga om et emne
        m = _EMNE_RE.search(text)
        emne = m.group(1).strip(" .!?") if m else ""
        passages = knowledge.retrieve(emne or text, k=3) if knowledge else []
        skrift = ""
        if passages:
            skrift = medieval_source(passages[0][1]["title"])
            msgs = persona.grounded_saga_messages(
                emne or text, [c for _, c in passages], skrift)
        else:
            msgs = persona.saga_messages(emne)
        # prefill annoncerer sagaen OG tvinger fortællingen i gang (dræber meta-snak + titel-stubs)
        seed = (f"Hør nu, herre — jeg kvæder sagaen om {emne}: " if emne
                else "Hør nu, herre — hør denne saga: ")
        svar = skald.reply(msgs, max_new_tokens=900, temperature=0.8, seed=seed)
        mem.add_turn(text, svar)
        return jsonify(svar=svar, kilder=[c["title"] for _, c in passages], skrift=skrift)

    # --- Herren deler sin dag/følelse -> mød ham med et passende sagn/saga/vers ---
    if _MOOD_RE.search(text):
        _maybe_summarize()
        msgs = persona.comfort_messages(text, mem.context(),
                                        mem.injected_history(), mem.data["resume"])
        # prefill tvinger skjalden ind i lore-sporet fremfor moderne trøst
        svar = skald.reply(msgs, max_new_tokens=480, temperature=0.6,
                           seed="Hør et sagn, herre: ")
        mem.add_turn(text, svar)
        return jsonify(svar=svar)

    # --- Hilsen / "hvem er du, hvad kan du" -> fast i-rolle intro (ingen grunding, intet opdigt) ---
    if _INTRO_RE.search(text):
        svar = persona.intro_text()
        mem.add_turn(text, svar)
        return jsonify(svar=svar)

    # --- Almindelig samtale (default): RAG-grundet + lærer om dig ---
    _maybe_summarize()
    passages = knowledge.retrieve(text, k=3) if knowledge else []
    hist = mem.injected_history()
    if passages:
        msgs = persona.grounded_chat_messages(
            text, [c for _, c in passages], mem.context(), hist, mem.data["resume"])
    else:
        msgs = persona.chat_messages(text, mem.context(), hist, mem.data["resume"])
    # stemme-lås: en kort skjald-åbning prefill'es, så svaret ikke falder i leksikon-tone
    svar = skald.reply(msgs, max_new_tokens=512, temperature=0.7,
                       seed=persona.skald_opener(text))
    mem.add_turn(text, svar)

    learned = []
    m = _NAME_RE.search(text)
    if m and not mem.data["navn"]:
        mem.set_navn(m.group(1).strip(" .,!?"))
        learned.append("dit navn: " + mem.data["navn"])
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
    return jsonify(svar=svar, learned=learned, kilder=[c["title"] for _, c in passages])


@app.route("/api/saga", methods=["POST"])
def saga():
    emne = (request.get_json(silent=True) or {}).get("emne", "").strip()
    # RAG: matcher emnet en kendt saga/overlevering, så byg sagaen på den og vis kilden.
    passages = knowledge.retrieve(emne, k=3) if (knowledge and emne) else []
    if passages:
        msgs = persona.grounded_saga_messages(emne, [c for _, c in passages])
    else:
        msgs = persona.saga_messages(emne)
    return jsonify(svar=skald.reply(msgs, max_new_tokens=900, temperature=0.8),
                   kilder=[c["title"] for _, c in passages])


@app.route("/api/minsaga", methods=["POST"])
def minsaga():
    msgs = persona.personal_saga_messages(
        mem.data["navn"], mem.data["bedrifter"], mem.context())
    return jsonify(svar=skald.reply(msgs, max_new_tokens=900))


@app.route("/api/samtalesaga", methods=["POST"])
def samtalesaga():
    hist = mem.data["historik"]
    if not hist:
        return jsonify(svar="Vi har endnu ikke talt sammen, herre. Sæt dig ved ilden og "
                            "fortæl mig om dig — så væver jeg din saga af vore ord.")
    msgs = persona.conversation_saga_messages(
        mem.data["navn"], hist, mem.data["bedrifter"], mem.context())
    return jsonify(svar=skald.reply(msgs, max_new_tokens=900, temperature=0.85))


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
    global skald, knowledge
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="valgfri LoRA-adapter (finpuds)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true",
                    help="åbn ikke browseren automatisk")
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
        # Browseren åbnes lige efter serveren er bundet (modellen er alt indlæst nu).
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
