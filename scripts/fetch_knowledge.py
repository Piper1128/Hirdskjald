"""
fetch_knowledge.py — hent en dansk viden-base om nordisk mytologi og sagaer
fra dansk Wikipedia (til RAG-grounding). Gemmes i data/knowledge.jsonl.

    python scripts/fetch_knowledge.py

Kilde: da.wikipedia.org (CC BY-SA). Committes IKKE — kør scriptet for at hente lokalt.
Efter dette kører alt offline; viden-basen bruges kun til opslag.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API = "https://da.wikipedia.org/w/api.php"

TITLES = [
    "Nordisk mytologi", "Aser", "Vaner", "Odin", "Thor", "Frej", "Freja", "Loke",
    "Balder", "Tyr (gud)", "Heimdal", "Njord", "Frigg", "Idun", "Brage", "Ægir",
    "Hel (mytologi)", "Vidar", "Ull", "Skade", "Sif", "Jætte", "Valkyrie", "Norne",
    "Dværg (mytologi)", "Yggdrasil", "Asgård", "Midgård", "Valhal", "Bifrost",
    "Niflheim", "Muspelheim", "Jotunheim", "Ragnarok", "Fimbulvinter", "Fenrisulven",
    "Midgårdsormen", "Sleipner", "Hugin og Munin", "Mjølner", "Draupner", "Gungner",
    "Den ældre Edda", "Snorris Edda", "Håvamål", "Vøluspá", "Ymer", "Ginnungagap",
    "Ragnar Lodbrog", "Njals saga", "Egil Skallagrimssons saga", "Vølsungesagaen",
    "Gunnlaug Ormstunges saga", "Runer", "Blót", "Vikingetiden", "Holmgang",
    "Ting (forsamling)", "Skjald", "Bersærk",
]


def fetch_batch(titles, retries=4):
    params = {"action": "query", "prop": "extracts", "explaintext": "1",
              "format": "json", "redirects": "1", "exlimit": "max",
              "titles": "|".join(titles)}
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Hirdskjald/1.0 (laeringsprojekt)"})
            data = json.load(urllib.request.urlopen(req, timeout=60))
            out = []
            for p in data.get("query", {}).get("pages", {}).values():
                if "extract" in p and p["extract"].strip():
                    out.append((p["title"], p["extract"]))
            return out
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            raise
    return []


def chunk(text, size=900):
    paras = [p.strip() for p in text.split("\n") if p.strip() and not p.startswith("==")]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) > size and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + "\n" + p).strip()
    if cur:
        chunks.append(cur)
    return chunks


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "data", "knowledge.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n_art = n_chunk = 0
    # Fuld artikel-tekst (explaintext) er begrænset til ÉN titel per forespørgsel,
    # så vi henter en ad gangen.
    with open(out, "w", encoding="utf-8") as f:
        for title in TITLES:
            try:
                results = fetch_batch([title])
            except Exception as e:
                print(f"  (fejl paa {title}: {e})")
                results = []
            if not results:
                print(f"  (sprang over: {title})")
            for real_title, extract in results:
                n_art += 1
                for c in chunk(extract):
                    if len(c) > 80:
                        f.write(json.dumps({"title": real_title, "text": c},
                                           ensure_ascii=False) + "\n")
                        n_chunk += 1
            time.sleep(1.0)
    print(f"Hentede {n_art} artikler -> {n_chunk} tekststykker i {out}")


if __name__ == "__main__":
    main()
