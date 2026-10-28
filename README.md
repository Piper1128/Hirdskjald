# Hirdskjald

**Din personlige vikinge-skjald — kørende helt lokalt på din egen GPU.** En dansk
skjald som kvæder sagaer, husker dig, og digter din *egen* saga ud
af dine bedrifter. Intet internet efter modellen er hentet.

Hjernen er [SnakModel-7B-Instruct](https://huggingface.co/NLPnorth/snakmodel-7b-instruct)
(dansk-specifik) kørt i **4-bit** — den taler flydende, arkaisk skjald-dansk ud af boksen.

> **Tredje del af skjald-slægten.** [SkjaldTale](https://github.com/Piper1128/SkjaldTale)
> bygger en LLM *fra bunden*; [SkjaldSvar](https://github.com/Piper1128/SkjaldSvar)
> LoRA-finetuner en lille model til skjald-persona; **Hirdskjald** samler oplevelsen til
> ét program på en dansk-stærk 7B — med hukommelse og din-egen-saga.

## To faner

- **Saga-fortælling** — bed om en saga. Nævner du et emne ("fortæl en saga om Ragnarok"),
  kvæder han den med det samme; ellers stiller han ét til to faste, sokratiske spørgsmål
  for at ringe den rigtige saga ind. Sagaen grundes i viden-basen (RAG), og han siger fra
  hvilket skrift den er (Edda/saga). "Ny saga" eller "forfra" starter forfra.
- **Din saga** — giv dit navn, dine bedrifter og fri tekst om dig selv. Han kvæder din egen
  saga kapitel for kapitel og fletter de nye bedrifter ind hver gang. "Nulstil min saga"
  sletter kapitlerne.

Skjalden får én fast identitet (navn, hjemstavn, forhistorie) første gang, og den,
dit navn, dine bedrifter og sagaens kapitler gemmes lokalt i `memory.json`.

## Kør det

**Nemmest (Windows):** dobbeltklik **`start.bat`** — skjalden vågner og browseren åbner af
sig selv. (Første gang skal afhængighederne dog installeres, se nedenfor.)

Manuelt:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python app.py            # åbner selv http://127.0.0.1:5000 i browseren
```
Første start henter SnakModel-7B (~14 GB). Derefter kører alt offline. Kræver en
CUDA-GPU (udviklet på RTX 3080, 10 GB — modellen fylder kun ~4 GB VRAM i 4-bit).

**Valgfrit finpuds** (LoRA-lås af persona):
```bash
python finetune.py --epochs 3 --out adapter
python app.py --adapter adapter
```

## Viden (RAG) — så han svarer korrekt, ikke bare stemningsfuldt

Hent en dansk viden-base om nordisk mytologi og sagaer (fra Wikipedia) én gang:
```bash
python scripts/fetch_knowledge.py     # -> data/knowledge.jsonl (63 titler fra da.wikipedia.org)
```
Så slår skjalden op i den **semantisk** med en lille flersproget embedding-model (e5,
~120 MB, kører på GPU'en; falder tilbage til BM25-nøgleord uden) og **grunder sine svar
i rigtige overleveringer** i stedet for at gætte. Uden basen kører han fint — bare ustyret
af fakta.

## Ærligt om hvad det er
- En **persona/rollespil**-skjald, ikke en historiker: han taler smukt og i-rolle, men
  finder på og husker ikke saga-fakta præcist. "Din egen saga" er stil + hukommelse.
- **"Han lærer dig at kende" = lokal hukommelse (genkald)**, ikke vægt-træning på enheden.
  Det er den stabile, offline måde — og fundamentet vi senere porterer til en telefon-app.
- Arkaisk-farvet **dansk**, ikke flydende oldnordisk (ingen model behersker oldnordisk samtale).
- **RAG gør ham mere korrekt, ikke alvidende.** Semantisk opslag over paragraf-fine chunks
  grounder de fleste svar korrekt (Odins øje ved Mimers brønd, Ragnarok, Tyrs hånd, Fenrir-
  bindingen) — men en 7B kan stadig pynte lidt oven på kilden. Fuld faktuel stramhed ville
  kræve en større model eller en reranker.

## Struktur
```text
Hirdskjald/
├── start.bat              # dobbeltklik-launcher (Windows)
├── app.py                 # Flask-server + web-UI
├── skald/
│   ├── model.py           # SnakModel-7B i 4-bit (+ valgfri LoRA), generering
│   ├── persona.py         # system-prompt, persona-anker + prompt-byggere
│   ├── memory.py          # lokal JSON-hukommelse (identitet, navn, bedrifter, kapitler)
│   ├── knowledge.py       # opslag i viden-basen (semantisk + BM25-fallback)
│   └── embeddings.py      # lille flersproget e5-embedder (dense retrieval)
├── templates/index.html   # UI (Saga-fortælling / Din saga)
├── scripts/
│   └── fetch_knowledge.py # hent dansk viden-base (Wikipedia) -> data/knowledge.jsonl
├── finetune.py            # valgfrit 4-bit QLoRA-finpuds
└── data/skald_sft.jsonl   # skjald-SFT-par (til finpuds)
```

## Licens
MIT — se [LICENSE](LICENSE).
