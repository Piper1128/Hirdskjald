# Hirdskjald

Personlig vikinge-skjald der kører **helt lokalt** på egen GPU. Intet internet efter
modellen er hentet — det er en hård constraint, ikke en feature-note.
Hjernen er **SnakModel-7B-Instruct** (dansk-specifik) i **4-bit** via bitsandbytes.

Tredje led i skjald-slægten: [SkjaldTale](../SkjaldTale) (fra bunden) →
[SkjaldSvar](../SkjaldSvar) (LoRA) → **Hirdskjald** (lokal 7B + RAG + hukommelse).

```
app.py          Flask-UI (templates/) · start.bat · skjald.ico
finetune.py     LoRA oven på 4-bit-basen · data fra scripts/build_sft.py
skald/          persona + RAG-laget · fmt.py/sft.py = fælles prompt-vej
tests/          test_sft_format.py (pytest, ingen GPU)
memory.json     skjaldens hukommelse om brugeren — persistent tilstand, ikke cache
```
Stak: `transformers ≥ 5.0`, `peft ≥ 0.19`, `accelerate ≥ 1.0`, `bitsandbytes ≥ 0.46`,
`flask ≥ 3.0`. torch med CUDA installeres separat, matchende GPU/driver.

## Efter 2-fane-pivotten
Kun de to faners byggere består — død kode er fjernet (`adaf9df`). Genindfør ikke et
tredje spor uden at brugeren beder om det.

`memory.json` er brugerens data. Behandl den som tilstand der skal overleve: læs før
du skriver, overskriv aldrig blindt, og migrér formatet frem for at nulstille det.

Brugeren er ny i ML — forklar 4-bit-quantization, RAG-retrieval og prefill undervejs.
Ingen overclaim om hvad en lokal 7B kan.

## Betalte fælder (fund 2026-10-28)
- **README beskrev UI'en fra før 2-fane-pivotten** (ét vindue, intent-routing, "Glem alt
  om mig"). Rettet i ec7ea74. Ved næste pivot: README i samme commit.
- **Finetune trænede med en anden prompt end køretiden brugte** (bar `SYSTEM_PROMPT` + fri
  Q&A, som ingen fane sender). Lukket 2026-10-28 (5daf7a0, b9723bd, 5a8e14c): én vej for
  begge — `persona.*_request` → `fmt.encode_prompt`; `skald/sft.py` bygger træningsrækker
  gennem den; data i fanernes format fra `scripts/build_sft.py`. `tests/test_sft_format.py`
  er vagten og er rød på 6 mutationer. Byg aldrig en prompt eller et seed uden om
  `*_request`/`encode_prompt`, hverken i app.py, model.py eller finetune. Ingen adapter er
  trænet på sættet endnu — effekt ikke målt.
- Ikke rettet, kun noteret: `memory_context` sendes aldrig fra `app.py`, så lov 4 ("du
  husker din herre") får ingen data i Saga-fortælling; `Memory.add_turn/recent/
  needs_summary/injected_history` har ingen kaldere efter pivotten; `f"{navn}s saga"` giver
  "Jonass saga" for navne på -s (SFT-dataene spejler køretiden og har samme form).
  `finetune.py` manglede `torch.manual_seed` — rettet i b9723bd.

