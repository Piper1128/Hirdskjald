# Hirdskjald

Personlig vikinge-skjald der kører **helt lokalt** på egen GPU. Intet internet efter
modellen er hentet — det er en hård constraint, ikke en feature-note.
Hjernen er **SnakModel-7B-Instruct** (dansk-specifik) i **4-bit** via bitsandbytes.

Tredje led i skjald-slægten: [SkjaldTale](../SkjaldTale) (fra bunden) →
[SkjaldSvar](../SkjaldSvar) (LoRA) → **Hirdskjald** (lokal 7B + RAG + hukommelse).

```
app.py          Flask-UI (templates/) · start.bat · skjald.ico
finetune.py     LoRA oven på 4-bit-basen
skald/          persona + RAG-laget
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
- **ÅBEN: finetune træner med en anden prompt end køretiden bruger.** `finetune.py`
  bygger `[system=SYSTEM_PROMPT, user, assistant]`. Køretid bygger
  `[system=_system() (+identitet, +hukommelse), *_ANCHOR, user=saga-/kapitel-bid]` + et
  prefill-seed. Adapteren lærer altså en ramme den aldrig ser. Derudover er
  `skald_sft.jsonl` fri Q&A ("Hvem er du?"), som ingen af de to faner sender, og dens svar
  siger "dreng af Island" mens identiteten kan tildele en anden hjemstavn. Ingen trænet
  adapter eller loss-log findes, så effekten er ikke målt.
- Ikke rettet, kun noteret: `memory_context` sendes aldrig fra `app.py`, så lov 4 ("du
  husker din herre") får ingen data i Saga-fortælling; `Memory.add_turn/recent/
  needs_summary/injected_history` har ingen kaldere efter pivotten; `finetune.py` sætter
  ikke `torch.manual_seed`, så LoRA-A-init er ikke reproducerbar.

