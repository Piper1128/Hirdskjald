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
