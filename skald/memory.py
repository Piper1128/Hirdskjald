"""Lokal hukommelse for Hirdskjald — helt offline, én JSON-fil.

Dette er den ærlige "skjalden lærer dig at kende": han husker navn, bedrifter og
samtale-historik og fletter det ind i konteksten (genkald), ikke vægt-træning.
Samme fundament porteres senere til telefon-app'en.
"""
import json
import os


def _default():
    return {"navn": "", "bedrifter": [], "fakta": [], "historik": [],
            "resume": "", "resume_count": 0,
            "skjald": None,  # skjaldens EGEN faste identitet (navn/hjemstavn/forhistorie)
            # Fane "Din saga": kapitler bygges op over tid; brugt_* = hvor meget der er foldet ind
            "min_saga": {"kapitler": [], "brugt_bedrifter": 0, "brugt_fakta": 0},
            # Fane "Saga-fortælling": kortvarig sokratisk tilstand (hvilken saga vil du høre)
            "saga_flow": {"aktiv": False, "runde": 0, "svar": []}}


class Memory:
    def __init__(self, path="memory.json"):
        self.path = path
        self.data = _default()
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    self.data = {**_default(), **json.load(f)}
            except (json.JSONDecodeError, OSError):
                self.data = _default()

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def set_navn(self, navn):
        self.data["navn"] = (navn or "").strip()
        self.save()

    def add_bedrift(self, bedrift):
        b = (bedrift or "").strip()
        if b and b not in self.data["bedrifter"]:
            self.data["bedrifter"].append(b)
            self.save()

    def add_fakta(self, fakta):
        f = (fakta or "").strip()
        if f and f not in self.data["fakta"]:
            self.data["fakta"].append(f)
            self.save()

    def remove_bedrift(self, bedrift):
        self.data["bedrifter"] = [b for b in self.data["bedrifter"] if b != bedrift]
        self.save()

    def remove_fakta(self, fakta):
        self.data["fakta"] = [f for f in self.data["fakta"] if f != fakta]
        self.save()

    def add_turn(self, user, skald):
        self.data["historik"].append({"user": user, "skald": skald})
        overflow = len(self.data["historik"]) - 60
        if overflow > 0:  # tab de ældste, og flyt resumé-markøren med, så indeks holder
            self.data["historik"] = self.data["historik"][overflow:]
            self.data["resume_count"] = max(0, self.data["resume_count"] - overflow)
        self.save()

    def recent(self, n=6):
        return self.data["historik"][-n:]

    # --- Rullende samtale-hukommelse: ældre ture foldes ind i et kort resumé ---
    def needs_summary(self, keep_raw=4, batch=4):
        """Sandt når nok gamle ture har hobet sig op til at folde ind i resuméet."""
        return len(self.data["historik"]) - self.data["resume_count"] > keep_raw + batch

    def to_summarize(self, keep_raw=4):
        """De ældre, endnu-ikke-resumerede ture (alt undtagen de seneste keep_raw)."""
        end = len(self.data["historik"]) - keep_raw
        return self.data["historik"][self.data["resume_count"]:end], end

    def set_resume(self, resume, covered_end):
        self.data["resume"] = (resume or "").strip()
        self.data["resume_count"] = covered_end
        self.save()

    def injected_history(self):
        """De rå ture der endnu ikke er foldet ind i resuméet (til prompten)."""
        return self.data["historik"][self.data["resume_count"]:]

    # --- Skjaldens egen identitet ---
    def set_skjald(self, identitet):
        self.data["skjald"] = identitet
        self.save()

    # --- Fane "Din saga": kapitler bygges op over tid ---
    def min_saga_add(self, kapitel):
        ms = self.data.get("min_saga") or {"kapitler": [], "brugt_bedrifter": 0, "brugt_fakta": 0}
        ms["kapitler"].append((kapitel or "").strip())
        ms["brugt_bedrifter"] = len(self.data["bedrifter"])  # alt kendt er nu foldet ind
        ms["brugt_fakta"] = len(self.data["fakta"])
        self.data["min_saga"] = ms
        self.save()

    def clear_min_saga(self):
        """Ryd DIN saga helt (navn, bedrifter, noter, kapitler) — behold skjaldens egen identitet."""
        self.data["navn"] = ""
        self.data["bedrifter"] = []
        self.data["fakta"] = []
        self.data["min_saga"] = {"kapitler": [], "brugt_bedrifter": 0, "brugt_fakta": 0}
        self.save()

    # --- Fane "Saga-fortælling": kortvarig sokratisk tilstand ---
    def flow_begin(self):
        self.data["saga_flow"] = {"aktiv": True, "runde": 1, "svar": []}
        self.save()

    def flow_answer(self, svar):
        f = self.data.get("saga_flow") or {"aktiv": True, "runde": 1, "svar": []}
        f["svar"].append((svar or "").strip())
        self.data["saga_flow"] = f
        self.save()

    def flow_set_runde(self, runde):
        self.data["saga_flow"]["runde"] = runde
        self.save()

    def flow_reset(self):
        self.data["saga_flow"] = {"aktiv": False, "runde": 0, "svar": []}
        self.save()

    def context(self):
        """Render hvad skjalden ved om herren, til indfletning i system-prompten."""
        lines = []
        if self.data["navn"]:
            lines.append(f"- Hans navn er {self.data['navn']}.")
        for b in self.data["bedrifter"]:
            lines.append(f"- Bedrift: {b}")
        for f in self.data["fakta"]:
            lines.append(f"- {f}")
        return "\n".join(lines)

    def clear(self):
        self.data = _default()
        self.save()
