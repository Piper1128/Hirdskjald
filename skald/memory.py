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
            "saga": {"aktiv": False, "historie": "", "del": 0}}  # interaktiv saga over flere ture


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

    def clear_conversation(self):
        """Ryd KUN samtalen (historik + resumé + evt. igangværende saga); behold navn,
        bedrifter, fakta OG skjaldens egen identitet (din skjald forbliver din)."""
        self.data["historik"] = []
        self.data["resume"] = ""
        self.data["resume_count"] = 0
        self.data["saga"] = {"aktiv": False, "historie": "", "del": 0}
        self.save()

    # --- Skjaldens egen identitet ---
    def set_skjald(self, identitet):
        self.data["skjald"] = identitet
        self.save()

    # --- Interaktiv saga over flere ture (herren styrer undervejs) ---
    def saga_active(self):
        return bool((self.data.get("saga") or {}).get("aktiv"))

    def saga_begin(self, part):
        self.data["saga"] = {"aktiv": True, "historie": part[:1500], "del": 1}
        self.save()

    def saga_advance(self, part):
        s = self.data.get("saga") or {"aktiv": True, "historie": "", "del": 0}
        s["historie"] = (s.get("historie", "") + "\n\n" + part)[-1500:]  # behold seneste narrativ
        s["del"] = s.get("del", 0) + 1
        s["aktiv"] = True
        self.data["saga"] = s
        self.save()

    def saga_stop(self):
        self.data["saga"] = {"aktiv": False, "historie": "", "del": 0}
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
