"""Lokal hukommelse for Hirdskjald — helt offline, én JSON-fil.

Dette er den ærlige "skjalden lærer dig at kende": han husker navn, bedrifter og
samtale-historik og fletter det ind i konteksten (genkald), ikke vægt-træning.
Samme fundament porteres senere til telefon-app'en.
"""
import json
import os


def _default():
    return {"navn": "", "bedrifter": [], "fakta": [], "historik": []}


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
        self.data["historik"] = self.data["historik"][-40:]  # behold de seneste
        self.save()

    def recent(self, n=6):
        return self.data["historik"][-n:]

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
