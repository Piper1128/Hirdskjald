"""Skjald-persona: base-model, system-prompt og prompt-byggere for Hirdskjald."""

BASE_MODEL = "NLPnorth/snakmodel-7b-instruct"

SYSTEM_PROMPT = (
    "Du er Hirdskjald — en skjald fra vikingetidens Norden i din herres personlige tjeneste. "
    "Du taler smukt, arkaisk dansk med islæt af oldnordiske ord (jarl, thing, blót, holmgang, "
    "Midgård, Valhal, Æsir). Du kvæder om guder, konger, søfart, ære og blodhævn, som havde du "
    "selv været til stede ved ilden. Du kender intet til den moderne verden; spørges du om "
    "maskiner, elektricitet eller fjerne nutids-riger, undrer du dig oprigtigt i din samtids "
    "sprog og vender ordet mod sagn og guder. Du husker din herre — hans navn og hans bedrifter "
    "— og fletter dem gerne ind i dine kvad. Svar fyndigt og billedrigt, aldrig langt og fladt."
)


def _system(memory_context=""):
    sys = SYSTEM_PROMPT
    if memory_context:
        sys += "\n\nDette ved du om din herre:\n" + memory_context
    return sys


def chat_messages(user_text, memory_context="", history=None):
    """Byg besked-listen til en samtale-tur, med hukommelse flettet ind i systemet."""
    msgs = [{"role": "system", "content": _system(memory_context)}]
    for turn in (history or []):
        msgs.append({"role": "user", "content": turn["user"]})
        msgs.append({"role": "assistant", "content": turn["skald"]})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def saga_messages(emne, memory_context=""):
    """Bed skjalden fortælle en saga om et emne (eller et frit valgt, hvis tomt)."""
    bid = ("Kvæd en kort saga" + (f" om {emne}" if emne else "") +
           ". Lad den have begyndelse, dåd og eftermæle, som en ægte skjald ved ilden.")
    return [{"role": "system", "content": _system(memory_context)},
            {"role": "user", "content": bid}]


def personal_saga_messages(navn, bedrifter, memory_context=""):
    """Bed skjalden digte HERRENS egen saga ud fra hans bedrifter."""
    hvem = navn or "min herre"
    daad = "; ".join(bedrifter) if bedrifter else "endnu ufortalte dåd"
    bid = (f"Kvæd {hvem}s egen saga. Byg den på disse bedrifter: {daad}. "
           "Gør ham til helten; giv ham et skjald-tilnavn, nævn hans dåd i hæder, "
           "og luk med hans eftermæle blandt mænd.")
    return [{"role": "system", "content": _system(memory_context)},
            {"role": "user", "content": bid}]
