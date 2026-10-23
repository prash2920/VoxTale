"""
VoxTale: ElevenLabs Voice Design, used only when the library has no suitable voice.

Built against the structures in app.py:
  profile  = {name, aliases, gender: male|female|unknown, age: int|None,
              age_group: child|teen|young_adult|middle_aged|old|unknown,
              gender_source, age_source, traits: [str], ...}
  cast     = {character name: voice dict}, where a voice dict is
              {voice_id, name, gender, age, accent, tags}   (same shape get_voices() returns)

ElevenLabs endpoints (checked against the docs on 2026-10-22):
  POST /v1/text-to-voice/design  body: voice_description, auto_generate_text, seed, ...
                                 -> {"previews": [{"generated_voice_id", "audio_base_64", ...}], "text"}
  POST /v1/text-to-voice         body: voice_name, voice_description, generated_voice_id, labels
                                 -> {"voice_id", ...}   (saves the voice to the account)

Environment variables (all optional):
  VOICE_DESIGN=0            turn the feature off
  VOICE_DESIGN_MAX=3        max NEW voices designed per /api/segment request
  VOICE_DESIGN_WORKERS=2    parallel design calls
"""
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://api.elevenlabs.io/v1"
NAME_PREFIX = "VoxTale-"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "designed_voices.json")

_cache_lock = threading.Lock()
_key_locks = {}
_key_locks_guard = threading.Lock()


# ---------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------
def _env_flag(name, default=True):
    v = os.getenv(name)
    return default if v is None else v.strip().lower() not in ("0", "false", "off", "no", "")


def _env_int(name, default):
    try:
        return max(0, int(os.getenv(name, default)))
    except (TypeError, ValueError):
        return default


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(s).lower())).strip()


def is_designed(voice):
    return bool(voice.get("designed")) or str(voice.get("name", "")).startswith(NAME_PREFIX)


def library_voices(voices):
    """Voices that normal matching may use: everything except voices VoxTale designed itself,
    so one book's designed 'Bob' is never handed to another book's character."""
    lib = [v for v in voices if not is_designed(v)]
    return lib or voices


# ---------------------------------------------------------------
# disk cache: character -> designed voice_id (keeps the voice consistent across runs)
# ---------------------------------------------------------------
def _load_cache():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cache(cache):
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    os.replace(tmp, CACHE_FILE)


def design_key(profile, scope=""):
    """Stable identity of a character. Deliberately excludes traits and exact age, because the LLM
    words those differently on every run and that would create a new voice each time."""
    parts = [_norm(scope), _norm(profile.get("name", "")), profile.get("gender", "unknown"), profile.get("age_group", "unknown")]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _key_lock(key):
    with _key_locks_guard:
        return _key_locks.setdefault(key, threading.Lock())


# ---------------------------------------------------------------
# when is the library voice unsuitable?
# ---------------------------------------------------------------
YOUTH_RE = re.compile(r"\b(teen|teenage|teenager|boy|girl|child|kid|youth|youthful|juvenile|adolescent)\b")


def library_voice_unsuitable(profile, voice):
    """Returns (bool, reason). Only characters with a known gender are ever designed:
    a description without a gender would be a coin flip and would waste a voice slot."""
    g, grp = profile.get("gender"), profile.get("age_group")
    if g not in ("male", "female"):
        return False, "gender unknown"
    if not voice:
        return True, "no library voice matched"
    if voice.get("gender") in ("male", "female") and voice["gender"] != g:
        return True, f"best library voice is {voice['gender']}, character is {g}"
    vt = f"{voice.get('name', '')} {voice.get('tags', '')}".lower()
    if grp in ("child", "teen"):
        if YOUTH_RE.search(vt):
            return False, "library voice sounds young"
        return True, f"library has no {grp}-sounding voice (best match: {voice.get('name', '?')})"
    if grp == "old" and voice.get("age") != "old":
        return True, f"best library voice is not labelled old (it is '{voice.get('age') or 'unlabelled'}')"
    return False, "library voice is a good fit"


# ---------------------------------------------------------------
# profile -> voice description (ElevenLabs wants 20-1000 characters)
# ---------------------------------------------------------------
def _age_int(profile):
    a = profile.get("age")
    return a if isinstance(a, int) and 0 < a < 120 else None


def build_description(profile):
    g = profile.get("gender")
    grp = profile.get("age_group", "unknown")
    age = _age_int(profile)
    male = g == "male"

    if grp == "child":
        noun = "boy" if male else "girl"
        who = f"A {age}-year-old {noun}" if age else f"A young {noun}"
        sound = "a high-pitched, light, bright and clearly childlike voice, with an unbroken child's timbre"
    elif grp == "teen":
        noun = "boy" if male else "girl"
        who = f"A {age}-year-old teenage {noun}" if age else f"A teenage {noun}"
        sound = "a youthful voice that is lighter and higher than an adult's, with natural teenage energy and no deep adult resonance"
    elif grp == "young_adult":
        who = "A young man" if male else "A young woman"
        sound = "a fresh, clear, youthful adult voice"
    elif grp == "middle_aged":
        who = "A middle-aged man" if male else "A middle-aged woman"
        sound = "a steady, mature adult voice"
    elif grp == "old":
        who = "An elderly man" if male else "An elderly woman"
        sound = "an aged voice, slightly gravelly and unhurried, with a warm, weathered tone"
    else:
        who = "A man" if male else "A woman"
        sound = "a natural adult voice"

    traits = [str(t).strip() for t in (profile.get("traits") or []) if str(t).strip()][:4]
    trait_txt = f" Personality: {', '.join(traits)}." if traits else ""
    desc = (f"{who} with {sound}.{trait_txt} Clear studio-quality recording, natural conversational "
            f"delivery, expressive enough for audiobook character dialogue.")
    return desc[:1000]


def _age_label(profile):
    return {"child": "young", "teen": "young", "young_adult": "young", "middle_aged": "middle_aged", "old": "old"}.get(
        profile.get("age_group"), "")


# ---------------------------------------------------------------
# ElevenLabs HTTP with retries
# ---------------------------------------------------------------
class DesignError(Exception):
    def __init__(self, msg, status=None, fatal=False):
        super().__init__(msg)
        self.status, self.fatal = status, fatal


# Errors that will not fix themselves within this request: stop trying more characters.
FATAL_MARKERS = ("detected_unusual_activity", "missing_permissions", "invalid_api_key",
                 "quota_exceeded", "voice_limit", "limit_reached", "payment")


def _post(path, payload, api_key, timeout):
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    last = None
    for attempt in range(3):
        try:
            r = requests.post(f"{API}{path}", headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            last = DesignError(f"network error: {e}")
            if attempt == 0:
                time.sleep(2)
                continue
            raise last
        if r.ok:
            try:
                return r.json()
            except ValueError:
                raise DesignError("ElevenLabs returned invalid JSON", r.status_code)
        body = (r.text or "")[:300]
        low = body.lower()
        if any(m in low for m in FATAL_MARKERS) or r.status_code in (401, 402, 403):
            raise DesignError(f"ElevenLabs {r.status_code}: {body}", r.status_code, fatal=True)
        if r.status_code in (409, 429) or r.status_code >= 500:  # 409 = preview still generating, 429 = busy
            last = DesignError(f"ElevenLabs {r.status_code}: {body}", r.status_code)
            time.sleep(3 * (attempt + 1))
            continue
        raise DesignError(f"ElevenLabs {r.status_code}: {body}", r.status_code)
    raise last or DesignError("request failed")


def _design_and_save(profile, key, api_key, scope):
    desc = build_description(profile)
    name = f"{NAME_PREFIX}{profile['name']}"[:100]
    seed = int(key[:8], 16) % (2 ** 31)  # same character -> same seed -> reproducible voice if ever re-designed

    data = _post("/text-to-voice/design",
                 {"voice_description": desc, "auto_generate_text": True, "seed": seed}, api_key, timeout=180)
    previews = data.get("previews") or []
    generated_id = previews[0].get("generated_voice_id") if previews else None
    if not generated_id:
        raise DesignError("Voice Design returned no previews")

    labels = {"gender": profile["gender"], "source": "voxtale"}
    if _age_label(profile):
        labels["age"] = _age_label(profile)
    created = _post("/text-to-voice",
                    {"voice_name": name, "voice_description": desc, "generated_voice_id": generated_id, "labels": labels},
                    api_key, timeout=60)
    voice_id = created.get("voice_id")
    if not voice_id:
        raise DesignError("Create voice returned no voice_id")
    return {"voice_id": voice_id, "name": name, "description": desc, "gender": profile["gender"],
            "age_group": profile.get("age_group", ""), "age_label": _age_label(profile), "scope": scope,
            "created": int(time.time())}


def _voice_dict(entry):
    """Same shape as get_voices() entries, so the rest of app.py treats it like any other voice."""
    return {"voice_id": entry["voice_id"], "name": entry["name"], "gender": entry.get("gender", "unknown"),
            "age": entry.get("age_label", ""), "accent": "", "tags": entry.get("description", "").lower(), "designed": True}


# ---------------------------------------------------------------
# main entry point (called from /api/segment)
# ---------------------------------------------------------------
def apply_voice_design(profiles, cast, api_key, voices=None, segments=None, scope=""):
    """
    profiles : list of profile dicts (as built in api_segment)
    cast     : dict name -> voice dict from match_voices()
    voices   : the list from get_voices(). Optional but recommended: used to detect designed voices that were
               deleted in the dashboard, and designed voices are appended to it so /api/voices (and your
               per-character dropdown) includes them.
    segments : optional, used to design the characters with the most lines first when the limit is hit
    scope    : the book title (empty if none), so same-named characters in different books get different voices

    Returns (new_cast, report). report[name] = {"status": "designed"|"library"|"failed", "cached": bool, "reason"/"error": str}
    Never raises: on any problem the original library voices stay in place.
    """
    report = {}
    try:
        if not _env_flag("VOICE_DESIGN", True) or not api_key:
            return cast, report

        existing_ids = {v["voice_id"] for v in voices} if voices else None
        lines = {}
        for s in segments or []:
            lines[s.get("speaker")] = lines.get(s.get("speaker"), 0) + 1

        new_cast = dict(cast)
        pending = []
        with _cache_lock:
            cache = _load_cache()
            dirty = False
            for p in profiles:
                name = p.get("name")
                if not name or name == "Narrator":
                    continue
                bad, reason = library_voice_unsuitable(p, cast.get(name))
                if not bad:
                    report[name] = {"status": "library", "reason": reason}
                    continue
                key = design_key(p, scope)
                entry = cache.get(key)
                if entry and existing_ids is not None and entry.get("voice_id") not in existing_ids:
                    cache.pop(key, None)  # voice was deleted from the account: design it again
                    dirty, entry = True, None
                if entry and entry.get("voice_id"):
                    new_cast[name] = _voice_dict(entry)
                    report[name] = {"status": "designed", "cached": True, "reason": reason}
                else:
                    pending.append((p, key, reason))
            if dirty:
                _save_cache(cache)

        pending.sort(key=lambda x: -lines.get(x[0]["name"], 0))
        limit = _env_int("VOICE_DESIGN_MAX", 3)
        for p, _, _ in pending[limit:]:
            report[p["name"]] = {"status": "library", "reason": f"design limit per request reached ({limit})"}
        pending = pending[:limit]

        stop = threading.Event()  # set after a fatal error (permissions, quota, flagged account)

        def work(item):
            p, key, reason = item
            if stop.is_set():
                return p, None, "skipped after an earlier fatal error"
            with _key_lock(key):
                with _cache_lock:  # another request may have finished this character meanwhile
                    entry = _load_cache().get(key)
                if entry and entry.get("voice_id"):
                    return p, entry, None
                try:
                    entry = _design_and_save(p, key, api_key, scope)
                except DesignError as e:
                    if e.fatal:
                        stop.set()
                    return p, None, str(e)
                except Exception as e:  # never let one character break the whole request
                    return p, None, f"unexpected error: {e}"
                with _cache_lock:
                    c = _load_cache()
                    c[key] = entry
                    _save_cache(c)
                return p, entry, None

        if pending:
            with ThreadPoolExecutor(max_workers=max(1, _env_int("VOICE_DESIGN_WORKERS", 2))) as ex:
                results = list(ex.map(work, pending))
            reasons = {p["name"]: r for p, _, r in pending}
            for p, entry, err in results:
                name = p["name"]
                if entry:
                    vd = _voice_dict(entry)
                    new_cast[name] = vd
                    report[name] = {"status": "designed", "cached": False, "reason": reasons[name]}
                    if voices is not None and all(v["voice_id"] != vd["voice_id"] for v in voices):
                        voices.append(vd)
                else:
                    print(f"Voice design failed for {name}, using library voice: {err}")
                    report[name] = {"status": "failed", "error": err, "reason": reasons[name]}
        return new_cast, report
    except Exception as e:
        print("Voice design skipped (unexpected error):", e)
        return cast, report


# ---------------------------------------------------------------
# command line helpers
#   python voice_design.py --describe   print sample descriptions (no network)
#   python voice_design.py --probe      one real Voice Design call that does NOT save a voice
#                                       (checks your key, permissions and account flag)
# ---------------------------------------------------------------
if __name__ == "__main__":
    import sys

    sample = {"name": "Bob", "gender": "male", "age": 15, "age_group": "teen", "traits": ["curious", "nervous"]}
    if "--describe" in sys.argv:
        print(build_description(sample))
    elif "--probe" in sys.argv:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        k = os.getenv("ELEVENLABS_API_KEY")
        if not k:
            sys.exit("ELEVENLABS_API_KEY is not set")
        try:
            d = _post("/text-to-voice/design", {"voice_description": build_description(sample), "auto_generate_text": True}, k, 180)
            print("OK: Voice Design works with this key. previews:", len(d.get("previews", [])))
            print("Nothing was saved. Saving a voice (POST /v1/text-to-voice) is only tested when VoxTale designs a real character.")
        except DesignError as e:
            print("FAILED:", e)
    else:
        print(__doc__)