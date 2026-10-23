import hashlib
import json
import os
import re
import time

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file, send_from_directory

from voice_design import apply_voice_design, library_voices

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
EXA_KEY = os.getenv("EXA_API_KEY")  # optional if exa fails-- we can use wikipedia
PORT = int(os.getenv("PORT", 3000))
# How much text one request may contain. Groq's free tier has token-per-minute limits,
# so if you see rate limi errror lower down the char
MAX_CHARS = int(os.getenv("MAX_CHARS", 12000))
CHUNK_CHARS = 3000  # text is segmented in pieces of about this size

# static_folder=None so we control the "/" route ourselves; frontend lives in ./public
app = Flask(__name__, static_folder=None)


# 1) ElevenLabs voices (cached in memory after the first call)
_voice_cache = None


def get_voices():
    global _voice_cache
    if _voice_cache:
        return _voice_cache
    r = requests.get(
        "https://api.elevenlabs.io/v1/voices",
        headers={"xi-api-key": ELEVEN_KEY},
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"ElevenLabs voices error {r.status_code}: {r.text}")
    out = []
    for v in r.json()["voices"]:
        labels = v.get("labels") or {}
        extra = [labels.get("descriptive"), labels.get("description"), labels.get("use_case"), v.get("description")]
        out.append(
            {
                "voice_id": v["voice_id"],
                "name": v["name"],
                "gender": labels.get("gender", "unknown"),
                "age": labels.get("age", ""),  # young / middle_aged / old
                "accent": labels.get("accent", ""),
                "tags": " ".join(str(x) for x in extra if x).lower(),
            }
        )
    _voice_cache = out
    return _voice_cache



#  LLM helper (Groq, GPT-OSS 120B)
def groq_json(prompt, max_tokens=8000):
    """Send one prompt to Groq and return the parsed JSON object it answers with."""
    for attempt in range(2):
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "reasoning_effort": "low",
                "temperature": 0.2,
                "max_completion_tokens": max_tokens,
            },
            timeout=90,
        )
        if r.status_code == 429 and attempt == 0:  # rate limited: wait a moment, try once more
            try:
                wait = min(float(r.headers.get("retry-after", 5)), 10)
            except ValueError:
                wait = 5
            time.sleep(wait)
            continue
        break
    if not r.ok:
        raise RuntimeError(f"Groq error {r.status_code}: {r.text[:300]}")
    raw = r.json()["choices"][0]["message"]["content"]
    return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])



# Cast analysis: read the WHOLE text first and build grounded character profiles
AGE_GROUPS = ("child", "teen", "young_adult", "middle_aged", "old")


def age_group_from_age(a):
    return "child" if a < 13 else "teen" if a < 20 else "young_adult" if a < 36 else "middle_aged" if a < 61 else "old"


def norm(s):
    """Lowercase, strip punctuation and extra spaces so quotes can be compared with the text."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(s).lower())).strip()


def clean_profile(c, text_norm, ref_norm="", book=""):
    name = str(c.get("name", "")).strip()
    if not name:
        return None
    gender = c.get("gender") if c.get("gender") in ("male", "female") else "unknown"
    group = c.get("age_group") if c.get("age_group") in AGE_GROUPS else "unknown"
    try:
        age = int(c.get("age"))
        age = age if 0 < age < 120 else None
    except (TypeError, ValueError):
        age = None

    # GROUNDING CHECK: every claim must say where it came from, and quotes must really exist.
    ev = str(c.get("evidence", "")).strip()
    rev = str(c.get("ref_evidence", "")).strip()
    ok_text = bool(ev) and norm(ev) in text_norm
    ok_ref = bool(rev) and bool(ref_norm) and norm(rev) in ref_norm

    def settle(src, known):
        if not known:
            return "unknown"
        if src == "text":
            return "text" if ok_text else "inferred"          # claimed a quote that is not in the text
        if src == "reference":
            return "reference" if ok_ref else "inferred"      # claimed a quote that is not in the reference
        if src == "knowledge":
            return "knowledge" if book else "inferred"        # model's own memory of a famous book (unverifiable)
        return "inferred"

    age_src = settle(c.get("age_source"), age is not None or group != "unknown")
    gender_src = settle(c.get("gender_source"), gender != "unknown")
    if age_src not in ("text", "reference"):
        age = None  # an exact number needs proof; otherwise only the rough age group is kept
    if age is not None:
        group = age_group_from_age(age)

    return {
        "name": name,
        "aliases": [str(a).strip() for a in (c.get("aliases") or []) if str(a).strip()][:6],
        "gender": gender,
        "age": age,
        "age_group": group,
        "gender_source": gender_src,
        "age_source": age_src,
        "traits": [str(t).lower().strip() for t in (c.get("traits") or []) if str(t).strip()][:4],
        "evidence": ev if ok_text else "",
        "ref_evidence": rev if ok_ref else "",
        "reason": str(c.get("reason", "")).strip()[:120],
        "verified": ok_text,
    }


def fetch_wikipedia(book, limit=8000):
    """Best effort: plain-text background about a known book from Wikipedia (free, no key).
    Returns (page_title, text), or ("", "") if anything fails or the page does not look related."""
    if not book:
        return "", ""
    try:
        api, hdr = "https://en.wikipedia.org/w/api.php", {"User-Agent": "VoxTale-hackathon/0.1"}
        hits = requests.get(api, params={"action": "query", "list": "search", "srsearch": f"{book} characters",
                                         "format": "json", "srlimit": 3}, headers=hdr, timeout=6).json()["query"]["search"]
        want = {w for w in norm(book).split() if len(w) > 2}
        title = next((h["title"] for h in hits if want & set(norm(h["title"]).split())), "")
        if not title:
            return "", ""  # nothing that looks like this book
        pages = requests.get(api, params={"action": "query", "prop": "extracts", "explaintext": 1, "exlimit": 1,
                                          "titles": title, "redirects": 1, "format": "json"},
                             headers=hdr, timeout=6).json()["query"]["pages"]
        text = next(iter(pages.values())).get("extract", "")
        return (title, text[:limit]) if text else ("", "")
    except Exception as e:
        print("Wikipedia lookup skipped:", e)
        return "", ""


def fetch_exa(book, limit=8000):
    """Exa web search: finds character guides, wikis and summaries for the book. Returns (label, text)."""
    try:
        r = requests.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": EXA_KEY, "Content-Type": "application/json"},
            json={"query": f"{book} characters ages and descriptions", "type": "auto", "numResults": 4,
                  "contents": {"text": {"maxCharacters": 3000}}},
            timeout=15,
        )
        if not r.ok:
            raise RuntimeError(f"Exa error {r.status_code}: {r.text[:200]}")
        want = {w for w in norm(book).split() if len(w) > 2}
        parts, hosts = [], []
        for hit in r.json().get("results", []):
            body = hit.get("text") or ""
            # keep only pages that actually look like they are about this book
            if not body or not (want & set(norm((hit.get("title") or "") + " " + body[:600]).split())):
                continue
            host = re.sub(r"^https?://(www\.)?", "", hit.get("url", "")).split("/")[0]
            parts.append(f"[{hit.get('title', '')} - {host}]\n{body}")
            hosts.append(host)
        text = "\n\n".join(parts)[:limit]
        return (", ".join(dict.fromkeys(hosts)), text) if text else ("", "")
    except Exception as e:
        print("Exa lookup skipped:", e)
        return "", ""


def fetch_reference(book):
    """Returns (source_name, label, text). Exa first when a key is set, then Wikipedia."""
    if not book:
        return "", "", ""
    if EXA_KEY:
        label, text = fetch_exa(book)
        if text:
            return "Exa", label, text
    label, text = fetch_wikipedia(book)
    return ("Wikipedia", label, text) if text else ("", "", "")


def analyze_cast(text, book="", reference=""):
    ref_block = f"\nREFERENCE (background about this book from web sources, may be incomplete):\n{reference}\n" if reference else ""
    book_line = f'This text is from the book "{book}".' if book else "The book is not named."
    prompt = f"""You are casting voice actors for an audiobook. {book_line}
Read the whole TEXT and list every character who speaks or matters to the story. For each character decide
gender and age, and say WHERE each fact comes from (age_source and gender_source):
- "text": the TEXT states it (e.g. "fifteen-year-old Bob", "she"). Copy ONE short exact quote (max 15 words) into "evidence".
- "reference": the REFERENCE states it. Copy ONE short exact quote from the REFERENCE into "ref_evidence".
- "inferred": not stated, but the TEXT gives clues: occupation, relationships ("her grandmother", "the schoolboy"),
  behaviour, or voice descriptions ("squeaky", "rasping", "booming"). Give the clue in "reason" (max 12 words).
- "knowledge": not in TEXT or REFERENCE, but you reliably know it about this famous book. Only if confident.
- "unknown": no clue at all. Prefer "unknown" over wild guessing.
Other fields:
- gender: "male" or "female", else "unknown".
- age: an integer ONLY when source is "text" or "reference" and the age is actually stated. Otherwise null.
- age_group: child (under 13), teen (13-19), young_adult (20-35), middle_aged (36-60), old (60+), or unknown.
- traits: up to 4 short words for how they sound or behave (gruff, cheerful, nervous, stern).
- name: the character's name; if the text never names them, use their role ("Old Woman", "Stranger").
- aliases: every other way the text refers to the same person: nicknames, titles, and descriptive phrases
  ("the man in the grey coat"), so one person never becomes two characters.
Never invent quotes. Do not include "Narrator".
Return ONLY JSON: {{"characters":[{{"name":"Bob","aliases":["Bobby"],"gender":"male","gender_source":"text","age":15,"age_group":"teen","age_source":"text","traits":["curious"],"evidence":"a fifteen-year-old boy","ref_evidence":"","reason":""}}]}}
{ref_block}
TEXT:
{text}"""
    data = groq_json(prompt, max_tokens=6000)
    text_norm, ref_norm = norm(text), norm(reference)
    profiles = [p for p in (clean_profile(c, text_norm, ref_norm, book) for c in data.get("characters", [])) if p]
    seen, unique = set(), []
    for p in profiles:  # drop duplicates and any "Narrator" the model added anyway
        k = p["name"].lower()
        if k not in seen and k != "narrator":
            seen.add(k)
            unique.append(p)
    return unique


def profile_label(p):
    if p["name"] == "Narrator":
        return "narrator"
    g, grp = p["gender"], p["age_group"]
    young = grp in ("child", "teen")
    noun = {"male": "boy" if young else "man" if grp != "unknown" else "male",
            "female": "girl" if young else "woman" if grp != "unknown" else "female"}.get(g, "character")
    if p["age"]:
        age = f"{p['age']}-year-old"
    else:
        age = {"child": "young", "teen": "teen", "young_adult": "young", "middle_aged": "middle-aged", "old": "elderly"}.get(grp, "")
        if grp == "young_adult" and not young:
            age = "young"
    return " ".join(x for x in [age, noun] if x)


def default_profile(name):
    return {"name": name, "aliases": [], "gender": "unknown", "age": None, "age_group": "unknown",
            "gender_source": "unknown", "age_source": "unknown", "traits": [], "evidence": "",
            "ref_evidence": "", "reason": "", "verified": False}


# ---------------------------------------------------------------
# 4) Voice matching: pick the ElevenLabs voice that best fits each profile
# ---------------------------------------------------------------
VOICE_AGE_FOR_GROUP = {"child": "young", "teen": "young", "young_adult": "young", "middle_aged": "middle_aged", "old": "old"}
VOICE_AGE_ORDER = ["young", "middle_aged", "old"]
YOUNG_WORDS = ("young", "youth", "teen", "boy", "girl", "child", "kid", "playful", "energetic", "bright")
OLD_WORDS = ("old", "elder", "wise", "grand", "gravel", "raspy", "aged", "mature")
DEEP_WORDS = ("deep", "gravel", "raspy", "resonant", "mature", "baritone")


def voice_text(v):
    return f"{v['name']} {v.get('tags', '')}".lower()


def score_voice(v, p, narrator=False):
    vt = voice_text(v)
    if narrator:
        s = 40 if any(w in vt for w in ("narrat", "story", "audiobook")) else 0
        return s + (10 if v.get("age") == "middle_aged" else 0)

    s = 0
    if p["gender"] in ("male", "female"):
        s += 50 if v.get("gender") == p["gender"] else -100
    want = VOICE_AGE_FOR_GROUP.get(p["age_group"])
    if want and v.get("age") in VOICE_AGE_ORDER:
        d = abs(VOICE_AGE_ORDER.index(want) - VOICE_AGE_ORDER.index(v["age"]))
        s += {0: 30, 1: 8}.get(d, -15)
    if p["age_group"] in ("child", "teen"):
        s += min(24, 12 * sum(w in vt for w in YOUNG_WORDS))
        if any(w in vt for w in DEEP_WORDS):
            s -= 15
    elif p["age_group"] == "old":
        s += min(30, 10 * sum(w in vt for w in OLD_WORDS))
    s += min(24, 6 * sum(t in vt for t in p["traits"]))
    return s


def match_voices(profiles, voices):
    """Narrator first, then the most clearly described characters. Each gets the best unused voice."""
    cast, used = {}, set()
    narrator = max(voices, key=lambda v: score_voice(v, None, narrator=True))
    cast["Narrator"] = narrator
    used.add(narrator["voice_id"])

    chars = [p for p in profiles if p["name"] != "Narrator"]
    chars.sort(key=lambda p: -((p["gender"] != "unknown") * 2 + (p["age_group"] != "unknown")))
    for p in chars:
        best = sorted(voices, key=lambda v: score_voice(v, p) - (60 if v["voice_id"] in used else 0), reverse=True)[0]
        cast[p["name"]] = best
        used.add(best["voice_id"])
    return cast


# ---------------------------------------------------------------
# 5) Splitting text into {speaker, text} segments
# ---------------------------------------------------------------
def split_chunks(text, size=CHUNK_CHARS):
    units = []
    for line in text.split("\n"):
        units += [line] if len(line) <= size else re.split(r"(?<=[.!?])\s+", line)
    chunks, cur = [], ""
    for u in units:
        while len(u) > size:  # a single huge sentence: cut it
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(u[:size])
            u = u[size:]
        if cur and len(cur) + len(u) + 1 > size:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n{u}" if cur else u
    if cur.strip():
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


def segment_with_llm(text, profiles):
    known = "\n".join(
        f"- {p['name']}" + (f" (also called: {', '.join(p['aliases'])})" if p["aliases"] else "") for p in profiles
    ) or "- (none known yet)"
    prompt = f"""Split this book excerpt into consecutive segments for a multi-voice audiobook.
Known characters (use these exact names; map nicknames and titles to them):
{known}

Rules:
- Dialogue (inside quotation marks) belongs to the character speaking. Everything else is "Narrator".
- Speech tags like "asked Alice" or "cried the Rabbit" are narration: give them to "Narrator".
- If a speaker is not in the known list, use their real name from the text, never "Speaker".
- Keep the original text exactly, in order. Do not summarise or skip anything.
- Strip the quotation marks from dialogue text.
- Return ONLY JSON: {{"segments":[{{"speaker":"Narrator","text":"..."}},{{"speaker":"Alice","text":"..."}}]}}

EXCERPT:
{text}"""
    data = groq_json(prompt, max_tokens=8000)
    segs = [s for s in data.get("segments", []) if str(s.get("text", "")).strip()]
    if not segs:
        raise RuntimeError("Groq returned no segments")
    return [{"speaker": str(s.get("speaker", "Narrator")).strip() or "Narrator", "text": str(s["text"]).strip()} for s in segs]


# No LLM needed: quotes = dialogue, rest = narrator.
VERBS = "said|asked|replied|cried|shouted|whispered|exclaimed|answered|called|muttered"
AFTER_RE = re.compile(rf"^[\s,]*(?:{VERBS})\s+([A-Z][a-z]+)|^[\s,]*([A-Z][a-z]+)\s+(?:{VERBS})")
BEFORE_RE = re.compile(rf"([A-Z][a-z]+)\s+(?:{VERBS})[\s,:]*$")
QUOTE_SPLIT = re.compile(r'(["“][^"”]+["”])')


def segment_simple(text):
    parts = [p for p in QUOTE_SPLIT.split(text) if p.strip()]
    segments, names = [], []

    for i, part in enumerate(parts):
        if not re.match(r'^["“]', part):
            segments.append({"speaker": "Narrator", "text": part.strip()})
            continue
        nxt = parts[i + 1] if i + 1 < len(parts) else ""
        prv = parts[i - 1] if i > 0 else ""
        m, m2 = AFTER_RE.match(nxt), BEFORE_RE.search(prv)
        speaker = (m and (m.group(1) or m.group(2))) or (m2 and m2.group(1)) or "Speaker"
        if speaker not in names:
            names.append(speaker)
        segments.append({"speaker": speaker, "text": re.sub(r'^["“]|["”]$', "", part).strip()})

    characters = [{"name": "Narrator", "gender": "unknown"}] + [
        {"name": n, "gender": "unknown"} for n in names
    ]
    return {"characters": characters, "segments": segments}


# GET /api/config -> limits the frontend should know about
@app.get("/api/config")
def api_config():
    return jsonify(max_chars=MAX_CHARS, llm=bool(GROQ_KEY))


# POST /api/segment  { text }  ->  { mode, grounded, truncated, segments, cast, profiles }
@app.post("/api/segment")
def api_segment():
    try:
        body = request.get_json(silent=True) or {}
        full = str(body.get("text", "")).strip()
        text = full[:MAX_CHARS]
        if not text:
            return jsonify(error="text is required"), 400
        if not ELEVEN_KEY:
            return jsonify(error="ELEVENLABS_API_KEY missing in .env"), 500

        # Step 1: read the whole text (plus optional Wikipedia background) and build grounded profiles
        book = str(body.get("book", "")).strip()[:120]
        ref_source, ref_title, reference = fetch_reference(book) if (book and GROQ_KEY) else ("", "", "")
        profiles, grounded = [], False
        if GROQ_KEY:
            try:
                profiles, grounded = analyze_cast(text, book, reference), True
            except Exception as e:
                print("Cast analysis failed, continuing without profiles:", e)

        # Step 2: split into speaker segments, chunk by chunk, using the known cast
        segments, modes = [], set()
        for chunk in split_chunks(text):
            segs = None
            if GROQ_KEY:
                try:
                    segs = segment_with_llm(chunk, profiles)
                    modes.add("groq")
                except Exception as e:  # fall back to the simple splitter for this chunk
                    print("Groq segmentation failed for a chunk, using simple splitter:", e)
            if segs is None:
                segs = segment_simple(chunk)["segments"]
                modes.add("simple")
            segments += segs
        mode = "groq" if modes == {"groq"} else "simple" if modes == {"simple"} else "groq+simple"

        # Map nicknames to canonical names, and make sure every speaker has a profile
        alias_map = {}
        for p in profiles:
            alias_map[norm(p["name"])] = p["name"]
            for a in p["aliases"]:
                alias_map.setdefault(norm(a), p["name"])
        alias_map["narrator"] = "Narrator"
        for s in segments:
            s["speaker"] = alias_map.get(norm(s["speaker"]), s["speaker"])
        known = {p["name"] for p in profiles}
        for s in segments:
            if s["speaker"] != "Narrator" and s["speaker"] not in known:
                profiles.append(default_profile(s["speaker"]))
                known.add(s["speaker"])
        profiles = [p for p in profiles if p["name"] in {s["speaker"] for s in segments} or p["name"] == "Narrator"]

        # Step 3: match ElevenLabs voices to profiles
        voices = get_voices()
        cast = match_voices(profiles, library_voices(voices))
        cast, design = apply_voice_design(profiles, cast, ELEVEN_KEY, voices=voices, segments=segments, scope=book)
        designed = {n for n, d in design.items() if d["status"] == "designed"}
        if designed:  # re-match the others so library voices the designed characters would have taken stay free
            kept = match_voices([p for p in profiles if p["name"] not in designed], library_voices(voices))
            cast = {**kept, **{n: cast[n] for n in designed}}
        out = {p["name"]: {**p, "label": profile_label(p),
                           "voice_source": design.get(p["name"], {}).get("status", "library"),
                           "design_error": design.get(p["name"], {}).get("error", "")} for p in profiles}
        out["Narrator"] = {**default_profile("Narrator"), "label": "narrator"}
        return jsonify(mode=mode, grounded=grounded, truncated=len(full) > MAX_CHARS, reference=ref_title, reference_source=ref_source,
                       segments=segments, cast=cast, profiles=out)
    except Exception as e:
        print(e)
        return jsonify(error=str(e)), 500


# GET /api/voices -> list of voices (handy for a "change voice" dropdown)
@app.get("/api/voices")
def api_voices():
    try:
        return jsonify(get_voices())
    except Exception as e:
        return jsonify(error=str(e)), 500


# ---------------------------------------------------------------
# 6) Text-to-speech proxy with disk cache (saves your credits)
# ---------------------------------------------------------------
# POST /api/tts  { text, voice_id }  ->  audio/mpeg
@app.post("/api/tts")
def api_tts():
    try:
        body = request.get_json(silent=True) or {}
        text, voice_id = body.get("text"), body.get("voice_id")
        if not text or not voice_id:
            return jsonify(error="text and voice_id are required"), 400

        key = hashlib.sha1(f"{voice_id}|{text}".encode("utf-8")).hexdigest()
        path = os.path.join(CACHE_DIR, key + ".mp3")

        if os.path.exists(path):
            resp = send_file(path, mimetype="audio/mpeg")
            resp.headers["X-Cache"] = "HIT"
            return resp

        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": ELEVEN_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            timeout=120,
        )
        if not r.ok:
            return jsonify(error=f"ElevenLabs: {r.text}"), r.status_code

        with open(path, "wb") as f:
            f.write(r.content)
        resp = send_file(path, mimetype="audio/mpeg")
        resp.headers["X-Cache"] = "MISS"
        return resp
    except Exception as e:
        print(e)
        return jsonify(error=str(e)), 500


# ---------------------------------------------------------------
# Frontend (served from ./public)
# ---------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(os.path.join(BASE_DIR, "public"), "index.html")


# The page loads style.css and index.js from the public folder, so they must be served too.
@app.get("/<path:filename>")
def public_files(filename):
    return send_from_directory(os.path.join(BASE_DIR, "public"), filename)


if __name__ == "__main__":
    print(f"VoxTale running at http://localhost:{PORT}")
    print("Cast analysis:", "Groq GPT-OSS 120B" if GROQ_KEY else "OFF (no GROQ_API_KEY): simple rules only")
    print("Book lookup:", "Exa (falls back to Wikipedia)" if EXA_KEY else "Wikipedia only (no EXA_API_KEY)")
    print(f"Max text per request: {MAX_CHARS} characters")
    app.run(host="0.0.0.0", port=PORT, debug=True)