"""Score raw generations: strict/lenient answer extraction, correctness, refusal & hedge flags.

strict  = an explicit final-answer marker is required (JSON "answer", "Final Answer:",
          "## Answer", "Answer:", "the answer is", \\boxed{}). Mimics a production parser.
lenient = strict if available, else best-effort fallbacks (last number, bolded/initial option
          letter, last line). Approximates whether the model's content was correct.
Usage: python src/score.py  -> results/scored.parquet
"""
import glob
import json
import re
import string
from collections import Counter

import pandas as pd

LET = "ABCDEFGHIJKLMNOP"
NUM_RE = r"-?\$?\d[\d,]*(?:\.\d+)?"

REFUSAL_RE = re.compile(
    r"(\bI (?:can(?:'|’)?t|cannot|am unable to|am not able to|won(?:'|’)t) (?:help|assist|provide|answer|determine|comply|choose|select|give)"
    r"|\bI(?:'|’)m (?:unable|not able) to (?:help|assist|provide|answer|determine|choose|select)"
    r"|\bI must (?:decline|refuse)|\bunable to (?:assist|comply)"
    r"|\b(?:could|can) you (?:please )?(?:clarify|provide more)|\bplease (?:clarify|provide more (?:details|information|context))"
    r"|\b(?:not enough|insufficient) information to (?:determine|answer)"
    r"|\bnone of the (?:above|options|provided options) (?:is|are) (?:correct|accurate))",
    re.I)
HEDGE_RE = re.compile(
    r"(consult (?:a|with a|your|an?)\b|seek (?:professional|medical|legal)|qualified (?:professional|healthcare|attorney|lawyer)"
    r"|\bas an AI\b|\bI(?:'|’)m an AI\b|\bI am an AI\b|not (?:medical|legal|financial) advice|\bdisclaimer"
    r"|\bI(?:'|’)m not (?:completely |entirely |100% )?(?:sure|certain)|\bnot (?:completely|entirely) (?:sure|certain)"
    r"|\bit(?:'|’)s important to note|\bit is important to note|\bplease note\b|\bmay vary\b|\buncertain(?:ty)?\b"
    r"|\bcaution\b|\bcaveat)",
    re.I)


def strip_md(s):
    return s.replace("**", "").replace("__", "").replace("`", "").strip()


def strict_extract(text):
    """Return the explicitly marked final answer string, or None."""
    t = text.strip()
    # JSON object with an "answer" field
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "answer" in obj:
                return str(obj["answer"]).strip()
        except Exception:
            m2 = re.search(r'"answer"\s*:\s*"?([^"\n}]*)', m.group(0))
            if m2:
                return m2.group(1).strip()
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", t)
    if m:
        return m[-1].strip()
    # markdown "## Answer" section
    m = re.search(r"#+\s*(?:Final\s+)?Answer\s*:?\s*\n+(.+)", t, re.I)
    if m:
        return strip_md(m.group(1))
    m = re.findall(r"(?:Final Answer|Answer)\s*[:：]\s*\**\s*(.+)", t, re.I)
    if m:
        return strip_md(m[-1])
    m = re.findall(r"(?:correct|final|best) (?:answer|option|choice) is[:\s]*\**\s*(.+)", t, re.I)
    if m:
        return strip_md(m[-1])
    m = re.findall(r"\bthe answer is[:\s]*\**\s*(.+)", t, re.I)
    if m:
        return strip_md(m[-1])
    return None


def parse_options(user):
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"^([A-P])\. (.+)$", user, re.M)}


def to_letter(ans, options):
    """Map an answer string to an option letter (or None)."""
    if ans is None:
        return None
    a = strip_md(ans)
    m = re.match(r"^\W*(?:option\s+|choice\s+)?\(?([A-P])\)?(?:[.):,\s]|$)", a, re.I)
    if m and m.group(1).upper() in options:
        return m.group(1).upper()
    al = a.lower().strip(" .")
    for k, v in options.items():
        if al and (al == v.lower().strip(" .") or (len(al) > 8 and al in v.lower())):
            return k
    return None


def lenient_letter(text, options):
    s = strict_extract(text)
    L = to_letter(s, options)
    if L:
        return L
    t = text
    pats = [r"(?:answer|option|choice)[^.\n]{0,40}? is[:\s]*\**\s*(?:option\s+)?\(?([A-P])\b",
            r"\b(?:is|be)[:\s]+\**\s*(?:option\s+)?\(?([A-P])[.):]\s",  # "... would be C. Energy"
            r"\b(?:option|choice)\s+\(?([A-P])\)?[^.\n]{0,50}\b(?:correct|right|best|answer|accurate)",
            r"\b(?:go|going|went) with[.\s…:]*\**\s*(?:option\s+)?\(?([A-P])\b",  # "I'm going to go with... D."
            r"(?:\.\.\.|…)\s*\**\(?([A-P])[.):]\s",  # "... D. sediment"
            r"\b([A-P])\)? is (?:the )?(?:correct|right|best|most accurate)",
            r"\*\*\(?([A-P])[.)]?\s",  # bolded letter, e.g. **C. text**
            r"\*\*\(?([A-P])\)?\*\*"]
    for p in pats:
        m = re.findall(p, t)
        m = [x for x in m if x in options]
        if m:
            return m[-1]
    exact = []
    for line in t.split("\n"):
        m = re.match(r"^\W*\(?([A-P])[.)]\s*(.+?)\s*$", strip_md(line))
        if m and m.group(1) in options and m.group(2).lower().strip(" .") == options[m.group(1)].lower().strip(" ."):
            exact.append(m.group(1))
    if exact and len(set(exact)) <= 2:
        return exact[-1]
    first = strip_md(t.strip().split("\n")[0])  # responses like "C. Influenza and HIV"
    L = to_letter(first, options)
    if L:
        return L
    # option text quoted exactly once among options in the last line
    last = strip_md([l for l in t.strip().split("\n") if l.strip()][-1]) if t.strip() else ""
    L = to_letter(last, options)
    if L:
        return L
    hits = [k for k, v in options.items() if len(v) > 3 and v.lower() in t.lower()]
    if len(hits) == 1:
        return hits[0]
    # explicit letter references in the closing part of the response; accept only if unambiguous
    tail = t[-300:]
    refs = re.findall(r"\b(?:option|choice)\s+\(?([A-P])\b|\(([A-P])\)|(?:^|[\s.…)*])([A-P])[.)]\s", tail)
    refs = {x for tup in refs for x in tup if x and x in options}
    if len(refs) == 1:
        return refs.pop()
    return fuzzy_letter(t, options)


STOP = set("the and for are but not you all any can had her was one our out has his how its may new now old see "
           "two way who did get let say she too use with that this from they will would there their what which "
           "when were been have more some than them then very into only also because about".split())


def content_tokens(s):
    return {w for w in norm_ans(s) if len(w) >= 3 and w not in STOP}


def fuzzy_letter(text, options, min_recall=0.6, margin=0.25):
    """Last-resort match for prose answers that restate an option without its letter: pick the option
    whose content words are best covered by the first two sentences, if the winner is clear."""
    head = " ".join(re.split(r"(?<=[.!?])\s+", strip_md(text).strip())[:2])
    ht = content_tokens(head)
    scores = []
    for k, v in options.items():
        ot = content_tokens(v)
        if ot:
            scores.append((len(ot & ht) / len(ot), k))
    if not scores:
        return None
    scores.sort(reverse=True)
    top = scores[0][0]
    second = scores[1][0] if len(scores) > 1 else 0.0
    return scores[0][1] if top >= min_recall and top - second >= margin else None


def first_num(s):
    if s is None:
        return None
    m = re.findall(NUM_RE, s)
    if not m:
        return None
    try:
        return float(m[0].replace("$", "").replace(",", ""))
    except ValueError:
        return None


def last_num(s):
    m = re.findall(NUM_RE, s or "")
    for x in reversed(m):
        try:
            return float(x.replace("$", "").replace(",", ""))
        except ValueError:
            continue
    return None


def norm_ans(s):
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def f1(pred, gold):
    p, g = norm_ans(pred), norm_ans(gold)
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def span_correct(pred, gold, max_words=60):
    """DROP (chatty models): correct if the gold answer is contained in the answer text.
    Numeric gold: some number in the answer equals it. Text gold: all gold tokens appear.
    Answers longer than `max_words` are not credited (avoids rewarding dumps of the passage)."""
    if pred is None or not pred.strip():
        return False, 0.0
    sc = f1(pred, gold)
    if len(pred.split()) > max_words:
        return False, sc
    if re.fullmatch(r"\s*" + NUM_RE + r"\s*", gold):
        g = first_num(gold)
        nums = [float(x.replace("$", "").replace(",", "")) for x in re.findall(NUM_RE, pred) if re.search(r"\d", x)]
        words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
        nums += [v for w, v in words.items() if re.search(r"\b" + w + r"\b", pred, re.I)]
        return any(abs(n - g) < 1e-6 for n in nums), sc
    gt, pt = norm_ans(gold), set(norm_ans(pred))
    return bool(gt) and all(w in pt for w in gt), sc


def score_row(r, it):
    text = r["response"]
    s = strict_extract(text)
    out = dict(strict_found=s is not None)
    if it["kind"] == "mc":
        opts = parse_options(it["user"])
        ls = to_letter(s, opts)
        ll = lenient_letter(text, opts)
        out.update(pred_strict=ls, pred_lenient=ll, correct_strict=ls == it["gold"], correct_lenient=ll == it["gold"],
                   f1=float(ll == it["gold"]))
    elif it["kind"] == "number":
        g = float(it["gold"])
        ns = first_num(s)
        nl = ns if ns is not None else last_num(text)
        out.update(pred_strict=ns, pred_lenient=nl,
                   correct_strict=ns is not None and abs(ns - g) < 1e-6,
                   correct_lenient=nl is not None and abs(nl - g) < 1e-6)
        out["f1"] = float(out["correct_lenient"])
    else:  # span (DROP)
        cs, fs = span_correct(s, it["gold"])
        cands = [s] if s is not None else []
        lines = [strip_md(l) for l in text.strip().split("\n") if l.strip()]
        if not cands:
            # first/last line plus first/last sentence (so verbose paragraphs are not penalised), bold spans
            sents = [x for x in re.split(r"(?<=[.!?])\s+", " ".join(lines)) if x.strip()]
            cands = lines[-1:] + lines[:1] + sents[:1] + sents[-1:] + re.findall(r"\*\*(.+?)\*\*", text)
        best = (False, 0.0)
        for c in cands:
            cc = span_correct(c, it["gold"])
            best = max(best, cc)
        out.update(pred_strict=s, pred_lenient=None, correct_strict=cs, correct_lenient=best[0], f1=best[1])
    out["refusal"] = bool(REFUSAL_RE.search(text))
    out["hedge"] = bool(HEDGE_RE.search(text))
    out["n_hedges"] = len(HEDGE_RE.findall(text))
    out["no_answer_lenient"] = out.get("pred_lenient") is None if it["kind"] != "span" else (not text.strip())
    return out


def main():
    items = {json.loads(l)["id"]: json.loads(l) for l in open("results/items.jsonl")}
    rows = []
    for path in sorted(glob.glob("results/raw/*.jsonl")):
        for l in open(path):
            r = json.loads(l)
            it = items[r["id"]]
            sc = score_row(r, it)
            rows.append({**{k: r[k] for k in ["model", "tag", "condition", "category", "task", "id", "n_tokens",
                                                "n_prompt_tokens", "finish_reason"]},
                         **{k: v for k, v in sc.items() if not k.startswith("pred")},
                         "pred_strict": str(sc.get("pred_strict")), "pred_lenient": str(sc.get("pred_lenient")),
                         "gold": str(it["gold"]), "subject": it["meta"].get("subject")})
    df = pd.DataFrame(rows)
    df.to_parquet("results/scored.parquet")
    print(df.groupby(["model", "tag"]).size())
    print(df.groupby(["model", "tag", "task"])[["correct_strict", "correct_lenient"]].mean().round(3))


if __name__ == "__main__":
    main()
