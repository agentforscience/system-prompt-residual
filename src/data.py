"""Load and sample the five evaluation tasks from the local HF cache.

Each item is a dict: {task, id, user, gold, kind, meta}. The user turn carries only a minimal
instruction so that the system prompt is the only thing that varies between conditions.
"""
import glob
import json
import os
import random
import re

import numpy as np
import pandas as pd

HUB = os.environ.get("HF_HUB_DIR", "/data/transformers/hub")
LETTERS = "ABCDEFGHIJKLMNOP"
# MMLU subjects where safety-oriented system prompts plausibly trigger hedging/refusal.
SENSITIVE_SUBJECTS = [
    "virology", "medical_genetics", "clinical_knowledge", "professional_medicine",
    "college_medicine", "nutrition", "human_sexuality", "computer_security",
    "security_studies", "moral_scenarios", "moral_disputes", "professional_law",
    "jurisprudence", "professional_psychology",
]


def _pq(pattern):
    files = glob.glob(os.path.join(HUB, pattern))
    assert files, f"missing dataset file: {pattern}"
    return pd.read_parquet(files[0])


def _mc_user(question, options):
    lines = [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
    return "Answer the following multiple-choice question.\n\n" + question.strip() + "\n\n" + "\n".join(lines)


def load_gsm8k(n, rng):
    df = _pq("datasets--openai--gsm8k/snapshots/*/main/test-*.parquet")
    idx = rng.sample(range(len(df)), n)
    out = []
    for i in idx:
        r = df.iloc[i]
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append(dict(task="gsm8k", id=f"gsm8k-{i}", kind="number", gold=gold,
                        user="Solve the following math problem.\n\n" + r["question"].strip(), meta={}))
    return out


def load_arc(n, rng):
    df = _pq("datasets--allenai--ai2_arc/snapshots/*/ARC-Challenge/test-*.parquet")
    df = df[df["choices"].apply(lambda c: list(c["label"]) in (list("ABCD"), list("ABCDE"), list("ABC")))]
    idx = rng.sample(list(df.index), n)
    out = []
    for i in idx:
        r = df.loc[i]
        opts = list(r["choices"]["text"])
        gold = r["answerKey"]
        out.append(dict(task="arc", id=f"arc-{r['id']}", kind="mc", gold=gold, n_opts=len(opts),
                        user=_mc_user(r["question"], opts), meta={}))
    return out


def load_mmlu_sensitive(n, rng):
    df = _pq("datasets--cais--mmlu/snapshots/*/all/test-*.parquet")
    df = df[df["subject"].isin(SENSITIVE_SUBJECTS)]
    idx = rng.sample(list(df.index), n)
    out = []
    for i in idx:
        r = df.loc[i]
        opts = list(r["choices"])
        out.append(dict(task="mmlu_sens", id=f"mmlu-{i}", kind="mc", gold=LETTERS[int(r["answer"])],
                        n_opts=len(opts), user=_mc_user(r["question"], opts), meta={"subject": r["subject"]}))
    return out


def load_truthfulqa(n, rng):
    df = _pq("datasets--truthful_qa/snapshots/*/multiple_choice/validation-*.parquet")
    idx = rng.sample(range(len(df)), n)
    out = []
    for i in idx:
        r = df.iloc[i]
        ch = list(r["mc1_targets"]["choices"])
        lab = list(r["mc1_targets"]["labels"])
        ch, lab = ch[:6], lab[:6]  # cap at 6 options (correct is always index 0 in the raw data)
        order = list(range(len(ch)))
        rng.shuffle(order)  # shuffle so the correct option is not always "A"
        opts = [ch[j] for j in order]
        gold = LETTERS[[lab[j] for j in order].index(1)]
        out.append(dict(task="truthfulqa", id=f"tqa-{i}", kind="mc", gold=gold, n_opts=len(opts),
                        user=_mc_user(r["question"], opts), meta={}))
    return out


def _norm(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9. ]", " ", s)
    return " ".join(s.split())


def load_drop(n, rng):
    df = _pq("datasets--ucinlp--drop/snapshots/*/data/validation-*.parquet")
    # keep items with a single unambiguous gold answer (one unique normalised span)
    keep = df["answers_spans"].apply(lambda a: len({_norm(s) for s in a["spans"]}) == 1)
    df = df[keep]
    idx = rng.sample(list(df.index), n)
    out = []
    for i in idx:
        r = df.loc[i]
        gold = r["answers_spans"]["spans"][0]
        typ = r["answers_spans"]["types"][0]
        out.append(dict(task="drop", id=f"drop-{r['query_id']}", kind="span", gold=gold,
                        user=("Read the passage and answer the question.\n\nPassage: " + r["passage"].strip()
                              + "\n\nQuestion: " + r["question"].strip()),
                        meta={"type": typ}))
    return out


def load_all(n_per_task=300, seed=42):
    items = []
    for f in [load_gsm8k, load_arc, load_mmlu_sensitive, load_truthfulqa, load_drop]:
        rng = random.Random(seed)
        items += f(n_per_task, rng)
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    return items


if __name__ == "__main__":
    items = load_all()
    os.makedirs("results", exist_ok=True)
    with open("results/items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, default=str) + "\n")
    df = pd.DataFrame(items)
    print(df.groupby("task").size())
    print(df[df.task != "gsm8k"].groupby("task")["gold"].value_counts().unstack().fillna(0).iloc[:, :8])
    for t in df.task.unique():
        print("=" * 20, t)
        print(df[df.task == t].iloc[0]["user"][:600], "\nGOLD:", df[df.task == t].iloc[0]["gold"])
