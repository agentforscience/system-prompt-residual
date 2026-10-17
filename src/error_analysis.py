"""Error analysis: *how* each system prompt flips baseline-correct items to wrong.

For every (model, condition) we take items the model got right under the baseline
(`neutral_short`) and wrong under the condition ("flips"), and classify each flip as:
  refusal       - refusal / clarification-request phrase present
  no_answer     - no answer could be extracted (lenient)
  hedged_wrong  - answered wrongly, with hedge/caveat phrases
  wrong_answer  - answered wrongly, no hedge
Also: format "spill-over" on DROP (single-letter answers to open questions), safety effects
by MMLU subject, and greedy-vs-sampled agreement of effects.
Writes results/error_*.csv and figures/flip_types.png.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from prompts import SYSTEM_PROMPTS  # noqa: E402

BASE = "neutral_short"
CONDS = [c for c in SYSTEM_PROMPTS if c != BASE]


def main():
    df = pd.read_parquet("results/scored.parquet")
    g = df[df.tag == "greedy"].copy()
    base = g[g.condition == BASE].set_index(["model", "id"])

    # ---- flip taxonomy ----
    rows = []
    for c in CONDS:
        d = g[g.condition == c].set_index(["model", "id"])
        b = base.loc[d.index]
        flips = d[b.correct_lenient.values & ~d.correct_lenient.values]
        gains = d[~b.correct_lenient.values & d.correct_lenient.values]
        typ = np.where(flips.refusal, "refusal",
                       np.where(flips.no_answer_lenient, "no_answer",
                                np.where(flips.hedge, "hedged_wrong", "wrong_answer")))
        r = dict(condition=c, category=d.category.iloc[0], n_items=len(d), n_flips_to_wrong=len(flips),
                 n_flips_to_right=len(gains), net_pp=100 * (len(gains) - len(flips)) / len(d))
        for t in ["refusal", "no_answer", "hedged_wrong", "wrong_answer"]:
            r[t] = int((typ == t).sum())
        rows.append(r)
    flips = pd.DataFrame(rows)
    flips.to_csv("results/error_flip_types.csv", index=False)
    print(flips.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = np.zeros(len(flips))
    for t, col in zip(["refusal", "no_answer", "hedged_wrong", "wrong_answer"], ["#c44e52", "#dd8452", "#8172b2", "#4c72b0"]):
        ax.bar(flips.condition, flips[t], bottom=bottom, label=t, color=col)
        bottom += flips[t].values
    ax.plot(flips.condition, flips.n_flips_to_right, "k_", ms=18, mew=2, label="flips wrong→right")
    ax.set_ylabel("# items (all models, all tasks)")
    ax.set_title(f"Items correct under '{BASE}' that become wrong, by failure type")
    plt.xticks(rotation=45, ha="right"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("figures/flip_types.png", dpi=150); plt.close()

    # ---- format spill-over on DROP: one-letter answers to open-ended questions ----
    drop = g[g.task == "drop"]
    spill = drop.assign(single_letter=drop.pred_strict.str.fullmatch(r"\(?[A-Z]\)?\.?")).groupby(
        ["model", "condition"]).single_letter.mean().unstack(0) * 100
    spill.round(1).to_csv("results/error_drop_single_letter.csv")
    print("\n% of DROP answers that are a single capital letter:\n", spill.round(1).to_string())

    # ---- safety effect by MMLU subject (pooled models & safety variants) ----
    mm = g[g.task == "mmlu_sens"]
    acc = mm.groupby(["subject", "category"]).correct_lenient.mean().unstack() * 100
    bs = mm[mm.condition == BASE].groupby("subject").correct_lenient.mean() * 100
    subj = pd.DataFrame({"baseline": bs, "safety": acc["safety"], "n_items": mm[mm.condition == BASE].groupby("subject").size()
                         / mm.model.nunique()})
    subj["delta"] = subj.safety - subj.baseline
    subj.round(1).sort_values("delta").to_csv("results/error_safety_by_subject.csv")
    print("\nSafety effect by MMLU subject:\n", subj.round(1).sort_values("delta").to_string())

    # ---- per-model category effects (all tasks, lenient) ----
    cell = pd.read_csv("results/effects_cell_greedy_correct_lenient.csv")
    pm = cell[cell.condition != BASE].groupby(["category", "model"]).agg(delta=("delta", "mean"),
                                                                       len_ratio=("len_ratio", "mean")).unstack()
    pm.round(2).to_csv("results/error_per_model_category.csv")
    print("\nPer-model category effects (pp, mean over tasks & variants) and length ratios:\n", pm.round(2).to_string())

    # ---- robustness: accuracy on items answered under BOTH baseline and condition ----
    rows = []
    for c in CONDS:
        d = g[g.condition == c].set_index(["model", "id"])
        b = base.loc[d.index]
        ok = ~d.no_answer_lenient.values & ~b.no_answer_lenient.values
        dd = d[ok]
        bb = b[ok]
        per_task = (dd.correct_lenient.groupby(dd.task).mean() - bb.correct_lenient.groupby(bb.task).mean()) * 100
        rows.append(dict(condition=c, category=d.category.iloc[0], frac_kept=ok.mean(), delta_answered=per_task.mean()))
    ans = pd.DataFrame(rows)
    ans.to_csv("results/error_answered_only.csv", index=False)
    print("\nEffect restricted to items answered in both conditions (pp, mean over tasks):\n",
          ans.groupby("category").delta_answered.mean().round(2).to_string())

    # ---- greedy vs sampled agreement ----
    if (df.tag == "t07s1").any():
        import glob
        out = {}
        for f in ["correct_lenient"]:
            a = pd.read_csv(f"results/effects_cell_greedy_{f}.csv")
            b = pd.read_csv(f"results/effects_cell_t07s1_{f}.csv")
            m = a.merge(b, on=["model", "task", "condition"], suffixes=("_g", "_s"))
            m = m[m.condition != BASE]
            out[f] = dict(n=len(m), pearson=m.delta_g.corr(m.delta_s), spearman=m.delta_g.corr(m.delta_s, method="spearman"),
                          sign_agree=float((np.sign(m.delta_g) == np.sign(m.delta_s)).mean()))
            m.to_csv("results/error_greedy_vs_sampled.csv", index=False)
        print("\nGreedy vs T=0.7 effect agreement:", out)
        del glob


if __name__ == "__main__":
    main()
