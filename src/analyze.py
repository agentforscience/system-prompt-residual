"""Statistical analysis + figures for the system-prompt residual-effects experiment.

Reads results/scored.parquet, writes results/*.csv, results/summary.json and figures/*.png.
All effects are paired differences vs. the baseline condition (default `neutral_short`),
in percentage points. CIs come from a paired bootstrap that resamples *items* (the same
resampled items are used for every model/condition, preserving the pairing).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from prompts import CATEGORY_OF, SYSTEM_PROMPTS  # noqa: E402

B = 2000
RNG = np.random.default_rng(42)
BASE = "neutral_short"
CONDS = list(SYSTEM_PROMPTS)
CATS = ["safety", "persona", "format", "constraint"]
TASKS = ["gsm8k", "arc", "mmlu_sens", "truthfulqa", "drop"]
TASK_LABEL = {"gsm8k": "GSM8K", "arc": "ARC-C", "mmlu_sens": "MMLU-sensitive", "truthfulqa": "TruthfulQA-MC1", "drop": "DROP"}
STRUCTURED = ["arc", "mmlu_sens", "truthfulqa", "drop"]  # tasks needing a discrete extractable answer
os.makedirs("figures", exist_ok=True)


def cube(df, metric, models):
    """Return {task: array[model, condition, item]} for one metric."""
    out = {}
    for t in TASKS:
        d = df[df.task == t]
        ids = sorted(d.id.unique())
        p = d.pivot_table(index=["model", "condition"], columns="id", values=metric, aggfunc="first")
        arr = np.stack([np.stack([p.loc[(m, c), ids].to_numpy(dtype=float) for c in CONDS]) for m in models])
        out[t] = arr
    return out


def boot_idx(n):
    return RNG.integers(0, n, size=(B, n))


def mcnemar_p(a, b):
    """Exact McNemar test on paired binary vectors."""
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    if n01 + n10 == 0:
        return 1.0
    return binomtest(n01, n01 + n10, 0.5).pvalue


def analyze(df, tag, metric="correct_lenient"):
    d = df[df.tag == tag]
    models = sorted(d.model.unique())
    C = {c: i for i, c in enumerate(CONDS)}
    acc = cube(d, metric, models)
    toks = cube(d, "n_tokens", models)
    idxs = {t: boot_idx(acc[t].shape[2]) for t in TASKS}  # shared resamples across metrics
    bi = C[BASE]
    res = {"models": models}

    # ---------- per (model, task, condition) effects ----------
    rows = []
    for t in TASKS:
        for mi, m in enumerate(models):
            base = acc[t][mi, bi]
            for c in CONDS:
                x = acc[t][mi, C[c]]
                dlt = (x - base)
                bs = dlt[idxs[t]].mean(1)
                rows.append(dict(model=m, task=t, condition=c, category=CATEGORY_OF[c], acc=100 * x.mean(),
                                 base_acc=100 * base.mean(), delta=100 * dlt.mean(),
                                 lo=100 * np.percentile(bs, 2.5), hi=100 * np.percentile(bs, 97.5),
                                 p_mcnemar=mcnemar_p(x, base) if c != BASE else 1.0,
                                 mean_tokens=toks[t][mi, C[c]].mean(),
                                 len_ratio=toks[t][mi, C[c]].mean() / toks[t][mi, bi].mean()))
    cell = pd.DataFrame(rows)
    cell.to_csv(f"results/effects_cell_{tag}_{metric}.csv", index=False)

    # ---------- category effects (pooled over variants and models) ----------
    def cat_effect(conds, tasks):
        """mean delta over conds x models x tasks (tasks weighted equally), with bootstrap CI."""
        ci = [C[c] for c in conds]
        point, bs = [], []
        for t in tasks:
            a = acc[t]
            dl = a[:, ci, :].mean(axis=(0, 1)) - a[:, bi, :].mean(axis=0)  # per-item delta
            point.append(dl.mean())
            bs.append(dl[idxs[t]].mean(1))
        point, bs = 100 * np.mean(point), 100 * np.mean(bs, axis=0)
        p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        return point, np.percentile(bs, 2.5), np.percentile(bs, 97.5), max(p, 1 / B)

    def len_ratio(conds, tasks):
        ci = [C[c] for c in conds]
        num = sum(toks[t][:, ci, :].mean(axis=(0, 1)) for t in tasks)  # not used directly
        pt, bs = [], []
        for t in tasks:
            a = toks[t]
            x, y = a[:, ci, :].mean(axis=(0, 1)), a[:, bi, :].mean(axis=0)
            pt.append(x.mean() / y.mean())
            bs.append(x[idxs[t]].mean(1) / y[idxs[t]].mean(1))
        del num
        pt, bs = np.mean(pt), np.mean(bs, axis=0)
        return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    crow = []
    groups = {**{cat: [c for c in CONDS if CATEGORY_OF[c] == cat] for cat in CATS},
              "none": ["none"], "neutral_long": ["neutral_long"]}
    for g, conds in groups.items():
        for t in TASKS + ["ALL", "STRUCTURED"]:
            ts = TASKS if t == "ALL" else STRUCTURED if t == "STRUCTURED" else [t]
            e, lo, hi, p = cat_effect(conds, ts)
            lr, llo, lhi = len_ratio(conds, ts)
            crow.append(dict(group=g, task=t, delta=e, lo=lo, hi=hi, p_boot=p, len_ratio=lr, len_lo=llo, len_hi=lhi))
    cat = pd.DataFrame(crow)
    # Holm correction across the 4 categories x 5 tasks family
    fam = cat[cat.group.isin(CATS) & cat.task.isin(TASKS)].index
    cat.loc[fam, "p_holm"] = multipletests(cat.loc[fam, "p_boot"], method="holm")[1]
    cat.to_csv(f"results/effects_category_{tag}_{metric}.csv", index=False)

    # ---------- H4: dispersion of effects across tasks/models ----------
    disp = []
    for g in CATS:
        conds = groups[g]
        sub = cell[cell.condition.isin(conds)]
        # variant-averaged effect per (model, task)
        mt = sub.groupby(["model", "task"]).delta.mean()
        tk = sub.groupby("task").delta.mean()
        # bootstrap SD across tasks of the pooled task effect
        ci = [C[c] for c in conds]
        bs_sd = np.std([np.stack([100 * (acc[t][:, ci][:, :, idxs[t][b]].mean() - acc[t][:, bi][:, idxs[t][b]].mean())
                                  for t in TASKS]) for b in range(0, B, 4)], axis=1, ddof=1)
        disp.append(dict(category=g, sd_task=tk.std(ddof=1), sd_task_lo=np.percentile(bs_sd, 2.5),
                         sd_task_hi=np.percentile(bs_sd, 97.5), sd_model_task=mt.std(ddof=1),
                         sd_cell=sub.delta.std(ddof=1), range_task=tk.max() - tk.min(),
                         min_cell=sub.delta.min(), max_cell=sub.delta.max(),
                         n_cells_le_m10=int((sub.delta <= -10).sum()), n_cells_pos_sig=int(((sub.lo > 0)).sum()),
                         n_cells_neg_sig=int(((sub.hi < 0)).sum()), n_cells=len(sub)))
    disp = pd.DataFrame(disp)
    disp.to_csv(f"results/dispersion_{tag}_{metric}.csv", index=False)

    # ---------- behavioural rates ----------
    beh = d.groupby(["condition"])[["refusal", "hedge", "no_answer_lenient", "strict_found"]].mean() * 100
    beh["category"] = [CATEGORY_OF[c] for c in beh.index]
    beh["mean_tokens"] = d.groupby("condition").n_tokens.mean()
    beh["truncated"] = d.groupby("condition").finish_reason.apply(lambda s: (s == "length").mean() * 100)
    beh = beh.loc[CONDS]
    beh.to_csv(f"results/behaviour_{tag}.csv")
    beh_task = d.groupby(["category", "task"])[["refusal", "hedge"]].mean().unstack() * 100
    beh_task.to_csv(f"results/behaviour_by_task_{tag}.csv")

    res.update(cell=cell, cat=cat, disp=disp, beh=beh)
    return res


def figures(r, tag, metric):
    cell, cat, disp, beh = r["cell"], r["cat"], r["disp"], r["beh"]
    sfx = f"{tag}_{metric}"
    # 1. Heatmap of pooled-over-models delta (condition x task)
    pooled = cell.groupby(["condition", "task"]).delta.mean().unstack()[TASKS].loc[[c for c in CONDS if c != BASE]]
    fig, ax = plt.subplots(figsize=(8, 7))
    v = np.nanmax(np.abs(pooled.values))
    im = ax.imshow(pooled.values, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(TASKS)), [TASK_LABEL[t] for t in TASKS], rotation=20)
    ax.set_yticks(range(len(pooled)), pooled.index)
    for i in range(pooled.shape[0]):
        for j in range(pooled.shape[1]):
            ax.text(j, i, f"{pooled.values[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label=f"Δ accuracy vs '{BASE}' (pp), mean of 3 models")
    ax.set_title(f"Accuracy change by system prompt ({metric.replace('correct_', '')} scoring, {tag})")
    plt.tight_layout(); plt.savefig(f"figures/heatmap_delta_{sfx}.png", dpi=150); plt.close()

    # 2. Category effects per task with CIs
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = {"safety": "#c44e52", "persona": "#8172b2", "format": "#55a868", "constraint": "#dd8452"}
    tl = TASKS + ["ALL"]
    w = 0.2
    for k, g in enumerate(CATS):
        s = cat[(cat.group == g)].set_index("task").loc[tl]
        x = np.arange(len(tl)) + (k - 1.5) * w
        ax.bar(x, s.delta, w, color=colors[g], label=g)
        ax.errorbar(x, s.delta, yerr=[s.delta - s.lo, s.hi - s.delta], fmt="none", ecolor="k", lw=0.8, capsize=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(tl)), [TASK_LABEL.get(t, "All tasks") for t in tl])
    ax.set_ylabel(f"Δ accuracy vs '{BASE}' (pp)")
    ax.set_title("Category effects (pooled over 3 variants × 3 models; 95% paired-bootstrap CI)")
    ax.legend(ncol=4, fontsize=8)
    plt.tight_layout(); plt.savefig(f"figures/category_effects_{sfx}.png", dpi=150); plt.close()

    # 3. Length ratios
    fig, ax = plt.subplots(figsize=(7, 4))
    gs = ["none", "neutral_long"] + CATS
    s = cat[(cat.task == "ALL")].set_index("group").loc[gs]
    ax.bar(gs, s.len_ratio, color=["#999", "#bbb"] + [colors[g] for g in CATS])
    ax.errorbar(gs, s.len_ratio, yerr=[s.len_ratio - s.len_lo, s.len_hi - s.len_ratio], fmt="none", ecolor="k", capsize=3)
    ax.axhline(1, color="k", lw=0.6); ax.axhspan(1.3, 1.5, color="#8172b2", alpha=0.12, label="H2 predicted range (persona)")
    ax.set_ylabel(f"Output length / '{BASE}' length"); ax.legend(fontsize=8)
    ax.set_title("Output-length ratio (mean over tasks & models)")
    plt.tight_layout(); plt.savefig(f"figures/length_ratio_{tag}.png", dpi=150); plt.close()

    # 4. Effect distribution per category (H4)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, g in enumerate(CATS):
        sub = cell[cell.category == g]
        xs = k + RNG.uniform(-0.18, 0.18, len(sub))
        ax.scatter(xs, sub.delta, s=12, alpha=0.6, color=colors[g])
        ax.boxplot(sub.delta, positions=[k], widths=0.5, showfliers=False)
    ax.axhline(0, color="k", lw=0.6); ax.axhline(-10, color="gray", ls="--", lw=0.6)
    ax.set_xticks(range(4), CATS); ax.set_ylabel("Δ accuracy (pp), one point per model×task×variant")
    ax.set_title("Spread of effects by category")
    plt.tight_layout(); plt.savefig(f"figures/effect_spread_{sfx}.png", dpi=150); plt.close()

    # 5. Refusal / hedge rates
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(beh))
    ax.bar(x - 0.2, beh.hedge, 0.4, label="hedge/caveat phrase present", color="#4c72b0")
    ax.bar(x + 0.2, beh.refusal, 0.4, label="refusal / clarification request", color="#c44e52")
    ax.set_xticks(x, beh.index, rotation=45, ha="right"); ax.set_ylabel("% of responses"); ax.legend(fontsize=8)
    ax.set_title("Hedging and refusal by system prompt (all tasks, all models)")
    plt.tight_layout(); plt.savefig(f"figures/hedge_refusal_{tag}.png", dpi=150); plt.close()


def main():
    df = pd.read_parquet("results/scored.parquet")
    summary = {}
    for tag in sorted(df.tag.unique()):
        for metric in ["correct_lenient", "correct_strict"]:
            r = analyze(df, tag, metric)
            figures(r, tag, metric)
            summary[f"{tag}/{metric}"] = dict(
                models=r["models"],
                category=r["cat"].round(3).to_dict(orient="records"),
                dispersion=r["disp"].round(3).to_dict(orient="records"))
            print(f"\n===== {tag} / {metric} =====")
            print(r["cat"][r["cat"].task.isin(["ALL", "STRUCTURED"] + TASKS)].round(2).to_string(index=False))
            print(r["disp"].round(2).to_string(index=False))
        print(r["beh"].round(1).to_string())
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)


if __name__ == "__main__":
    main()
