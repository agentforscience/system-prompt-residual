# System Prompt Residual Effects on Downstream Task Performance

This is a controlled study of how four common categories of system prompt (safety, persona,
format, constraint) change accuracy, output length and refusals on tasks they weren't written
for. It covers 15 system prompts, 5 benchmarks (GSM8K, ARC-C, MMLU-sensitive subjects,
TruthfulQA-MC1, DROP) and 3 open models (Qwen2.5-7B, Llama-3.1-8B, Qwen3-4B), all run
locally with vLLM. That is 67,500 greedy generations plus a full 67,500-generation T=0.7 replicate.

## Key findings (lenient accuracy vs. "You are a helpful assistant.", pooled over models)
- **Safety prompts cost ~nothing:** +0.2 pp [−0.4, 0.9]. TruthfulQA improves by +3.7 pp.
  Hedging rises (up to 7% of responses), refusals stay ≤1%. The hypothesized −4 to −8 pp
  appears only under a strict answer parser, and comes from answer formatting, not reasoning.
- **Persona prompts double output length** (×1.99, vs. the hypothesized +30–50%) and lower
  accuracy by −3.1 pp. Qwen3-4B was immune to the accuracy loss.
- **Format prompts make answers parseable, not more correct.** Strict-parser accuracy is
  +19.6 pp, but content accuracy is −2.5 pp. JSON/sectioned formats shorten math reasoning
  (GSM8K −4.2 pp). "Answer with the option letter" rules spill over onto open questions
  (single-letter DROP answers).
- **Constraint prompts are the most harmful:** −9.0 pp overall, as low as −61 pp, with the
  widest spread of effects. They never helped on any task.
- **A deployment blurb with no instructions** made Llama refuse off-topic questions
  ("not related to the Northwind Learning Hub").

- **Robust to sampling:** in the full T=0.7 replicate, every lenient category effect lands within
  0.6 pp of greedy (cell-level r = 0.95), and every hypothesis verdict is unchanged.

Full details, tables and figures are in **[REPORT.md](REPORT.md)**.

## Reproduce
```bash
uv venv --python 3.12 && source .venv/bin/activate && uv sync   # vllm 0.31, torch 2.13
export VLLM_USE_FLASHINFER_SAMPLER=0                            # FlashInfer JIT fails on this host
python src/data.py                       # sample 5×300 items from the local HF cache -> results/items.jsonl
bash src/run_all.sh                      # all models × 15 conditions (greedy + T=0.7), GPU-adaptive, checkpointed
#   or: python src/run.py --model qwen25_7b --tag greedy
python src/score.py                      # strict/lenient scoring -> results/scored.parquet
python src/analyze.py                    # effects, bootstrap CIs, Holm, figures -> results/, figures/
python src/error_analysis.py             # flip taxonomy, spill-over, robustness -> results/error_*.csv
```
Datasets and models are read from the local HF cache (`/data/transformers/hub`). Set
`HF_HUB_DIR` or edit the paths in `src/data.py` and `src/run.py` to point elsewhere.

## Layout
```
planning.md              research plan & motivation
REPORT.md                full report
src/prompts.py           the 15 system prompts
src/data.py              task loading/sampling
src/run.py, run_all.sh   vLLM generation (checkpointed; adaptive to shared-GPU memory)
src/score.py             answer extraction, correctness, refusal/hedge flags
src/analyze.py           statistics + figures
src/error_analysis.py    error taxonomy and robustness checks
results/raw/             raw generations (one JSONL per model × decoding setting) + metadata
results/*.csv            effect tables (cell, category, dispersion, behaviour, errors)
figures/                 plots
logs/                    run logs
```
