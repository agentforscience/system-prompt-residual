# Planning: System Prompt Residual Effects on Downstream Task Performance

## Motivation & Novelty Assessment

### Why This Research Matters
Deployed LLMs almost never run "bare": product teams prepend system prompts with safety
policies, brand personas, output-format rules and behavioural constraints. These prompts are
written for one purpose (safety, tone, parsing) but stay in context for *every* request, so
they can leave "residual" effects on unrelated tasks (accuracy, verbosity, refusals). Knowing
which categories of system prompt are cheap and which carry hidden accuracy or cost penalties
helps practitioners write system prompts and helps evaluators choose evaluation conditions.

### Gap in Existing Work
Note: the resource-finder stage was skipped for this run (no `literature_review.md`,
`papers/`, `datasets/`, `code/` were provided), so this assessment is based on the author's
knowledge of the literature. Prior work studies pieces of this: persona/role prompting
("You are an expert…") has small, inconsistent effects on factual accuracy (e.g. Zheng et al.
2023, "Is 'A Helpful Assistant' the Best Role?"); format restrictions can hurt reasoning
("Let Me Speak Freely?", Tam et al. 2024); safety-tuned models over-refuse benign prompts
(XSTest, Röttger et al. 2023; OR-Bench); prompt-formatting sensitivity is large (Sclar et al.
2023). What is missing is a **controlled, side-by-side comparison of the four common
system-prompt categories (safety, persona, format, constraint) on the same tasks and models**,
with length-matched neutral controls and separate measurement of accuracy, output length and
refusal/hedging.

### Our Novel Contribution
A factorial experiment: 4 system-prompt categories × 3 wording variants each + 3 controls
(no system prompt, minimal "helpful assistant", length-matched irrelevant context) × 5
benchmark tasks × 3 open instruction-tuned models, measuring accuracy (strict and lenient
parsing), output length, and refusal/hedging rates, with paired item-level statistics.

### Experiment Justification
- **Exp 1 (main grid, greedy decoding)**: needed to estimate each category's effect on
  accuracy/length/refusal relative to controls; directly tests all four sub-hypotheses.
- **Exp 2 (length-matched control)**: separates "any long system prompt" effects from content
  effects (a safety prompt is also just a long prompt).
- **Exp 3 (strict vs lenient parsing)**: separates "format prompt makes answers parseable"
  from "format prompt makes the model reason better".
- **Exp 4 (sampling replicate, T=0.7)**: checks that greedy-decoding conclusions are not
  artefacts of a single decoding path.

## Research Question
Do system prompts of different categories (safety-heavy, detailed persona, format
specification, behavioural constraint) systematically change accuracy, output length and
refusal behaviour on downstream tasks unrelated to the prompt's purpose?

## Hypothesis Decomposition
- **H1 (safety)**: safety-heavy prompts reduce accuracy by 4–8 points vs. neutral control,
  with increased hedging/refusal, concentrated on ambiguous/sensitive items.
- **H2 (persona)**: detailed persona prompts increase output length by 30–50% with no accuracy gain.
- **H3 (format)**: format-specification prompts are the only category with a positive effect
  (+2–5 points on structured tasks).
- **H4 (constraint)**: constraint prompts have the largest across-task variance of effects,
  with some task effects ≤ −10 points and others positive.

IVs: system-prompt condition (15 levels, grouped into categories), task, model.
DVs: accuracy (lenient = best-effort answer extraction; strict = explicit final-answer
marker required), output tokens, refusal rate, hedge-phrase rate.

## Proposed Methodology

### Approach
Real open-weight instruct models run locally with vLLM (no API keys are available on this
host; one free RTX A6000). Models: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct,
Qwen3-4B-Instruct-2507 (three families/generations). The user turn is identical across
conditions and carries only a minimal task instruction, mimicking a deployed assistant
receiving a plain user question.

### Experimental Steps
1. Sample 300 items each (seed 42) from GSM8K (math reasoning), ARC-Challenge (science MC),
   MMLU "sensitive" subjects (medicine, law, security, morality – items where safety prompts
   might trigger hedging), TruthfulQA MC1 (misconception-laden / ambiguous), DROP (reading
   comprehension with numeric/span answers – structured extraction).
2. Write 15 system prompts (3 controls + 4 categories × 3 variants).
3. Run every model × condition × item with greedy decoding (max 1024 new tokens).
4. Score with strict & lenient parsers, detect refusals / hedges with regex lexicons.
5. Replicate the full grid for one model at T=0.7 (seed 1).

### Baselines
Primary: `neutral_short` ("You are a helpful assistant."). Also reported: `none` and
`neutral_long` (length-matched irrelevant deployment context).

### Evaluation Metrics
Accuracy (MC exact letter; GSM8K numeric match; DROP max-F1/EM against gold spans),
output token count, refusal rate, hedge rate, parse-failure rate.

### Statistical Analysis Plan
Paired item-level comparisons vs. baseline: McNemar exact test per model×task×condition;
category-level effects via paired cluster bootstrap over items (10k resamples), pooled across
variants. Holm correction across the family of category×task tests. α = 0.05. Length effects
as ratio of means with bootstrap CI. H4: compare the SD of per-(model,task,variant) effects
across categories, with bootstrap CI.

## Expected Outcomes
Support: safety effect in [−8,−4] with CI excluding 0; persona length ratio 1.3–1.5 with
accuracy CI including 0; format the only category with significantly positive effect;
constraint SD largest with min effect ≤ −10.

## Timeline and Milestones
Setup 20 min; implementation 45 min; runs ~1.5–2 h GPU; analysis 45 min; writing 30 min.

## Potential Challenges
- Shared GPU host: only one free GPU; run models sequentially.
- Parser bias: report both strict and lenient metrics; spot check outputs.
- Prompt-wording idiosyncrasy: 3 variants per category.
- Small open models may differ from frontier API models — stated as a limitation.

## Success Criteria
Complete grid run, all four sub-hypotheses evaluated with CIs, honest report.
