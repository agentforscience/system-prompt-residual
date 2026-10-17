# Outline: System prompts leave residue on unrelated tasks

1. Intro: hook (system prompts sit in front of every request) -> gap (no side-by-side category comparison w/ length-matched controls) -> design (Fig 1) -> preview (safety +0.2, persona x1.99 & -3.1, format -2.5 lenient / +19.6 strict, constraint -9.0, min -61) -> contributions.
2. Related work: persona/role prompts; format restrictions; over-refusal; prompt sensitivity & multi-prompt eval; CoT.
3. Method: problem formulation (paired delta vs neutral_short); conditions table; tasks; models & inference; scoring (lenient/strict/behaviour); statistics.
4. Results: main table + fig (H1-H4 evidence); strict scoring table; per-model; length; hedging/refusal; flip taxonomy; dispersion (H4); robustness (answered-only, length control, variants, T=0.7 replicate); hypothesis verdict table.
5. Discussion: why safety is cheap; parseability vs correctness; constraints dangerous; model dependence; limitations; broader implications.
6. Conclusion + future work.
Appendix: full prompt texts, baseline accuracies, heatmap, strict & T=0.7 category figures.
