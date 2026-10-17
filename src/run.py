"""Generate responses for every (system prompt condition x item) with a local vLLM model.

Usage: python src/run.py --model qwen25_7b [--temperature 0.0 --seed 0 --tag greedy]
Writes results/raw/<model>__<tag>.jsonl (one line per condition x item).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from prompts import CATEGORY_OF, SYSTEM_PROMPTS  # noqa: E402

MODELS = {
    "qwen25_7b": "/data/transformers/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    "llama31_8b": "/data/transformers/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1",
    "qwen3_4b": "/data/transformers/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="greedy")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="items per task (0 = all), for smoke tests")
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--quant", default="", help="'' for bf16 weights, 'fp8' for on-the-fly FP8 weight quantisation")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs")
    ap.add_argument("--eager", action="store_true", help="disable CUDA graphs (saves memory)")
    ap.add_argument("--conditions", default="", help="comma-separated subset of conditions")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    items = [json.loads(l) for l in open("results/items.jsonl")]
    if args.limit:
        seen = {}
        items = [it for it in items if seen.setdefault(it["task"], []).append(1) or len(seen[it["task"]]) <= args.limit]
    conds = args.conditions.split(",") if args.conditions else list(SYSTEM_PROMPTS)

    llm = LLM(model=MODELS[args.model], dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=args.gpu_util, seed=args.seed, enable_prefix_caching=True,
              quantization=args.quant or None, tensor_parallel_size=args.tp, enforce_eager=args.eager)
    sp = SamplingParams(temperature=args.temperature, top_p=1.0 if args.temperature == 0 else 0.95,
                        max_tokens=args.max_tokens, seed=args.seed)

    # Checkpointing: conditions are generated in chunks and appended to a .partial file, so a killed
    # run resumes from the last completed chunk. The final .jsonl is written only when all are done.
    os.makedirs("results/raw", exist_ok=True)
    path = f"results/raw/{args.model}__{args.tag}.jsonl"
    part = path + ".partial"
    done = set()
    if os.path.exists(part):
        done = {json.loads(l)["condition"] for l in open(part)}
        # drop any condition that is only partially written (cannot happen with whole-chunk writes, but be safe)
        cnt = {}
        for l in open(part):
            c = json.loads(l)["condition"]
            cnt[c] = cnt.get(c, 0) + 1
        done = {c for c in done if cnt[c] == len(items)}
        keep = [l for l in open(part) if json.loads(l)["condition"] in done]
        open(part, "w").writelines(keep)
    todo = [c for c in conds if c not in done]
    print(f"{len(conds)} conditions x {len(items)} items; already done: {sorted(done)}", flush=True)
    t0 = time.time()
    CHUNK = 3
    for k in range(0, len(todo), CHUNK):
        convs, keys = [], []
        for c in todo[k:k + CHUNK]:
            sysp = SYSTEM_PROMPTS[c]
            for it in items:
                msgs = ([{"role": "system", "content": sysp}] if sysp is not None else []) + [{"role": "user", "content": it["user"]}]
                convs.append(msgs)
                keys.append((c, it))
        outs = llm.chat(convs, sp, use_tqdm=True)
        with open(part, "a") as f:
            for (c, it), o in zip(keys, outs):
                g = o.outputs[0]
                f.write(json.dumps(dict(model=args.model, tag=args.tag, condition=c, category=CATEGORY_OF[c],
                                        task=it["task"], id=it["id"], response=g.text,
                                        n_tokens=len(g.token_ids), n_prompt_tokens=len(o.prompt_token_ids),
                                        finish_reason=g.finish_reason)) + "\n")
        print(f"chunk done: {todo[k:k + CHUNK]} ({time.time() - t0:.0f}s)", flush=True)
    dt = time.time() - t0
    rows = [json.loads(l) for l in open(part)]
    assert len(rows) == len(conds) * len(items), (len(rows), len(conds) * len(items))
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.remove(part)
    meta = dict(model=args.model, path=MODELS[args.model], tag=args.tag, quant=args.quant or "bf16", tp=args.tp, eager=args.eager,
                gpu_util=args.gpu_util, temperature=args.temperature,
                seed=args.seed, max_tokens=args.max_tokens, n=len(rows), seconds_this_session=dt,
                out_tokens=sum(r["n_tokens"] for r in rows))
    with open(f"results/raw/{args.model}__{args.tag}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(meta, flush=True)


if __name__ == "__main__":
    main()
