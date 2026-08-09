# Is a Plan Load-Bearing? A Black-Box Causal Audit of Plan-Based Oversight for LLM Agents

Code and data release for the paper published at the 2nd Workshop on Lifelong Agents (LLA) at COLM 2026 (`paper.pdf`, [OpenReview](https://openreview.net/forum?id=2ptkLSaok8)).

## Citation

```bibtex
@inproceedings{
fareed2026is,
title={Is a Plan Load-Bearing? A Black-Box Causal Audit of Plan-Based Oversight for {LLM} Agents},
author={Kareem Fareed},
booktitle={COLM 2026 The 2nd Workshop on Lifelong Agents: Learning, Aligning, and Evolving},
year={2026},
url={https://openreview.net/forum?id=2ptkLSaok8}
}
```

## Data availability

The raw rollout logs of the runs reported in the paper were lost in a storage failure after analysis. This release contains the full experimental harness together with a complete independent re-run of the primary experiment (Qwen3-30B-A3B on τ²-bench retail) under the identical protocol. The re-run reproduces the paper's qualitative findings — binding 0.93 on its 62 gated tasks vs. the reported 0.88 on 59 — with the gated task sets differing because the request rewrites were regenerated. The other three models' runs have not been regenerated; their statistics rest on the recovered analysis records described in the paper's Limitations section.

## Contents

### `harness/`

- `plan_swap_outcome.py` — the main protocol: outcome-pair construction, indifference rewrites, template-plan ratification, forced-execution validity gates, counterbalanced plan-seeded free rollouts, same-seed noise floor, LLM judging.
- `plan_swap_airline.py` — the airline-domain variant (write-set outcomes).
- `plan_swap_tau2.py` — the earlier pilot version of the protocol.
- `unseeded_pref.py` — unseeded rollouts for revealed-preference measurement.
- `pref_analysis.py` — preference-vs-override (dose-response) analysis.
- `rejudge.py` — fixed-judge re-judging of free rollouts.
- `backfill_cache.py` — rewrite-cache utility.

Requirements: a local OpenAI-compatible endpoint (the paper used vLLM 0.24 at `http://localhost:8000/v1`) and [tau2-bench](https://github.com/sierra-research/tau2-bench).

### `data/`

- `abstract_cache_v2.json` — the regenerated indifference rewrites, one per retail task (114).
- `qwen_v2_full.json` — the Qwen3-30B retail re-run: per-task ratification and forced-execution gate results, forced trajectories, and the four counterbalanced plan-seeded free rollouts with realized outcomes (`summary` + `rows`).
- `unseeded_qwen_v2.json` — the unseeded (no-plan) rollouts used for revealed-preference measurement.
- `canonical_ids.txt` — the 62 task IDs passing all validity gates in the re-run.

## Reproducing

To re-run the analysis on the released data (no GPU needed):

```
python harness/pref_analysis.py
```

To regenerate the experiment against a locally served model:

```
python harness/plan_swap_outcome.py --model <served-model-id> --out results/run.json
python harness/unseeded_pref.py --model <served-model-id> --task-ids data/canonical_ids.txt \
    --abstract-cache data/abstract_cache_v2.json --out results/unseeded.json
```

Serving nondeterminism applies: the paper reports per-model same-seed noise floors of 6–19%, and re-runs should be compared against those, not expected to match exactly.
