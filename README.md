# Is a Plan Load-Bearing? A Black-Box Causal Audit of Plan-Based Oversight for LLM Agents

Code and data release for the paper published at the 2nd Workshop on Lifelong Agents (LLA) at COLM 2026 ([paper](paper.pdf), [OpenReview](https://openreview.net/forum?id=2ptkLSaok8)).

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

## Results

The release contains the complete 62-task regeneration used in the final paper for all four models. Each task has two behaviorally validated, policy-compliant outcomes. Four counterbalanced free rollouts seed plan A or plan B as the agent's own stated plan; separate unseeded rollouts measure revealed preference. An outcome is scored only when its write succeeds in the environment. gpt-oss-120b pools two identical passes, so its rollout count is doubled.

| Model | Scored seeded rollouts | P(A \| plan A) | P(A \| plan B) | Binding | Seed-invariant tasks | Override: preference agrees | Override: preference conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 229/248 | 0.972 | 0.107 | 0.930 | 5/62 | 0.000 | 0.122 |
| gpt-oss-120b | 241/496 | 0.467 | 0.022 | 0.751 | 6/62 | 0.023 | 0.488 |
| GLM-4.5-Air | 131/248 | 0.788 | 0.253 | 0.763 | 6/62 | 0.081 | 0.415 |
| GLM-4.7-Flash | 212/248 | 0.818 | 0.177 | 0.821 | 7/62 | 0.032 | 0.341 |

The seeded plan shifts the realized outcome in every model. Overrides are more frequent when the seeded plan conflicts with the model's separately measured revealed preference; task-clustered confidence intervals for the conflict-minus-agree contrast exclude zero in all four models. All 19 seed-invariant tasks with a measurable unseeded preference realize the preferred outcome.

Filtering to rollouts that a single fixed judge marks resolved and policy-compliant retains binding of 89/95 (0.937), 92/128 (0.719), 57/76 (0.750), and 96/120 (0.800), respectively. The airline-domain transfer has five gated tasks and 17/18 binding scored rollouts; it is evidence that the protocol transfers, not a powered independent test of the preference mechanism.

## Data

### Shared retail inputs and Qwen3-30B

- `data/abstract_cache_v2.json` — regenerated indifference rewrites for the 114 retail tasks considered by the pipeline.
- `data/canonical_ids.txt` — the 62 task IDs that pass the Qwen validity gates and define the common evaluation set.
- `data/qwen_v2_full.json` — Qwen ratification, forced-execution gates, forced trajectories, and seeded free rollouts.
- `data/unseeded_qwen_v2.json` — Qwen temperature-0 unseeded rollouts.
- `data/unseeded_qwen_t07_v2.json` — Qwen temperature-0.7 preference replication.

### gpt-oss-120b

- `data/gptoss_v2_full_p1.json`, `data/gptoss_v2_full_p2.json` — the two seeded passes pooled in the paper.
- `data/unseeded_gptoss_v2_p1.json`, `data/unseeded_gptoss_v2_p2.json` — corresponding unseeded passes.

### Exact-protocol GLM runs and fixed judging

- `data/exact/glm45air_exact_v2_full.json`, `data/exact/unseeded_glm45air_exact_v2.json` — GLM-4.5-Air seeded and unseeded runs.
- `data/exact/glm47_exact_v2_full.json`, `data/exact/unseeded_glm47_exact_v2.json` — GLM-4.7-Flash seeded and unseeded runs.
- `data/exact/unseeded_glm47_t07_exact_v2.json` — GLM-4.7-Flash temperature-0.7 preference replication.
- `data/exact/rejudge_fixed.json` — fixed-judge results for all 1,236 regenerated seeded transcripts.
- `data/exact/*.sha256` — frozen checksums for the exact GLM and fixed-judge artifacts.

### Airline transfer

- `data/abstract_cache_airline.json` — mechanism-indifferent airline rewrites.
- `data/airline_qwen_v2b.json` — airline ratification, force gates, seeded rollouts, and unseeded rollouts.

## Harness

- `harness/plan_swap_outcome.py` — retail protocol: outcome construction, rewrite, ratification, forced-execution gates, and plan-seeded rollouts.
- `harness/plan_swap_airline.py` — airline write-set variant.
- `harness/unseeded_pref.py` — unseeded revealed-preference rollouts.
- `harness/pref_analysis.py` — preference-versus-override analysis.
- `harness/rejudge.py` — fixed-judge re-judging.
- `harness/backfill_cache.py` — rewrite-cache utility.

The harness expects [tau2-bench](https://github.com/sierra-research/tau2-bench) and a local OpenAI-compatible model endpoint; the paper used vLLM 0.24. Primary runs use temperature 0 and eight concurrent task workers. Maximum generation lengths are 600 tokens for Qwen, 2,000 for gpt-oss, and 2,500 for both GLM models.

To launch a new retail run against a locally served model:

```bash
python harness/plan_swap_outcome.py --model <served-model-id> --out results/run.json
python harness/unseeded_pref.py --model <served-model-id> \
  --task-ids data/canonical_ids.txt \
  --abstract-cache data/abstract_cache_v2.json \
  --out results/unseeded.json
```

To verify the frozen exact artifacts:

```bash
cd data/exact
sha256sum -c glm45_exact.sha256
sha256sum -c glm47_exact.sha256
sha256sum -c rejudge_exact.sha256
```

Serving nondeterminism remains measurable even at temperature 0, so fresh generations should be compared statistically rather than expected to match rollout-for-rollout.
