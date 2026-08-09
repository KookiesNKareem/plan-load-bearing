"""Spontaneous outcome preference: unseeded rollouts (no stated plan) on the canonical tasks."""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import plan_swap_template as pst


def process_task(model, task, policy, tools, reason, k):
    plan_a, wa, plan_b, wb = pst.template_plans(task)
    outs = []
    for _ in range(k):
        try:
            traj, log = pst.rollout(model, task, policy, tools, None, reason=reason)
            outs.append({"traj": traj, "realized": pst.realized(log, wa, wb)})
        except Exception:
            outs.append({"traj": [], "realized": "error"})
    return {"task_id": task.id, "pair": [wa, wb],
            "realized": [o["realized"] for o in outs],
            "trajs": [o["traj"] for o in outs]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task-ids", required=True)
    p.add_argument("--abstract-cache", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-gen", type=int, default=600)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    pst.MAX_GEN = args.max_gen
    pst.DEFAULT_TEMP = args.temperature
    if args.reasoning_effort:
        pst.GEN_KW["extra_body"] = {"reasoning_effort": args.reasoning_effort}
    with open(args.task_ids) as f:
        keep = {line.strip() for line in f if line.strip()}
    with open(args.abstract_cache) as f:
        cache = json.load(f)
    tasks = [t for t in pst.get_tasks("base") if t.id in keep]
    env0 = pst.get_environment()
    policy = env0.get_policy()
    tools = pst.openai_tools(env0)

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_task, args.model, t, policy, tools, cache[t.id], args.k): t.id
                for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            print(f"task {r['task_id']}: {r['realized']}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "k": args.k, "rows": rows}, f, indent=1)
    scored = [x for r in rows for x in r["realized"] if x in ("A", "B")]
    frac_a = sum(1 for x in scored if x == "A") / len(scored) if scored else None
    print("SUMMARY:", json.dumps({"model": args.model, "n_tasks": len(rows),
                                  "scored": len(scored), "frac_A": round(frac_a, 3) if frac_a is not None else None}), flush=True)


if __name__ == "__main__":
    main()
