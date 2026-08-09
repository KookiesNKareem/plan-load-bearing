"""Re-judge stored free-rollout logs from prior runs with a single fixed judge model."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from plan_swap_template import judge_rollout, get_tasks, get_environment, GEN_KW

judge_model = sys.argv[1]
files = sys.argv[2:]
GEN_KW["extra_body"] = {"reasoning_effort": "low"}
tasks = {t.id: t for t in get_tasks("base")}
policy = get_environment().get_policy()

jobs = []
for path in files:
    d = json.load(open(path))
    for r in d["rows"]:
        if r["status"] == "ok":
            for key, v in r["frees"].items():
                jobs.append((path, r["task_id"], key, r["reason"], v["log"]))

def run(job):
    path, tid, key, reason, log = job
    j = judge_rollout(judge_model, tasks[tid], policy, log, reason)
    return {"file": path, "task_id": tid, "key": key, **j}

with ThreadPoolExecutor(max_workers=8) as ex:
    out = list(ex.map(run, jobs))
with open("results/rejudge_fixed.json", "w") as f:
    json.dump(out, f, indent=1)
res = sum(1 for o in out if o["resolved"])
print(f"rejudged {len(out)} rollouts, resolved={res}")
