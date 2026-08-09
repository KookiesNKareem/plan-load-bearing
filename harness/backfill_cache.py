"""Fill missing entries in the abstraction cache using the currently served model."""
import json, sys
from plan_swap_template import abstract_intent, template_plans, get_tasks

model, path = sys.argv[1], sys.argv[2]
with open(path) as f:
    cache = json.load(f)
tasks = {t.id: t for t in get_tasks("base")}
missing = [tid for tid, v in cache.items() if not v and template_plans(tasks[tid])[0]]
print(f"backfilling {len(missing)} entries")
for tid in missing:
    cache[tid] = abstract_intent(model, tasks[tid])
    print(tid, "ok" if cache[tid] else "STILL_FAILED", flush=True)
with open(path, "w") as f:
    json.dump(cache, f, indent=1)
print("done:", sum(1 for v in cache.values() if v), "/", len(cache))
