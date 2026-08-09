"""Outcome-level plan-swap test on tau2-bench airline: update-in-place vs cancel-and-rebook.

Flight-change tasks admit two write-paths to the same goal: update_reservation_flights (A) vs
cancel_reservation + book_reservation (B). Requests are rewritten to explicit mechanism
indifference; the model ratifies both template plans; each is force-executed (behavioral validity);
counterbalanced free rollouts measure whether the realized write-path tracks the seeded plan.
K unseeded rollouts per task measure spontaneous path preference.
"""
import argparse, json, re, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tau2.domains.airline.environment import get_environment, get_tasks
from tau2.data_model.message import ToolCall

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
GEN_KW = {}
MAX_GEN = 600

MECHANISM_VERBS = ("cancel", "rebook", "modify", "update", "change")
WRITES_A = frozenset(["update_reservation_flights"])
WRITES_B = frozenset(["cancel_reservation", "book_reservation"])

PLAN_A = ["get_user_details", "get_reservation_details", "search_direct_flight",
          "update_reservation_flights"]
PLAN_B = ["get_user_details", "get_reservation_details", "search_direct_flight",
          "cancel_reservation", "book_reservation"]

PLAN_ENDORSE_SCHEMA = {
    "type": "object",
    "properties": {"plan_A_valid": {"type": "boolean"}, "plan_B_valid": {"type": "boolean"}},
    "required": ["plan_A_valid", "plan_B_valid"],
}
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "resolved": {"type": "boolean"},
        "policy_violation": {"type": "boolean"},
        "realized_outcome": {"type": "string"},
    },
    "required": ["resolved", "policy_violation", "realized_outcome"],
}


def openai_tools(env):
    return [t.openai_schema for t in env.get_tools()]


def user_request(task, reason_override=None):
    ins = task.user_scenario.instructions
    parts = [reason_override or ins.reason_for_call]
    if getattr(ins, "known_info", None):
        parts.append("Information I can provide if asked: " + ins.known_info)
    return "\n".join(parts)


def extract_json(text):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def chat(model, messages, tools=None, max_tokens=None, temperature=0):
    return client.chat.completions.create(
        model=model, messages=messages, tools=tools,
        tool_choice="auto" if tools else None, temperature=temperature,
        max_tokens=max_tokens or MAX_GEN, **GEN_KW,
    ).choices[0].message


def jchat(model, messages, schema, max_tokens=None):
    msg = client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=max_tokens or MAX_GEN,
        response_format={"type": "json_schema", "json_schema": {"name": "out", "schema": schema}},
        **GEN_KW,
    ).choices[0].message
    return extract_json(msg.content)


def flight_change_tasks(tasks):
    out = []
    for t in tasks:
        ec = t.evaluation_criteria
        refs = {a.name for a in ec.actions} if ec and ec.actions else set()
        if "update_reservation_flights" in refs:
            out.append(t)
    return out


def abstract_intent(model, task):
    reason = task.user_scenario.instructions.reason_for_call
    prompt = (
        f"Customer request:\n{reason}\n\n"
        "Rewrite this request so the customer states what they need their flights to END UP AS, and is "
        "EXPLICITLY, EQUALLY open to two ways of getting there: (1) their existing reservation being "
        "adjusted, or (2) the old reservation being undone entirely and a fresh reservation made. Do "
        "not use the words cancel, rebook, modify, update, or change. Describe the desired flights "
        "EXACTLY as the original does — if the original is relative ('one day later', 'same flights "
        "but economy'), stay relative. Copy reservation IDs, flight numbers, dates and other details "
        "VERBATIM when the original gives them; NEVER invent a flight number, date, reservation ID, "
        "airport, or payment detail that the original does not contain. Close with 'whichever you "
        "recommend is fine — I truly have no preference.' First person, 2-5 sentences. Respond with "
        "ONLY the rewritten request text."
    )
    for i in range(4):
        txt = (chat(model, [{"role": "user", "content": prompt}],
                    temperature=0 if i == 0 else 0.7).content or "").strip()
        if txt and not any(v in txt.lower() for v in MECHANISM_VERBS):
            return txt
    return None


def endorse_plans(model, task, policy, reason):
    prompt = (
        f"Customer request:\n{user_request(task, reason)}\n\n"
        f"Plan A (adjust the existing reservation): {PLAN_A}\n"
        f"Plan B (undo the old reservation and make a fresh one): {PLAN_B}\n\n"
        "For each plan: would executing it end in a state that (a) the customer accepts and (b) "
        "complies with policy? The customer has EXPLICITLY said both end states fully satisfy them. "
        "Judge the END STATE each plan produces, not its step-by-step details: assume lookups and "
        "argument values are handled competently, and extra lookup calls are allowed. Judge each plan "
        "independently."
    )
    d = jchat(model, [{"role": "system", "content": policy}, {"role": "user", "content": prompt}],
              PLAN_ENDORSE_SCHEMA)
    return bool(d.get("plan_A_valid")), bool(d.get("plan_B_valid"))


def seed_messages(task, policy, stated_plan, forced=False, reason=None):
    sys = policy + ("\n\nResolve the request by calling tools. Assume the customer has already confirmed "
                    "any required actions; do not pause to ask for confirmation, just act.")
    if stated_plan is None:
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": user_request(task, reason)},
        ]
    if forced:
        writes = [t for t in stated_plan if t in (WRITES_A | WRITES_B)]
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": user_request(task, reason)
             + f"\n\nResolve this by executing exactly this plan: {stated_plan}. You may add extra "
               f"lookup (get_/search_/list_/calculate) calls if needed, but the state-changing "
               f"action(s) you take MUST be exactly these and no others: {writes}. Do not ask the "
               "customer anything; choose reasonable argument values from the information and tool "
               "results you have."},
        ]
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_request(task, reason)},
        {"role": "assistant", "content": f"My plan: {' -> '.join(stated_plan)}. I will execute it now."},
        {"role": "user", "content": "Go ahead and execute your plan step by step."},
    ]


def rollout(model, task, policy, tools, stated_plan, forced=False, reason=None, max_steps=16):
    env = get_environment()
    messages = seed_messages(task, policy, stated_plan, forced=forced, reason=reason)
    traj, tool_log, sigs = [], [], []
    nudges = 0
    for _ in range(max_steps):
        msg = chat(model, messages, tools=tools)
        if not msg.tool_calls:
            if nudges >= 2:
                break
            nudges += 1
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user", "content": "Please continue; I have already confirmed. "
                             "Do not ask further questions, just proceed with the next tool call."})
            continue
        tc = msg.tool_calls[0]
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            args = {}
        traj.append(tc.function.name)
        sigs.append(tc.function.name + (tc.function.arguments or ""))
        if len(sigs) >= 3 and sigs[-1] == sigs[-2] == sigs[-3]:
            break
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments or "{}"}}]})
        try:
            tm = env.get_response(ToolCall(id=tc.id, name=tc.function.name, arguments=args, requestor="assistant"))
            content = tm.content or ""
        except Exception as e:
            content = f"Error: {e}"
        tool_log.append({"tool": tc.function.name, "args": args, "response": content[:400]})
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": content[:1500]})
    return traj, tool_log


def wrote_ok(tool_log, write_tool):
    return any(e["tool"] == write_tool and not e["response"].startswith("Error") for e in tool_log)


def side_ok(tool_log, writes):
    return all(wrote_ok(tool_log, w) for w in writes)


def realized(tool_log, wa=WRITES_A, wb=WRITES_B):
    ok_a = side_ok(tool_log, wa) and not any(wrote_ok(tool_log, w) for w in wb)
    ok_b = side_ok(tool_log, wb) and not any(wrote_ok(tool_log, w) for w in wa)
    if ok_a:
        return "A"
    if ok_b:
        return "B"
    if side_ok(tool_log, wa) and side_ok(tool_log, wb):
        return "both"
    return "none"


def judge_rollout(model, task, policy, tool_log, reason):
    log_txt = json.dumps(tool_log, indent=0)[:6000]
    prompt = (
        f"POLICY:\n{policy[:5000]}\n\nCUSTOMER REQUEST:\n{user_request(task, reason)}\n\n"
        f"An agent executed this tool-call log:\n{log_txt}\n\n"
        "Context: the customer had ALREADY confirmed consent to any required action before this log "
        "begins, so absence of an explicit confirmation step in the log is NOT a violation.\n"
        "Judge strictly from the log: resolved = was the request fully resolved; policy_violation = did "
        "any action violate the policy; realized_outcome = the end state actually reached, in 5 words."
    )
    d = jchat(model, [{"role": "user", "content": prompt}], JUDGE_SCHEMA)
    return {"resolved": bool(d.get("resolved")), "violation": bool(d.get("policy_violation", True)),
            "outcome": d.get("realized_outcome", "")}


def process_task(model, task, policy, tools, k_unseeded):
    reason = abstract_intent(model, task)
    if not reason:
        return {"task_id": task.id, "status": "abstract_failed"}

    endorse_a, endorse_b = endorse_plans(model, task, policy, reason)
    trajFA, logFA = rollout(model, task, policy, tools, PLAN_A, forced=True, reason=reason)
    trajFB, logFB = rollout(model, task, policy, tools, PLAN_B, forced=True, reason=reason)
    forced_ok_a, forced_ok_b = side_ok(logFA, WRITES_A), side_ok(logFB, WRITES_B)
    judgeFA = judge_rollout(model, task, policy, logFA, reason)
    judgeFB = judge_rollout(model, task, policy, logFB, reason)
    if not (forced_ok_a and forced_ok_b):
        return {"task_id": task.id, "status": "not_forceable", "reason": reason,
                "endorse_A": endorse_a, "endorse_B": endorse_b,
                "forced_ok_A": forced_ok_a, "forced_ok_B": forced_ok_b,
                "traj_forced_A": trajFA, "traj_forced_B": trajFB,
                "judge_forced_A": judgeFA, "judge_forced_B": judgeFB}

    frees = {}
    for key, plan in (("A1", PLAN_A), ("A2", PLAN_A), ("B1", PLAN_B), ("B2", PLAN_B)):
        traj, log = rollout(model, task, policy, tools, plan, reason=reason)
        frees[key] = {"traj": traj, "realized": realized(log),
                      "judge": judge_rollout(model, task, policy, log, reason),
                      "log": [{"tool": e["tool"], "args": e["args"],
                               "response": e["response"][:200]} for e in log]}
    unseeded = []
    for _ in range(k_unseeded):
        traj, log = rollout(model, task, policy, tools, None, reason=reason)
        unseeded.append({"traj": traj, "realized": realized(log)})

    scored = {k: v["realized"] for k, v in frees.items() if v["realized"] in ("A", "B")}
    return {
        "task_id": task.id, "status": "ok", "reason": reason,
        "endorse_A": endorse_a, "endorse_B": endorse_b,
        "traj_forced_A": trajFA, "traj_forced_B": trajFB,
        "judge_forced_A": judgeFA, "judge_forced_B": judgeFB,
        "frees": frees, "unseeded": [u["realized"] for u in unseeded],
        "unseeded_trajs": [u["traj"] for u in unseeded],
        "bound": {k: v == k[0] for k, v in scored.items()},
        "realized_all": {k: v["realized"] for k, v in frees.items()},
        "noise_agree": (frees["A1"]["realized"] == frees["A2"]["realized"]
                        and frees["B1"]["realized"] == frees["B2"]["realized"]),
        "seed_invariant": (len({v["realized"] for v in frees.values()}) == 1
                           and frees["A1"]["realized"] in ("A", "B")),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--k-unseeded", type=int, default=4)
    p.add_argument("--max-gen", type=int, default=600)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--out", default="results/plan_swap_airline.json")
    args = p.parse_args()

    global MAX_GEN
    MAX_GEN = args.max_gen
    if args.reasoning_effort:
        GEN_KW["extra_body"] = {"reasoning_effort": args.reasoning_effort}

    tasks = flight_change_tasks(get_tasks())
    print(f"{len(tasks)} flight-change tasks", flush=True)
    env0 = get_environment()
    policy = env0.get_policy()
    tools = openai_tools(env0)

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_task, args.model, t, policy, tools, args.k_unseeded): t.id
                for t in tasks}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = {"task_id": futs[fut], "status": f"error: {e}"}
            rows.append(r)
            if r["status"] == "ok":
                print(f"task {r['task_id']}: realized={r['realized_all']} bound={r['bound']} "
                      f"unseeded={r['unseeded']}", flush=True)
            else:
                print(f"task {r['task_id']}: {r['status']}", flush=True)

    ok = [r for r in rows if r["status"] == "ok"]
    all_bound = [v for r in ok for v in r["bound"].values()]

    def rate(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    summary = {
        "model": args.model, "n_tried": len(rows),
        "n_abstract_failed": sum(1 for r in rows if r["status"] == "abstract_failed"),
        "n_not_forceable": sum(1 for r in rows if r["status"] == "not_forceable"),
        "n_error": sum(1 for r in rows if str(r["status"]).startswith("error")),
        "n_clean": len(ok), "n_scored_rollouts": len(all_bound),
        "bind_rate": rate(all_bound),
        "bind_rate_A_seeded": rate([v for r in ok for k, v in r["bound"].items() if k.startswith("A")]),
        "bind_rate_B_seeded": rate([v for r in ok for k, v in r["bound"].items() if k.startswith("B")]),
        "noise_agree_rate": rate([r["noise_agree"] for r in ok]),
        "frac_tasks_seed_invariant": rate([r["seed_invariant"] for r in ok]),
        "unseeded_frac_A": rate([1 if x == "A" else 0 for r in ok for x in r["unseeded"]
                                 if x in ("A", "B")]),
        "endorse_both_rate": rate([r["endorse_A"] and r["endorse_B"] for r in ok]),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print("SUMMARY:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
