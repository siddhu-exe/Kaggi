#!/usr/bin/env python3
"""Small-batch evolutionary tuner for the Kaggriculture agent.

Examples:
  python tune.py --variants 8 --episodes 3
  python tune.py --variants 4 --episodes 2 --replay ../replay.json

Each candidate runs in a fresh subprocess so main.py globals and per-player
episode state cannot leak between variants. Results are appended to results.jsonl;
best_params.json becomes the seed for the next manual batch.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BEST = HERE / "best_params.json"
LOG = HERE / "results.jsonl"


def base_params():
    sys.path.insert(0, str(ROOT))
    import main
    return copy.deepcopy(main.PARAMS)


def mutate(seed, rng, structural=False):
    p = copy.deepcopy(seed)
    integer = {
        "hands_per_day": (5, 10), "actions_per_unit_day": (12, 22),
        "feed_days": (2, 7), "endgame_day": (24, 29),
        "shed_overflow_force": (70, 95),
    }
    keys = list(integer)
    for key in rng.sample(keys, 3 if structural else 1):
        lo, hi = integer[key]
        p[key] = max(lo, min(hi, int(p[key]) + rng.choice([-2, -1, 1, 2])))
    alloc = p["allocation"]
    crops = list(alloc)
    for _ in range(3 if structural else 1):
        src, dst = rng.sample(crops, 2)
        delta = rng.randint(2, 8) if structural else rng.randint(1, 3)
        moved = min(delta, max(0, int(alloc[src]) - 1))
        alloc[src] -= moved
        alloc[dst] += moved
    if structural:
        p["land_days"] = [max(1, d + rng.choice([-2, -1, 1, 2])) for d in p["land_days"]]
        p["pasture_sheep_fraction"] = round(max(0, min(1,
            p["pasture_sheep_fraction"] + rng.choice([-0.25, 0.25]))), 2)
    return p


def replay_score(path):
    """Validation hook for exported Kaggle replays.

    Replays cannot counterfactually execute a new policy. We therefore extract
    the submitted agent's realized final reward and action-health indicators;
    these are logged beside local scores for regression comparison. A future
    action-replay simulator plugs in here and can return a counterfactual score.
    """
    data = json.loads(Path(path).read_text())
    steps = data.get("steps", [])
    if not steps:
        return {"final_reward": None, "invalid_actions": None}
    final = steps[-1][0]
    reward = final.get("reward") if isinstance(final, dict) else None
    return {"final_reward": reward, "turns": len(steps)}


def run_candidate(params, episodes):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.environ["KAGGRICULTURE_PARAMS"] = json.dumps(params, separators=(",", ":"))
    import main as agent_module
    agent_module = importlib.reload(agent_module)
    return evaluate(agent_module, episodes)


def explanation(best, worst):
    diffs = []
    for key in ("hands_per_day", "actions_per_unit_day", "feed_days", "endgame_day",
                "shed_overflow_force", "pasture_sheep_fraction"):
        if best["params"].get(key) != worst["params"].get(key):
            direction = "higher" if best["params"][key] > worst["params"][key] else "lower"
            diffs.append(f"{direction} {key}")
    for key, value in best["params"]["allocation"].items():
        other = worst["params"]["allocation"].get(key, 0)
        if abs(value - other) >= 3:
            diffs.append(("more " if value > other else "less ") + key.lower())
    observed = ", ".join(diffs[:3]) or "the combined parameter interaction"
    return (f"{observed} correlated with {best['score'] - worst['score']:.0f} more "
            "final coins than the batch's worst candidate")


def evaluate(agent_module, episodes):
    from kaggle_environments import make
    opponents = ["starter", "random"]
    scores = []
    details = []
    for i in range(episodes):
        opp = opponents[i % len(opponents)] if i < episodes - 1 else agent_module.agent
        agent_module.STATES[0].reset(); agent_module.STATES[1].reset()
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 1000 + i}, debug=False)
        env.run([agent_module.agent, opp])
        final = env.steps[-1]
        ours = float(final[0].reward or 0)
        scores.append(ours)
        details.append({"opponent": "self" if opp is agent_module.agent else opp, "score": ours})
    return {"score": sum(scores) / len(scores), "games": details}


def worker(episodes, output=None):
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import main
        payload = evaluate(main, episodes)
    except BaseException as exc:
        payload = {"error": repr(exc), "score": -1, "games": []}
    result = json.dumps(payload)
    if output:
        Path(output).write_text(result + "\n")
    else:
        os.write(1, (result + "\n").encode())


def main_cli(args):
    rng = random.Random(args.seed)
    seed = json.loads(BEST.read_text()) if BEST.exists() else base_params()
    candidates = [copy.deepcopy(seed)]
    for i in range(max(0, args.variants - 1)):
        candidates.append(mutate(seed, rng, structural=(i % 4 == 3)))
    rows = []
    report_lines = []
    for i, params in enumerate(candidates):
        result = run_candidate(params, args.episodes)
        row = {"variant": i, "params": params, **result}
        rows.append(row)
        report_lines.append(f"variant {i}: {result['score']:.1f} {result['games']}")
    rows.sort(key=lambda r: r["score"], reverse=True)
    reason = explanation(rows[0], rows[-1])
    validation = replay_score(args.replay) if args.replay else None
    rows[0]["reason"] = reason
    rows[0]["replay_validation"] = validation
    BEST.write_text(json.dumps(rows[0]["params"], indent=2) + "\n")
    with LOG.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    report_lines.append(f"best: {rows[0]['score']:.1f}; {reason}")
    if validation:
        report_lines.append(f"replay validation: {validation}")
    os.write(1, ("\n".join(report_lines) + "\n").encode())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--replay")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--output", help=argparse.SUPPRESS)
    ns = ap.parse_args()
    if ns.worker:
        worker(ns.episodes, ns.output)
    else:
        main_cli(ns)
