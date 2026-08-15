"""Measurement harness: run our agent vs starter, one config, deterministic seed.
Prints per-day diagnostics (money, hands, per-quadrant occupancy, animal-quadrant
health) as JSON so baseline vs dedicated variants can be compared apples-to-apples.

Usage:
    KAGGRICULTURE_PARAMS='{"animal_quadrant": false}' python3 measure.py <seed> <tag>
"""
import json, sys, os
from kaggle_environments import make
import main as A  # imports with whatever KAGGRICULTURE_PARAMS is set

QUAD = {
    "NW": [(x, y) for y in range(0, 5) for x in range(0, 5)],
    "NE": [(x, y) for y in range(0, 5) for x in range(5, 10)],
    "SW": [(x, y) for y in range(5, 10) for x in range(0, 5)],
    "SE": [(x, y) for y in range(5, 10) for x in range(5, 10)],
}


def quad_stats(tiles, coords):
    unlocked = occ = plants = structs = animals = healthy = 0
    for (x, y) in coords:
        t = tiles[y][x] if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]) else "LOCKED"
        if t == "LOCKED":
            continue
        unlocked += 1
        if isinstance(t, dict):
            occ += 1
            k = t.get("kind")
            if k == "PLANT":
                plants += 1
            elif k in ("COOP", "PASTURE"):
                structs += 1
                if t.get("animal") is not None:
                    animals += 1
                    if int(t.get("consecutive_unfed", 0) or 0) == 0:
                        healthy += 1
    return dict(unlocked=unlocked, occ=occ, plants=plants, structs=structs,
                animals=animals, healthy=healthy)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    tag = sys.argv[2] if len(sys.argv) > 2 else "run"
    A.STATES[0].reset(); A.STATES[1].reset()
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.run([A.agent, "starter"])

    days = []
    for step_no, step in enumerate(env.steps):
        if not isinstance(step, list) or step_no % 24 != 1:
            continue
        src = step[0] if step and isinstance(step[0], dict) else {}
        obs = src.get("observation", {}) if isinstance(src, dict) else {}
        farms = obs.get("farms", []) or []
        if not farms:
            continue
        farm = farms[0]
        tiles = farm.get("tiles", []) or []
        unlocked_q = farm.get("unlocked_quadrants", [])
        rec = {
            "day": step_no // 24,
            "money": farm.get("money"),
            "hands": len(farm.get("hands", []) or []),
            "hires_today": farm.get("hires_today"),
            "quads": unlocked_q,
        }
        for q in ("NW", "NE", "SW", "SE"):
            rec[q] = quad_stats(tiles, QUAD[q])
        # animal quadrant = 3rd unlocked, per agent logic
        aqn = unlocked_q[2] if len(unlocked_q) >= 3 else None
        rec["animal_quad_name"] = aqn
        days.append(rec)

    final = env.steps[-1]
    res = {}
    for i, s in enumerate(final):
        if isinstance(s, dict):
            res[f"player{i}"] = {"reward": s.get("reward"), "status": s.get("status")}
    out = {"tag": tag, "seed": seed, "final": res, "days": days,
           "animal_quadrant": A.PARAMS.get("animal_quadrant")}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
