import json

def analyze_replay(filepath):
    try:
        with open(filepath, 'r') as f:
            replay = json.load(f)
    except FileNotFoundError:
        return {"error": f"File not found: {filepath}"}

    steps = replay.get("steps", [])
    if not steps:
        return {}

    stats = {
        "p1_cash": 0,
        "p2_cash": 0,
        "win": False,
        "days_zero_cash": 0,
        "total_workers": 0,
        "total_moves": 0,
        "total_productive": 0,
        "completed_tasks": 0,
        "dead_weeds": 0,
        "missed_watering": 0,
        "missed_feeding": 0,
        "occupied_tiles": 0,
        "productive_tiles": 0,
        "shed_overflow_days": 0,
        "days": 0,
    }
    
    for i, step in enumerate(steps):
        if not step: continue
        
        p1_state = step[0]
        p2_state = step[1] if len(step) > 1 else {}
        
        obs = p1_state.get("observation", {})
        farms = obs.get("farms", [])
        if not farms: continue
            
        farm = farms[0]
        
        if obs.get("hour") == 23:
            stats["days"] += 1
            if farm.get("money", 0) <= 0:
                stats["days_zero_cash"] += 1
                
            stats["total_workers"] += 1 + len(farm.get("hands", []))
            
            tiles = farm.get("tiles", [])
            occ = 0
            prod = 0
            weeds = 0
            for r in tiles:
                for t in r:
                    if isinstance(t, dict):
                        occ += 1
                        if t.get("kind") in ("PLANT", "COOP", "PASTURE"):
                            prod += 1
                            if t.get("kind") == "PLANT" and t.get("consecutive_unwatered", 0) > 0:
                                stats["missed_watering"] += 1
                            if t.get("kind") in ("COOP", "PASTURE") and t.get("consecutive_unfed", 0) > 0:
                                stats["missed_feeding"] += 1
                        if t.get("kind") == "WEED":
                            weeds += 1
            stats["occupied_tiles"] += occ
            stats["productive_tiles"] += prod
            stats["dead_weeds"] += weeds
            
            private = obs.get("private", {})
            shed = private.get("shed", {})
            shed_total = sum(shed.values())
            if shed_total >= 100:
                stats["shed_overflow_days"] += 1

        # Track actions
        if i > 0:
            actions = p1_state.get("action", {})
            if actions:
                farmer_act = actions.get("farmer", [])
                hand_acts = actions.get("hands", [])
                
                for act in [farmer_act] + hand_acts:
                    if act and act[0] in ("NORTH", "SOUTH", "EAST", "WEST"):
                        stats["total_moves"] += 1
                    elif act and act[0] != "PASS":
                        stats["total_productive"] += 1
                        stats["completed_tasks"] += 1
                        
    final_step = steps[-1]
    stats["p1_cash"] = final_step[0].get("reward", 0)
    stats["p2_cash"] = final_step[1].get("reward", 0) if len(final_step) > 1 else 0
    stats["win"] = stats["p1_cash"] > stats["p2_cash"]
    
    # Calculate derived stats
    days = max(1, stats["days"])
    workers = stats["total_workers"] / days if days else 1
    
    return {
        "final_cash": stats["p1_cash"],
        "opponent_cash": stats["p2_cash"],
        "win/loss": "WIN" if stats["win"] else "LOSS",
        "workers_per_day": round(workers, 1),
        "productive_turns_per_day": round(stats["total_productive"] / days, 1),
        "movement_turns_per_day": round(stats["total_moves"] / days, 1),
        "completed_tasks_per_day": round(stats["completed_tasks"] / days, 1),
        "empirical_capacity_per_day": round(24.0 * (stats["total_productive"] / max(1, stats["total_productive"] + stats["total_moves"])), 2),
        "avg_distance_per_task": round(stats["total_moves"] / max(1, stats["total_productive"]), 2),
        "total_worker_travel": stats["total_moves"],
        "occupied_tiles_per_day": round(stats["occupied_tiles"] / days, 1),
        "productive_tiles_per_day": round(stats["productive_tiles"] / days, 1),
        "missed_watering_total": stats["missed_watering"],
        "missed_feeding_total": stats["missed_feeding"],
        "dead_weed_crops": stats["dead_weeds"],
        "days_zero_cash": stats["days_zero_cash"],
        "shed_overflow_days": stats["shed_overflow_days"]
    }

def main():
    files = {
        "V8 vs V8": "baseline_vs_baseline.json",
        "Candidate vs V8": "candidate_vs_baseline.json",
        "Candidate vs Starter": "candidate_vs_starter.json",
        "Candidate vs Random": "candidate_vs_random.json"
    }
    
    results = {}
    for name, path in files.items():
        results[name] = analyze_replay(path)
            
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
