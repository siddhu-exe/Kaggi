import json
import collections
import sys

def analyze_replay(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    steps = data.get("steps", [])
    
    # We track player 0 only for this analysis
    moves = 0
    actions = 0
    shed_ops = 0
    passes = 0
    total = 0
    
    action_counts = collections.Counter()
    
    for step in steps:
        if not step or len(step) < 1: continue
        
        p0 = step[0]
        if "action" in p0 and isinstance(p0["action"], dict):
            agent_act = p0["action"]
            
            # Extract all unit actions
            unit_ops = []
            if "farmer" in agent_act and agent_act["farmer"]:
                unit_ops.append(agent_act["farmer"])
            if "hands" in agent_act and isinstance(agent_act["hands"], list):
                for h in agent_act["hands"]:
                    if h: unit_ops.append(h)
                    
            for op in unit_ops:
                total += 1
                cmd = op[0]
                
                display_cmd = cmd
                if cmd == "PICKUP" and len(op) > 1:
                    display_cmd = f"PICKUP_{op[1]}"
                action_counts[display_cmd] += 1
                
                if cmd in ("NORTH", "SOUTH", "EAST", "WEST"):
                    moves += 1
                elif cmd in ("PICKUP", "PLACE", "DROP"):
                    shed_ops += 1
                elif cmd == "PASS":
                    passes += 1
                else:
                    actions += 1

    print(f"=== REPLAY WORKER ANALYSIS ===")
    print(f"Total Worker-Turns: {total}")
    if total == 0:
        return
    print(f"Moves:      {moves:5d} ({moves/total*100:.1f}%)")
    print(f"Actions:    {actions:5d} ({actions/total*100:.1f}%)")
    print(f"Shed Ops:   {shed_ops:5d} ({shed_ops/total*100:.1f}%)")
    print(f"Passes:     {passes:5d} ({passes/total*100:.1f}%)")
    print(f"\nAction Breakdown:")
    for k, v in action_counts.most_common():
        print(f"  {k}: {v}")
        
    print(f"\nProductive Capacity estimate: {(actions + shed_ops) / max(1, (total / 24)):.2f} actions/worker/day")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_replay(sys.argv[1])
    else:
        analyze_replay("replay_new.json")
