import json
import sys

def run():
    with open("replay_new.json") as f:
        replay = json.load(f)
        
    for step_no, step in enumerate(replay["steps"]):
        if not isinstance(step, list): continue
        
        for p_idx, player_data in enumerate(step):
            action = player_data.get("action", {})
            if not isinstance(action, dict): continue
            
            hands_actions = action.get("hands", [])
            farmer_action = action.get("farmer", [])
            
            all_actions = []
            if farmer_action: all_actions.append(("FARMER", farmer_action))
            for i, ha in enumerate(hands_actions):
                if ha: all_actions.append((f"HAND_{i}", ha))
                
            for who, act in all_actions:
                if act[0] == "PICKUP" and len(act) > 1 and act[1] == "GOOSE":
                    # Check next turn if they actually have a goose!
                    print(f"Step {step_no} Player {p_idx} {who} tried PICKUP GOOSE.")
                    if step_no + 1 < len(replay["steps"]):
                        next_obs = replay["steps"][step_no + 1][0].get("observation", {})
                        farms = next_obs.get("farms", [])
                        if farms and p_idx < len(farms):
                            priv = next_obs.get("private", {}) if p_idx == next_obs.get("player") else {} # wait, replay has private for player 0 in step[0], player 1 in step[1]
                            
                        # proper extraction
                        next_priv = replay["steps"][step_no + 1][p_idx].get("observation", {}).get("private", {})
                        shed = next_priv.get("shed", {})
                        invs = next_priv.get("inventories", [])
                        
                        inv = {}
                        if who == "FARMER" and len(invs) > 0:
                            inv = invs[0]
                        elif who.startswith("HAND_"):
                            idx = int(who.split("_")[1]) + 1
                            if idx < len(invs):
                                inv = invs[idx]
                        
                        print(f"  -> Next turn: Shed GOOSE: {shed.get('GOOSE', 0)}, Worker Inv GOOSE: {inv.get('GOOSE', 0)}")
                        
                        if inv.get('GOOSE', 0) > 0:
                            print("  -> SUCCESS! Worker has GOOSE.")
                            # trace what this worker does next
                            for future_step in range(step_no + 1, min(step_no + 20, len(replay["steps"]))):
                                f_act = replay["steps"][future_step][p_idx].get("action", {})
                                if who == "FARMER":
                                    a = f_act.get("farmer", [])
                                else:
                                    idx = int(who.split("_")[1])
                                    h_acts = f_act.get("hands", [])
                                    a = h_acts[idx] if idx < len(h_acts) else []
                                print(f"    Turn {future_step} action: {a}")
                                if a and a[0] == "DROP":
                                    print("      -> DROPPED!")
                            return
                        else:
                            print("  -> FAILED!")
                            
if __name__ == "__main__":
    run()
