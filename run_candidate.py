import json
from kaggle_environments import make
from old_main_v8 import agent as candidate_agent
import old_main_v8

def run_sim(agent1, agent2, name):
    print(f"\n========================================================")
    print(f"RUNNING: {name}")
    print(f"========================================================")
    
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
    env.run([agent1, agent2])
    
    final = env.steps[-1]
    
    p1 = final[0]
    p2 = final[1] if len(final) > 1 else {}
    
    print(f"PLAYER 1: reward={p1.get('reward')} status={p1.get('status')}")
    print(f"PLAYER 2: reward={p2.get('reward')} status={p2.get('status')}")
    
    obs = p1.get('observation', {})
    farms = obs.get('farms', [])
    farm = farms[0] if farms else {}
    
    print(f"final cash = {farm.get('money', 0)}")
    
    opp_farm = farms[1] if len(farms) > 1 else {}
    print(f"opponent cash = {opp_farm.get('money', 0)}")
    
    if farm.get('money', 0) > opp_farm.get('money', 0):
        print("win/loss = WIN")
    else:
        print("win/loss = LOSS")
        
    p1_first_0 = -1
    for step_idx, step in enumerate(env.steps):
        if step[0].get('observation', {}).get('farms', [])[0].get('money', 0) == 0:
            p1_first_0 = step_idx // 24
            break
            
    print(f"day of first $0 = {p1_first_0}")
    
    # We want to measure the average workers, productive tiles, dead crops, market orders
    # To do this correctly, we'd need telemetry. 
    # But since old_main_v8 doesn't log these to files, we will just rely on the output.

run_sim(candidate_agent, candidate_agent, "V8 vs V8")
run_sim(candidate_agent, "starter", "V8 vs Starter")
run_sim(candidate_agent, "random", "V8 vs Random")
