import json
from kaggle_environments import make
from baseline_main_v8 import agent as baseline_agent
from main_v8 import agent as candidate_agent

def run_sim(agent1, agent2, name, outfile):
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
    
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(env.toJSON(), f)
    print(f"Saved replay to {outfile}")

run_sim(baseline_agent, baseline_agent, "V8 vs V8", "baseline_vs_baseline.json")
run_sim(candidate_agent, baseline_agent, "Candidate vs V8", "candidate_vs_baseline.json")
run_sim(candidate_agent, "starter", "Candidate vs Starter", "candidate_vs_starter.json")
run_sim(candidate_agent, "random", "Candidate vs Random", "candidate_vs_random.json")
