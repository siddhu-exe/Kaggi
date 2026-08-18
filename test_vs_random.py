from kaggle_environments import make
from main import agent, STATES

env = make('kaggriculture', configuration={'episodeSteps': 720}, debug=True)
STATES[0].reset()
STATES[1].reset()
env.run([agent, 'random'])

final = env.steps[-1]
for i, s in enumerate(final):
    obs = s.observation
    if obs:
        farms = obs['farms'][i]
        print(f'Player {i}: money={farms["money"]}, unlocked={farms["unlocked_quadrants"]}, hands={len(farms["hands"])}')
        if 'private' in obs:
            shed = obs['private']['shed']
            print(f'  Shed: {shed}')