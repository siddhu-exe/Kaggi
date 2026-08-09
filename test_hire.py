from kaggle_environments import make


def agent(obs):

    if obs["step"] == 0:
        return {
            "farmer": ["PASS"],
            "hands": [],
            "market": [
                ["HIRE"]
            ]
        }

    return {
        "farmer": ["PASS"],
        "hands": [],
        "market": []
    }


env = make(
    "kaggriculture",
    configuration={
        "episodeSteps": 48
    },
    debug=True
)

env.run([agent, "random"])

for step_no, step in enumerate(env.steps):

    if step_no in [0, 1, 2, 23, 24, 25, 47]:

        print("\nSTEP:", step_no)

        try:
            player = step[0]

            print(
                "reward:",
                player.get("reward")
            )

            obs = player.get("observation", {})
            farms = obs.get("farms", [])

            if farms:
                print(
                    "hands:",
                    farms[0].get("hands")
                )

                print(
                    "hires_today:",
                    farms[0].get("hires_today")
                )

                print(
                    "money:",
                    farms[0].get("money")
                )

        except Exception as e:
            print("ERROR:", e)
