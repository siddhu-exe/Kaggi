"""Export a Kaggriculture replay using the built-in Kaggle HTML visualizer.

Usage:
    /home/siddharth/Desktop/Projects/projects/Kaggriculture/kagg/bin/python visualization.py
    /home/siddharth/Desktop/Projects/projects/Kaggriculture/kagg/bin/python visualization.py replay.json -o replay.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.utils import Struct


def wrap(value):
    if isinstance(value, dict):
        return Struct(**{key: wrap(inner) for key, inner in value.items()})
    if isinstance(value, list):
        return [wrap(item) for item in value]
    return value


def load_replay(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_html(replay: dict) -> str:
    env = make(
        "kaggriculture",
        configuration=replay.get("configuration", {}),
        debug=True,
    )
    env.steps = [[wrap(state) for state in step] for step in replay.get("steps", [])]
    return env.render(mode="html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the default Kaggle HTML replay viewer.")
    parser.add_argument("replay", nargs="?", default="replay.json", help="Path to the replay JSON file.")
    parser.add_argument("-o", "--output", default="visualization.html", help="Output HTML file.")
    args = parser.parse_args()

    replay = load_replay(Path(args.replay))
    html = build_html(replay)
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"Wrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()