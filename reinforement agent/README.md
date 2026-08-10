# Strategy iteration

`tune.py` performs small, manual evolutionary-search batches. It perturbs the
single `PARAMS` configuration in `../main.py`, evaluates each candidate in fresh
processes against starter/random/self-play, logs JSONL results, and saves the
winner to `best_params.json` as the next batch's seed.

Run from this folder:

```bash
python tune.py --variants 8 --episodes 3
python tune.py --variants 4 --episodes 3 --replay ../replay.json
```

Replay input currently supplies realized-score validation. The `replay_score`
function is the explicit integration point for a future counterfactual action
replayer.
