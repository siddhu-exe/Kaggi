# Kaggriculture — Game Model & Agent Guide

A single reference for any LLM (or human) picking up this project. Part 1 is a
condensed, correct restatement of the game; Part 2 explains how our `main.py`
agent is built and where to tune it. The authoritative spec is
[`../README.md`](../README.md) and [`../AGENTS.md`](../AGENTS.md) — when in doubt,
those win.

---

## Part 1 — The Game (condensed)

**What it is.** Two-player, turn-based farming/economy sim. Each player runs a
farm, plants/harvests crops, raises animals, hires help, and trades on a shared
dynamic market. **Win = most coins in the bank at the end of the season.** Unsold
inventory is worth nothing at the end — only banked money counts.

**Timeline.** 24 turns/day × 30 days = **720 turns**. Start with **$3000**.

**Board.** `boardSize × boardSize` = **10×10**, four **5×5 quadrants**. Only **NW**
is unlocked at start; buy **NE/SW/SE** via `BUY_LAND` for **$1k / $2k / $4k**.

### Coordinate convention (easy to get wrong)
- Tiles are indexed **`tiles[y][x]`** (row `y` first).
- Unit positions are **`[x, y]`** (column first).
- `y` grows **southward**: `SOUTH = +y`, `NORTH = -y`, `EAST = +x`, `WEST = -x`.
- NW quadrant tiles are `x,y ∈ 0..4`.

### The shed / inventory / SELL pipeline (the biggest gotcha)
- The **shed** is a central store, cap **100 non-seed items**. It is **not a tile**
  — it never appears in `tiles`.
- "**Shed-adjacent**" = standing on one of the four center tiles `(4,4) (5,4)
  (4,5) (5,5)`. With only NW unlocked, the only unlocked one is **`(4,4)`**.
- `HARVEST` / `PICKUP` put goods into the acting **unit's inventory**, *not* the
  shed.
- **`SELL` sells from `private["shed"]`**, not from inventory. Goods reach the shed
  by `DROP` (on a center tile) or by the **automatic end-of-day inventory drop**
  (every unit dumps its inventory into the shed at day end; overflow past 100 is
  discarded).
- Consequence: **selling naturally lags harvesting by up to a day.** Don't expect
  to sell something the same turn you harvest it unless a unit walks it to the shed
  and `DROP`s.
- **Seeds** live in their own uncapped slot; `PLANT` consumes them directly.

### Crops
| Crop | Seed $ | First yield | Max-yield day | Type | Base $ | Notes |
|------|-------:|:-----------:|:-------------:|------|-------:|-------|
| Wheat | 10 | day 2 | day 4 | one-time | 25 | staple, feed, absorbs gluts |
| Carrot | 20 | day 2 | day 3 | one-time | 35 | **crashes on gluts** — pace sells |
| Tomato | 50 | day 8 | day 11 | ongoing (×4) | 60 | daily production once mature |
| Strawberry | 100 | day 10 | day 16 | ongoing (×4) | 120 | **premium, crashes on gluts** |
| Melon | 80 | day 10 | day 10 | one-time | 250 | **premium, crashes on gluts** |

- **Watering** is mandatory **every day**. **2 consecutive missed days → weed
  (total loss).** A **freshly planted seed starts at `consecutive_unwatered = 1`**
  (planting day counts) → it **must be watered the same day** or it weeds that
  night.
- **Watering bonus** (one-time crops): watering during the window (starts at
  `ceil(max_yield_day / 2)`) adds **+1 unit/day** to final yield. **`FERTILIZE`
  doubles that to +2/day for 3 days**, but only on days the plant is also watered.
- **Ongoing crops** (tomato/strawberry): scheduled production of 1/tick, **doubled
  to 2 if fertilized AND watered** that day. Capped at 4 scheduled yields, then the
  plant decays into a weed.

### Animals
| Animal | Cost | Structure | Product | First yield | Interval | Base $ |
|--------|-----:|-----------|---------|:-----------:|:--------:|-------:|
| Goose | 300 | Coop (+1 build) | Egg | day 4 | daily | 50 |
| Cow | 400 | Pasture (+1 build) | Milk | day 8 | every 2 days | 160 |
| Sheep | 500 | Pasture (+1 build) | Wool | day 6 | every 3 days | 200 |

- Flow: **build structure** (unit action `BUILD_COOP`/`BUILD_PASTURE`) → **buy
  animal** (`BUY_ANIMAL`) → carry from shed (`PICKUP`) → **`PLACE`** on the matching
  empty structure.
- **Fed wheat daily.** 2 missed days → **escape (unrecoverable)**. A newly placed
  animal survives its first day unfed (`consecutive_unfed = 0`).
- **`CARE`** banks +1 bonus per fed-and-cared day; paid out in full on the **next
  scheduled production** tick (only if fed that day), then resets.
- **`COLLECT_FERTILIZER`**: 1/animal/day, available end-of-day, does **not**
  accumulate (skip a few days → still just 1).
- Animals produce **indefinitely** while fed; `max_held` caps unharvested product
  on the tile, not lifetime output.

### Hiring
- `HIRE` is a market order. Cost = `fib(hires_today)` = **1, 1, 2, 3, 5, 8, …**,
  **resets each day**. `farms[player]["hires_today"]` drives the next cost.
- Hands re-spawn at the shed each day and disappear at day end (drop inventory
  first). So 3 hands/day costs only **1+1+2 = $4/day** — very cheap.

### Market & prices
- ≤ **10 market orders per turn** (`maxMarketOrdersPerTurn`); extras silently
  dropped. Orders processed one unit at a time, concurrently across players.
- Only **WHEAT** and **FERTILIZER** are buyable (`BUY_PRODUCT`). Everything is
  sellable (`SELL`).
- Prices move with market inventory: `base` at `I0=10000`, **up** as inventory
  falls, **down** as it grows. Each resource has independent shape/target per side.
- **Glut sensitivity** (from the price table):
  - **Crash to $1 on oversupply → pace & cap sells:** carrot, strawberry, melon,
    milk, wool.
  - **Absorb gluts gently → sell freely:** wheat, eggs.
- **Town demand** drains market inventory (raising prices). New shops unlock every
  3 days (random). Town-center demand scales **2× after day 10, 4× after day 20**.
  Selling into currently-demanded goods gets better prices.

### Turn processing order
1. Validate actions → 2. Record player actions → 3. Process market queue (in
order) → 4. Town consumption → 5. Update obs: day refresh (reset watered/fed,
grow, decay, weeds), market price refresh, income update, farm update.

### Observation & action schema (essentials)
- `obs`: `player, step, day, hour, farms[2], market{inventory,prices},
  town{unlocked_shops}, private{shed, seeds, inventories}`.
- `farm`: `money, tiles[y][x], farmer[x,y], hands[[x,y]…], unlocked_quadrants,
  hires_today`. Opponent's farm is **public**; their `private` is hidden.
- Tile is `None` (empty), `"LOCKED"`, a `PLANT` dict, a `WEED` dict, or a
  `COOP`/`PASTURE` dict (optionally with `animal`).
- Action dict: `{"farmer":[op,…], "hands":[[op,…],…], "market":[[op,…],…]}`. The
  `hands` list length must match `len(farm["hands"])`.

---

## Part 2 — Our Agent (`main.py`)

### Strategy in one paragraph
A **phased** economy. **Phase 1 (Foundation, day 0–~4):** wheat + carrot staple
engine; hire 3 hands; sell wheat freely, pace carrot; hold a wheat reserve as
future animal feed. **Phase 2 (Transition, day ~5–9):** add tomato; build a coop +
buy a goose once wheat feed is sustainable; feed + `CARE` daily; start fertilizing
high-value crops inside their bonus window. **Phase 3 (Expansion, day 10+):** buy
land only when the farm is nearly full **and** cash is healthy; add
strawberry/melon/cow/sheep; give premium crops/animals first priority for
water/feed/care/fertilizer; pace premium sells; bias selling toward town-demanded
goods; cut production of a glut good the opponent is already flooding.

### Code map
| Piece | Responsibility |
|-------|----------------|
| `GameState` / `STATE` | Module-global, persists across turns within an episode. Holds `phase`, `tile_plan` (intended crop/structure per tile), `last_day_hired`, `want_coop`, `want_pastures`. `reset()` at episode start via `_reset_if_new_episode`. |
| `TurnContext` | Parses `obs` once per turn: money, tiles, hands, shed, seeds, prices, town. Scans the world into task lists: `harvest_tiles`, `water_tiles`, `fertilize_tiles`, `weed_tiles`, `empty_tiles`, `animal_tiles`, `empty_structs`. Tracks `claimed` (tiles handled this turn) and `shed_ledger`/`seeds_remaining` (mutable, so units don't double-spend). |
| `update_phase` | Soft day-based phase advance (≥5 → 2, ≥10 → 3); real gating is by cash/infrastructure in the buy logic. |
| `assign_plans` / `desired_allocation` | Decide what each empty tile should become; keep stable per-tile plans; adjust down if the opponent is flooding a glut good. |
| `decide_unit_action(ctx, state, idx)` | **Priority checklist** for one unit (farmer = idx 0, hands = idx≥1). First match wins: HARVEST → WATER → FERTILIZE → FEED → CARE → COLLECT_FERTILIZER → PLACE animal → BUILD → PLANT → DIG weed; else move to nearest highest-priority task; else carry produce to shed / `DROP`; else `PASS`. High-value tiles outrank staples via `_tile_value`. |
| `decide_market_orders(ctx, state)` | Assembles, in priority order then truncates to 10: HIRE → BUY_LAND → BUY_ANIMAL → BUY_SEED → BUY_PRODUCT wheat (feed) → BUY_PRODUCT fertilizer → paced SELLs. |
| `decide_hiring(ctx, state)` | Once/day at hour 0; 3 hands in phases 1–2, scales with land in phase 3; Fibonacci-cost-aware. |
| `decide_sells(ctx, state, feed_reserve)` | Per-good SELL orders honoring per-turn caps, a "don't sell below `min_frac × base`" price gate for glut goods, the wheat feed reserve, and a forced dump when the shed nears overflow. Biases town-demanded goods first. |
| `agent(obs)` | Top-level dispatcher. **Wrapped in try/except** — any bug returns a safe `{"farmer":["PASS"], "hands":[["PASS"]…], "market":[]}` so a runtime error can never fail the validation episode. |

### Key invariants the code maintains
- **Never plant what can't be watered today.** `PLANT` only fires when the seed is
  held **and** `hour < turnsPerDay - 2` (time left for a unit to water it), so a
  fresh seed never weeds on planting night.
- **Always hold a wheat feed reserve** = `animals × 3 + 2`. Wheat sells only the
  surplus above this; feed top-ups buy wheat when short.
- **Pace glut goods.** `SELL_RULES` caps per-turn volume (strawberry/melon/milk/
  wool ≤ 2, carrot ≤ 3, tomato ≤ 6; wheat/eggs uncapped) and skips selling below a
  price floor — unless the shed is about to overflow (then dump to avoid discard).
- **Loose multi-unit de-confliction.** Each unit claims the nearest unclaimed task
  (`ctx.claimed`), so hands spread out instead of stacking on one tile.

### Tuning knobs (where to adjust behavior)
- **Phase day thresholds:** `update_phase` (`day >= 5`, `day >= 10`). Every
  phase-gated branch is marked with a `# --- PHASE N ---` comment.
- **Cash reserves:** `RESERVE_FLOOR = {1:800, 2:600, 3:500}` (seed-money floor);
  big-buy buffers inline in `decide_market_orders` (animal buffer, land `+1200`).
- **Tile allocation targets:** `desired_allocation` (counts per crop/structure per
  phase).
- **Sell pacing:** `SELL_RULES` (`cap`, `min_frac`), `SHED_OVERFLOW_FORCE`.
- **Feed safety:** `_wheat_feed_reserve`.
- **Hiring counts:** `decide_hiring` (`target` per phase).
- **Which crops get fertilizer:** `FERTILIZE_CROPS`.
- **Glut classification:** `GLUT_SENSITIVE`, `GLUT_TOLERANT`.

### Testing
`main.py` has an `if __name__ == "__main__":` harness that runs the agent vs
`random` and `starter` over a full 720-turn season and prints final rewards:

```bash
pip install -U kaggle-environments
python main.py
```

It also imports cleanly (the harness is guarded), so Kaggle's
`from main import agent` works for submission:

```bash
kaggle competitions submit kaggriculture -f main.py -m "phased agent v1"
```

### Known limitations / future work (for the RL folder or a later pass)
- Selling lags harvest by design (shed pipeline); a smarter courier policy could
  route a dedicated hand to `DROP` mid-day for faster liquidity.
- Animal expansion (cow/sheep) is conservative; thresholds in `decide_market_orders`
  and `desired_allocation` can be pushed once cash flow is validated in replays.
- No explicit opponent-price modeling beyond the flood-avoidance heuristic in
  `desired_allocation`; a reinforcement-learning agent (the `reinforement agent/`
  folder, intentionally left empty here) could learn sell timing and expansion
  cadence from replays.
