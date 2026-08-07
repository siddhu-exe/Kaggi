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
**Deploy capital on day 0, then scale from there — no phases.** In a 30-day season
idle cash produces nothing, so on turn 1 we spend close to the full $3000: hire the
max **5 hands** (≈ $12 under Fibonacci — near-free labor), **buy animals + their
structures immediately** (buying market wheat as feed rather than waiting to grow
it, since animals take days to reach first production), and **buy seed across ALL
crop types** to fill every tile — staples *and* premium/ongoing crops from the
start. Everything then matures while we sit near-flat, and income spikes mid-game
as animals and ongoing crops hit production stride. **Selling is price-reactive:**
every turn we read the live market price, compute `price / base`, and scale sell
volume to it — dump near/above base, throttle hard toward the $1 floor (steeper for
premium goods, shallow for wheat/eggs). Because both players sell into the *same
shared price curve*, reacting to the real price is the correct response — there is
no fixed cap schedule and no opponent "targeting" us. We also **diversify away from
the opponent**: their farm is public, so we back off glut-sensitive crops they're
flooding and lean into lanes they've left open. Land expands only when the current
quadrant is nearly full **and** cash is healthy.

### Code map
| Piece | Responsibility |
|-------|----------------|
| `GameState` / `STATE` | Module-global, persists across turns within an episode. Holds `tile_plan` (intended crop/structure per tile), `last_day_hired`, plus day-0 bookkeeping (`animals_queued`, `feed_bought`). `reset()` at episode start via `_reset_if_new_episode`. |
| `TurnContext` | Parses `obs` once per turn: money, tiles, hands, shed, seeds, prices, town. Scans the world into task lists: `harvest_tiles`, `water_tiles`, `fertilize_tiles`, `weed_tiles`, `empty_tiles`, `animal_tiles`, `empty_structs`. Tracks `claimed` (tiles handled this turn) and `shed_ledger`/`seeds_remaining` (mutable, so units don't double-spend). |
| `update_phase` | **No-op stub** (kept so the `agent()` dispatcher is untouched). The phased strategy was replaced by day-0 deployment. |
| `assign_plans` / `desired_allocation` | Decide what each empty tile should become. `desired_allocation` returns a **single** aggressive plan (`BASE_ALLOCATION`) used from day 0, scaled up per unlocked quadrant, then adjusted for **opponent diversification** (back off glut goods they flood, lean into ones they have zero of). |
| `decide_unit_action(ctx, state, idx)` | **Priority checklist** for one unit (farmer = idx 0, hands = idx≥1). First match wins: HARVEST → WATER → FERTILIZE → FEED → CARE → COLLECT_FERTILIZER → PLACE animal → BUILD → PLANT → DIG weed; else move to nearest highest-priority task; else carry produce to shed / `DROP`; else `PASS`. High-value tiles outrank staples via `_tile_value`. *(Unchanged in this pivot.)* |
| `decide_market_orders(ctx, state)` | Assembles, then truncates to 10: HIRE → **BUY_ANIMAL** (front-loaded) → **BUY_SEED all crops** → **BUY_PRODUCT wheat feed** → BUY_PRODUCT fertilizer → BUY_LAND → price-reactive SELLs. Spends down toward `WORKING_CASH` — no big reserve. |
| `decide_hiring(ctx, state)` | Once/day at hour 0; **`HANDS_PER_DAY` (5) every day**, Fibonacci-cost-aware. |
| `_struct_animal_counts(ctx)` | Counts planned vs built coops/pastures and owned animals (shed + placed), so we front-load animal buys without re-buying after placement. |
| `decide_sells(ctx, state, feed_reserve)` → `_sell_quantity` | **Price-reactive** SELL orders: volume scales linearly between a per-good `SELL_THROTTLE` floor and base price; dump at/above base or on overflow; hold near the floor. Carves out the wheat feed reserve; biases town-demanded goods first; clamps premium goods to ≤ half the holding per turn. |
| `agent(obs)` | Top-level dispatcher. **Wrapped in try/except** — any bug returns a safe `{"farmer":["PASS"], "hands":[["PASS"]…], "market":[]}` so a runtime error can never fail the validation episode. |

### Key invariants the code maintains
- **Never plant what can't be watered today.** `PLANT` only fires when the seed is
  held **and** `hour < turnsPerDay - 2` (time left for a unit to water it), so a
  fresh seed never weeds on planting night.
- **Feed before you own.** The wheat feed reserve is sized to animals we *intend* to
  own (planned + built structures), not just placed ones, and we buy market wheat to
  reach it — so animals are safe to buy on day 0. Wheat sells only the surplus above
  this reserve.
- **Price-reactive selling.** `_sell_quantity` scales volume to `price/base` between
  a per-good `SELL_THROTTLE` floor and 1.0; dumps at/above base or on shed overflow;
  holds near the floor. Premium goods are additionally clamped to ≤ half the holding
  per turn so one order can't tank the shared curve.
- **Loose multi-unit de-confliction.** Each unit claims the nearest unclaimed task
  (`ctx.claimed`), so hands spread out instead of stacking on one tile.

### Tuning knobs (where to adjust behavior)
- **Day-0 allocation:** `BASE_ALLOCATION` (counts per crop/structure) and the
  per-quadrant scale-up in `desired_allocation`.
- **Working cash / spend aggression:** `WORKING_CASH` (single small buffer; raise it
  to spend less aggressively).
- **Hiring:** `HANDS_PER_DAY`.
- **Sell throttle:** `SELL_THROTTLE` (`floor` per good — lower = sell cheaper),
  `SHED_OVERFLOW_FORCE`, and the premium half-clamp in `_sell_quantity`.
- **Feed safety:** `_wheat_feed_reserve` (multiplier per animal).
- **Land expansion:** `nearly_full` threshold + cash buffer in
  `decide_market_orders` step 5.
- **Which crops get fertilizer:** `FERTILIZE_CROPS`.
- **Glut classification:** `GLUT_SENSITIVE`, `GLUT_TOLERANT`.
- **Opponent diversification:** the flood/open-lane rule in `desired_allocation`.

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
kaggle competitions submit kaggriculture -f main.py -m "front-loaded agent v2"
```

### Known limitations / future work (for the RL folder or a later pass)
- The 10-orders/turn cap means a full day-0 buy list (5 hires + 2 animals + several
  seed/feed buys) spills across the first few turns rather than all landing on turn
  0 — fine in practice, but worth noting when reading replays.
- Selling lags harvest by design (shed pipeline); a smarter courier policy could
  route a dedicated hand to `DROP` mid-day for faster liquidity.
- Opponent modeling is a simple flood-avoidance / open-lane heuristic in
  `desired_allocation`; a reinforcement-learning agent (the `reinforement agent/`
  folder, intentionally left empty here) could learn sell timing and expansion
  cadence from replays.
