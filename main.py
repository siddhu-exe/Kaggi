"""
Kaggriculture competitive agent — V6 economic agent
==========================================

A single-file, defensive agent for the Kaggle "Kaggriculture" two-player farming
simulation.  Strategy: **deploy capital on day 0, then scale from there.**

    In a 30-day season, idle cash produces nothing.  So on turn 1 we spend close
    to the full $3000: max out hands (5 hands ~= $12 under Fibonacci pricing),
    buy animals + their structures immediately (buying wheat feed from the market
    rather than waiting to grow it), and buy seed across ALL crop types to fill
    every tile.  Things then mature while we sit near-flat, and income spikes
    mid-game as animals and ongoing crops hit production stride.  There are no
    gated phases — everything starts day 0 and simply scales up.

Selling is **price-reactive**, not a fixed schedule: every turn we read the live
market price, compute price/base, and scale sell volume to it — dump near/above
base, throttle hard toward the $1 floor.  This reacts to real conditions (which
include our OWN selling moving the shared price curve) instead of guessing a cap.

We also **diversify away from the opponent**: their farm is public, so we scan
what they grow and lean toward glut-sensitive goods they are NOT flooding.

Architecture:
    * GameState            module-global object, persists across turns.
    * TurnContext           per-turn parsed view of the observation + task lists.
    * decide_unit_action    one action per farmer / hand (priority checklist).
    * decide_market_orders  BUY_SEED / BUY_ANIMAL / BUY_PRODUCT / SELL / BUY_LAND.
    * decide_hiring         Fibonacci-priced HIRE orders at day start (5/day).
    * agent(obs)            top-level dispatcher, wrapped so it NEVER throws.

Key spec facts baked in:
    * tiles are indexed tiles[y][x]; positions are [x, y].
    * SELL pulls from private["shed"]; HARVEST/PICKUP fill a unit's *inventory*.
      Produce reaches the shed via DROP (on a center tile) or the automatic
      end-of-day inventory drop.  So selling naturally lags harvest by ~a day.
    * A freshly planted seed has consecutive_unwatered = 1 -> it MUST be watered
      the same day or it weeds that night.  We only plant when there is time left
      in the day for a unit to water it.
    * FEED and FERTILIZE consume the acting unit's inventory. Units explicitly
      PICKUP wheat/fertilizer at a shed-adjacent tile before travelling to work.
    * Glut-sensitive goods (carrot, strawberry, melon, milk, wool) crash toward
      the $1 floor -> steep sell throttle.  Wheat / eggs absorb gluts -> shallow
      throttle, sell freely (keeping a wheat feed reserve).
"""

import json
import math
import os

# ---------------------------------------------------------------------------
# CONSTANTS  (tune these — every threshold is commented for easy adjustment)
# ---------------------------------------------------------------------------

TURNS_PER_DAY = 24          # game default; overridden from obs when available
BOARD_SIZE = 10
SHED_HALF = BOARD_SIZE // 2  # = 5
# The four center tiles orthogonally adjacent to the shed (shed itself is no tile)
SHED_TILES = {
    (SHED_HALF - 1, SHED_HALF - 1),  # (4,4) NW  — only one unlocked at start
    (SHED_HALF,     SHED_HALF - 1),  # (5,4) NE
    (SHED_HALF - 1, SHED_HALF),      # (4,5) SW
    (SHED_HALF,     SHED_HALF),      # (5,5) SE
}

# Per-crop data.  first = first_yield_day, max_day = time-to-max-yield (age in
# days), ongoing = produces on a schedule vs one shot, base = base sale price.
CROPS = {
    "WHEAT":      {"seed": 10,  "first": 2,  "max_day": 4,  "ongoing": False, "base": 25},
    "CARROT":     {"seed": 20,  "first": 2,  "max_day": 3,  "ongoing": False, "base": 35},
    "TOMATO":     {"seed": 50,  "first": 8,  "max_day": 11, "ongoing": True,  "base": 60},
    "STRAWBERRY": {"seed": 100, "first": 10, "max_day": 16, "ongoing": True,  "base": 120},
    "MELON":      {"seed": 80,  "first": 10, "max_day": 10, "ongoing": False, "base": 250},
}

# Per-animal data.  structure = tile type it needs, product = shed item it makes.
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "product": "EGG",  "base": 50},
    "COW":   {"cost": 400, "structure": "PASTURE", "product": "MILK", "base": 160},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "product": "WOOL", "base": 200},
}

BASE_PRICES = {
    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
    "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100,
}

# Goods whose price collapses toward $1 on oversupply — pace + cap their sells.
GLUT_SENSITIVE = {"CARROT", "STRAWBERRY", "MELON", "MILK", "WOOL"}
# Goods with the steepest glut curves specifically — extra-cautious pacing.
PREMIUM_GLUT = {"MELON", "MILK", "WOOL", "STRAWBERRY"}
# Goods that absorb gluts gently — sell freely (wheat kept above feed reserve).
GLUT_TOLERANT = {"WHEAT", "EGG"}

# Crops worth spending fertilizer on (high value / good return on the bonus).
FERTILIZE_CROPS = {"MELON", "STRAWBERRY", "TOMATO", "CARROT"}

# Price-reactive selling thresholds (ratio = live_price / base_price).
PARAMS = {
    # Capacity / labor.  A conservative daily load cap prevents planting faster
    # than the available units can water and feed.
    "hands_per_day": 8,
    "actions_per_unit_day": 18,
    "crop_daily_load": 1.0,
    "animal_daily_load": 3.0,
    "working_cash": 50,
    "feed_days": 4,
    "endgame_day": 26,
    "land_days": [5, 12, 18],
    "land_last_day": 18,
    "land_reserves": [700, 1400, 2200],
    "shed_overflow_force": 85,
    # Allocation weights are normalized to current unlocked/serviceable space.
    "allocation": {
        "MELON": 45, "TOMATO": 20, "STRAWBERRY": 12,
        "WHEAT": 8, "CARROT": 7, "COOP": 2, "PASTURE": 6,
    },
    "pasture_sheep_fraction": 0.5,
    # Price/base ratio -> fraction of shed stock sold this turn.
    "premium_sell": [[0.85, 0.50], [0.65, 0.25], [0.45, 0.10]],
    "normal_sell": [[0.70, 0.75], [0.45, 0.40], [0.0, 0.10]],
    "fertilizer_buy_price": 120,
    "wheat_buy_price": 40,
}

# Optional tuner override; Kaggle submissions simply omit this environment var.
try:
    if os.environ.get("KAGGRICULTURE_PARAMS"):
        PARAMS.update(json.loads(os.environ["KAGGRICULTURE_PARAMS"]))
except Exception:
    pass

SHED_CAPACITY = 100
SHED_OVERFLOW_FORCE = 85    # when shed non-seed items exceed this, force paced sells
MAX_MARKET_ORDERS = 10

# Tiny working-cash buffer so we never fail mid-order — we otherwise spend it all.
WORKING_CASH = PARAMS["working_cash"]

# Hands to hire every day.  5 hands ~= $1+1+2+3+5 = $12/day under Fibonacci — the
# labor is near-free, so there is no reason to hold back.
HANDS_PER_DAY = PARAMS["hands_per_day"]
MAX_TARGET_TILES = 100
ENDGAME_DAY = PARAMS["endgame_day"]

# Single aggressive day-0 allocation for the opening (25-tile NW quadrant).  No
# phases: this is the target from turn 1.  Ongoing crops (tomato/strawberry) and
# animals are represented immediately so they start their long maturation early.
BASE_ALLOCATION_P0 = list(PARAMS["allocation"].items())

# Same economic portfolio for either submission slot. The opponent is handled
# through runtime market/task adaptation rather than a hard-coded P1 economy.
BASE_ALLOCATION_P1 = list(BASE_ALLOCATION_P0)



# ---------------------------------------------------------------------------
# GAME STATE  (module-global; Kaggle keeps our script globals alive per episode)
# ---------------------------------------------------------------------------

class GameState:
    def __init__(self):
        self.reset()

    def reset(self):
        # Intended purpose per tile: {(x,y): {"kind":"CROP","crop":X}} or
        # {(x,y): {"kind":"STRUCT","structure":"COOP","animal":"GOOSE"}}
        self.tile_plan = {}
        self.last_day_hired = {}      # guard: hire once per player/day
        self.animals_queued = set()  # animals we committed to buy on day 0
        self.feed_bought = False     # bought the initial market-wheat feed stock
        self.notes = {}              # scratch for debugging / future tuning
        self.near_full_days = 0      # consecutive days current land is >=94% productive
        self.utilization_day = -1    # guard utilization accounting once per day


STATES = {0: GameState(), 1: GameState()}


def _state_for_player(player):
    if player not in STATES:
        STATES[player] = GameState()
    return STATES[player]


def _reset_if_new_episode(obs):
    """Re-init globals at the very start of an episode (step 0)."""
    try:
        if int(obs.get("step", 0)) == 0 and int(obs.get("day", 0)) == 0:
            _state_for_player(int(obs.get("player", 0))).reset()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------------------------

def _get(d, key, default=0):
    try:
        v = d.get(key, default)
        return v if v is not None else default
    except Exception:
        return default


def get_tile(tiles, x, y):
    """Safe tiles[y][x] with bounds checking. Returns 'LOCKED' for off-board."""
    try:
        if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
            return tiles[y][x]
    except Exception:
        pass
    return "LOCKED"


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def move_toward(src, dst):
    """Return a movement op stepping one tile from src toward dst (y grows south)."""
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return ["EAST"] if dx > 0 else ["WEST"]
    if dy != 0:
        return ["SOUTH"] if dy > 0 else ["NORTH"]
    if dx != 0:
        return ["EAST"] if dx > 0 else ["WEST"]
    return ["PASS"]


def nearest_shed_tile(pos, unlocked):
    """Nearest reachable shed-adjacent tile (only unlocked ones are actionable)."""
    cands = [t for t in SHED_TILES if t in unlocked]
    if not cands:
        cands = list(SHED_TILES)  # fall back; still passable to stand on
    return min(cands, key=lambda t: manhattan(pos, t))


def bonus_window_start(cropinfo):
    return math.ceil(cropinfo["max_day"] / 2)


# ---------------------------------------------------------------------------
# TURN CONTEXT  — parse the observation once per turn into a convenient view.
# ---------------------------------------------------------------------------

class TurnContext:
    def __init__(self, obs):
        self.obs = obs
        self.player = int(obs.get("player", 0))
        self.step = int(obs.get("step", 0))
        self.tpd = TURNS_PER_DAY
        self.day = int(obs.get("day", self.step // self.tpd))
        self.hour = int(obs.get("hour", self.step % self.tpd))

        farms = obs.get("farms", []) or []
        self.farm = farms[self.player] if self.player < len(farms) else {}
        self.opp_farm = farms[1 - self.player] if len(farms) > 1 else {}

        self.tiles = self.farm.get("tiles", []) or []
        self.money = float(_get(self.farm, "money", 0))
        self.hands = self.farm.get("hands", []) or []
        self.unlocked_quadrants = self.farm.get("unlocked_quadrants", ["NW"]) or ["NW"]
        self.hires_today = int(_get(self.farm, "hires_today", 0))

        private = obs.get("private", {}) or {}
        self.shed = dict(private.get("shed", {}) or {})
        self.seeds_remaining = dict(private.get("seeds", {}) or {})   # mutable copy
        self.inventories = private.get("inventories", []) or []

        market = obs.get("market", {}) or {}
        self.prices = dict(market.get("prices", {}) or {})
        self.market_inv = dict(market.get("inventory", {}) or {})

        town = obs.get("town", {}) or {}
        self.unlocked_shops = town.get("unlocked_shops", []) or []

        # Mutable per-turn resource ledgers (so multiple units don't double-spend)
        self.shed_ledger = dict(self.shed)        # for FEED / PICKUP accounting
        self.claimed = set()                       # tile coords already handled/targeted

        # Unlocked tile coordinates (anything not "LOCKED").
        self.unlocked = set()
        for y in range(len(self.tiles)):
            row = self.tiles[y]
            for x in range(len(row)):
                if row[x] != "LOCKED":
                    self.unlocked.add((x, y))

        # Build task lists + world summary.
        self._scan_world()

    # ---- world scan --------------------------------------------------------
    def _scan_world(self):
        self.growing_counts = {}      # crop -> count of live plants
        self.placed_animals = 0       # animals needing feed
        self.empty_structs = []       # (x,y, structure, tile) coop/pasture w/o animal
        self.animal_tiles = []        # (x,y) occupied animal structures needing attention
        self.unfed_tiles = []
        self.harvest_tiles = []
        self.water_tiles = []
        self.fertilize_tiles = []
        self.weed_tiles = []
        self.empty_tiles = []         # unlocked None tiles

        for (x, y) in self.unlocked:
            tile = get_tile(self.tiles, x, y)
            if tile is None:
                self.empty_tiles.append((x, y))
                continue
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                crop = tile.get("crop")
                self.growing_counts[crop] = self.growing_counts.get(crop, 0) + 1
                info = CROPS.get(crop)
                if info is None:
                    continue
                if crop_ready(tile, info, self.day, self.step):
                    self.harvest_tiles.append((x, y))
                elif not tile.get("watered_today", False):
                    self.water_tiles.append((x, y))
                if fert_eligible(tile, info, self.day):
                    self.fertilize_tiles.append((x, y))
            elif kind == "WEED":
                self.weed_tiles.append((x, y))
            elif kind in ("COOP", "PASTURE"):
                animal = tile.get("animal")
                if animal is None:
                    self.empty_structs.append((x, y, kind, tile))
                else:
                    self.placed_animals += 1
                    unfed = not tile.get("fed_today", False)
                    uncared = tile.get("fed_today", False) and not tile.get("cared_today", False)
                    has_fert = tile.get("fertilizer_available", False)
                    has_yield = int(_get(tile, "yield_units", 0)) > 0
                    if unfed or uncared or has_fert or has_yield:
                        self.animal_tiles.append((x, y))
                    if unfed:
                        self.unfed_tiles.append((x, y))

        # Shed non-seed item total (for overflow-forced sells).
        self.shed_total = sum(int(v) for v in self.shed.values() if isinstance(v, (int, float)))

    # ---- inventory helpers -------------------------------------------------
    def unit_inv(self, idx):
        if 0 <= idx < len(self.inventories) and isinstance(self.inventories[idx], dict):
            return self.inventories[idx]
        return {}

    def unit_pos(self, idx):
        if idx == 0:
            f = self.farm.get("farmer", [SHED_HALF - 1, SHED_HALF - 1])
            return (int(f[0]), int(f[1]))
        h = idx - 1
        if h < len(self.hands):
            p = self.hands[h]
            return (int(p[0]), int(p[1]))
        return (SHED_HALF - 1, SHED_HALF - 1)


# ---------------------------------------------------------------------------
# CROP / ANIMAL PREDICATES
# ---------------------------------------------------------------------------

def crop_ready(tile, info, day, step):
    """Is this plant worth harvesting now?"""
    yu = int(_get(tile, "yield_units", 0))
    if yu <= 0:
        return False
    age = day - int(_get(tile, "planted_day", day))
    if info["ongoing"]:
        return True  # collect scheduled produce whenever it's present
    # One-time crop: harvest at peak (age >= max_day) or once it starts decaying.
    if age >= info["max_day"]:
        return True
    mls = int(_get(tile, "max_lifespan_step", -1))
    if mls != -1 and step >= mls:
        return True
    return False


def fert_eligible(tile, info, day):
    """Is this a high-value plant currently inside its fertilizer window & not
    already fertilized?  (Application still requires the unit to carry fertilizer.)"""
    crop = tile.get("crop")
    if crop not in FERTILIZE_CROPS:
        return False
    age = day - int(_get(tile, "planted_day", day))
    if info["ongoing"]:
        in_window = info["first"] <= age <= info["max_day"]
    else:
        in_window = bonus_window_start(info) <= age <= info["max_day"]
    if not in_window:
        return False
    if int(_get(tile, "fertilized_until_day", -1)) >= day:
        return False  # bonus already active
    return True


# ---------------------------------------------------------------------------
# PHASE MANAGEMENT  (no-op: replaced by day-0 capital deployment, kept as stub)
# ---------------------------------------------------------------------------

def update_phase(ctx, state):
    return


# ---------------------------------------------------------------------------
# TILE PLANNING  — decide what each empty tile *should* become.
# ---------------------------------------------------------------------------

def desired_allocation(ctx, state):
    """Use the full four-quadrant capacity.  P0/P1 deliberately use different
    portfolios so two copies of our agent do not manufacture the same glut."""
    raw = list(BASE_ALLOCATION_P0 if ctx.player == 0 else BASE_ALLOCATION_P1)
    # Only commit work the current daily labor can actually service.  Empty
    # purchased land remains available and is filled as labor grows.
    units = 1 + HANDS_PER_DAY
    budget = int(units * PARAMS["actions_per_unit_day"])
    animal_weight = PARAMS["animal_daily_load"]
    crop_weight = PARAMS["crop_daily_load"]
    raw_weight = sum(n * (animal_weight if k in ("COOP", "PASTURE") else crop_weight)
                     for k, n in raw)
    capacity_scale = min(1.0, budget / max(1.0, raw_weight))
    tile_scale = min(len(ctx.unlocked), MAX_TARGET_TILES) / float(MAX_TARGET_TILES)
    scale = min(capacity_scale, tile_scale)
    plan = [(k, max(0, int(round(n * scale)))) for k, n in raw]

    # If the opponent is heavily concentrated in one premium good, shift a few
    # tiles toward the other premium lane.  Keep the portfolio total at 100.
    opp = _opp_crop_counts(ctx)
    if opp.get("MELON", 0) >= 18:
        for i, (k, n) in enumerate(plan):
            if k == "MELON":
                plan[i] = (k, max(12, n - 6))
            if k == "TOMATO":
                plan[i] = (k, n + 4)
            if k == "CARROT":
                plan[i] = (k, n + 2)
    elif opp.get("TOMATO", 0) >= 18:
        for i, (k, n) in enumerate(plan):
            if k == "TOMATO":
                plan[i] = (k, max(10, n - 5))
            if k == "MELON":
                plan[i] = (k, n + 3)
            if k == "STRAWBERRY":
                plan[i] = (k, n + 2)

    # Normalize to unlocked capacity, not a fixed 100 before land exists.
    target_tiles = min(len(ctx.unlocked), sum(n for _, n in plan))
    total = sum(n for _, n in plan)
    if total < target_tiles:
        for i, (k, n) in enumerate(plan):
            if k == "MELON":
                plan[i] = (k, n + (target_tiles - total))
                break
    elif total > target_tiles:
        excess = total - target_tiles
        for i in range(len(plan) - 1, -1, -1):
            k, n = plan[i]
            if k in ("CARROT", "WHEAT", "STRAWBERRY") and excess > 0:
                cut = min(excess, max(0, n - 2))
                plan[i] = (k, n - cut)
                excess -= cut

    return plan


def _opp_crop_counts(ctx):
    counts = {}
    try:
        for row in ctx.opp_farm.get("tiles", []) or []:
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    c = tile.get("crop")
                    counts[c] = counts.get(c, 0) + 1
    except Exception:
        pass
    return counts

def assign_plans(ctx, state):
    """Fill every unlocked tile with a stable production plan, then keep the
    same plan as new quadrants unlock.  Pastures alternate cow/sheep by player."""
    for coord in list(state.tile_plan.keys()):
        if coord not in ctx.unlocked or get_tile(ctx.tiles, coord[0], coord[1]) is not None:
            del state.tile_plan[coord]

    planned_counts = {}
    pasture_animals = []
    for p in state.tile_plan.values():
        key = p["crop"] if p["kind"] == "CROP" else p["structure"]
        planned_counts[key] = planned_counts.get(key, 0) + 1
        if p.get("kind") == "STRUCT" and p.get("structure") == "PASTURE":
            pasture_animals.append(p.get("animal"))

    existing = dict(ctx.growing_counts)
    struct_counts = {"COOP": 0, "PASTURE": 0}
    for x, y in ctx.unlocked:
        t = get_tile(ctx.tiles, x, y)
        if isinstance(t, dict) and t.get("kind") in struct_counts:
            struct_counts[t["kind"]] += 1
    existing.update(struct_counts)

    # Animal structures are deliberately assigned first, to the tiles nearest
    # the initial farmer/hand staging area.  This keeps the daily feed/care/
    # collect routes short; crops can use the remaining tiles and retain the
    # old shed-oriented ordering.  The replay shows this materially reduces
    # worker travel for animal-heavy farms.
    free = [c for c in ctx.empty_tiles if c not in state.tile_plan]
    # Animal structures get the shortest routes to the shed.  Use the nearest
    # currently actionable shed tile (rather than a board corner); feeding,
    # care, and collection happen every day and are the most failure-prone work.
    shed_anchor = min(SHED_TILES, key=lambda t: manhattan(t, (SHED_HALF - 1, SHED_HALF - 1)))
    animal_free = sorted(free, key=lambda c: (manhattan(c, shed_anchor), c[1], c[0]))
    crop_free = sorted(free, key=lambda c: (manhattan(c, (SHED_HALF - 1, SHED_HALF - 1)), c[1], c[0]))

    pasture_index = sum(1 for p in state.tile_plan.values() if p.get("structure") == "PASTURE")
    allocation = desired_allocation(ctx, state)
    # Reserve the closest available tiles for COOP/PASTURE before assigning
    # crops.  This changes animal placement only; portfolio counts are intact.
    # After animals, put the highest-turnover crops nearest the shed.  The old
    # portfolio order put long-lived melons nearest and short-cycle wheat/
    # carrot at the remote NE edge, where harvest/replant travel caused the
    # chronic 47-49/50 occupancy plateau.
    service_order = {
        "COOP": 0, "PASTURE": 0,
        "CARROT": 1, "WHEAT": 2, "TOMATO": 3, "STRAWBERRY": 4, "MELON": 5,
    }
    allocation = sorted(allocation, key=lambda kv: service_order.get(kv[0], 9))
    for item, target in allocation:
        have = existing.get(item, 0) + planned_counts.get(item, 0)
        need = target - have
        while need > 0 and (animal_free if item in ("COOP", "PASTURE") else crop_free):
            pool = animal_free if item in ("COOP", "PASTURE") else crop_free
            coord = pool.pop(0)
            # A tile may have been consumed from the other ordering pool.
            if coord not in free:
                continue
            free.remove(coord)
            if item == "COOP":
                state.tile_plan[coord] = {"kind":"STRUCT", "structure":"COOP", "animal":"GOOSE"}
            elif item == "PASTURE":
                # P0 emphasizes sheep, P1 balances cow/sheep.
                frac = float(PARAMS["pasture_sheep_fraction"])
                animal = "SHEEP" if (pasture_index % 100) / 100.0 < frac else "COW"
                state.tile_plan[coord] = {"kind":"STRUCT", "structure":"PASTURE", "animal":animal}
                pasture_index += 1
            else:
                state.tile_plan[coord] = {"kind":"CROP", "crop":item}
            planned_counts[item] = planned_counts.get(item, 0) + 1
            need -= 1


# ---------------------------------------------------------------------------
# HIRING  — Fibonacci-priced HIRE orders, once per day at hour 0.
# ---------------------------------------------------------------------------

def decide_hiring(ctx, state):
    """Hire up to HANDS_PER_DAY workers at hour 0.

    V5 makes hiring a hard priority.  The same Python agent can be invoked for
    both players, so the guard is keyed by (player, day), not just day.
    """
    if ctx.hour != 0:
        return []

    key = (ctx.player, ctx.day)
    if state.last_day_hired.get(key, False):
        return []
    state.last_day_hired[key] = True

    orders = []
    cost = 0
    a, b = 1, 1

    for _ in range(HANDS_PER_DAY):
        price = a
        if ctx.money - cost < price:
            break
        orders.append(["HIRE"])
        cost += price
        a, b = b, a + b

    return orders


# ---------------------------------------------------------------------------
# STRUCTURE / ANIMAL BOOKKEEPING
# ---------------------------------------------------------------------------

def _struct_animal_counts(ctx, state):
    planned = {"COOP": 0, "PASTURE": 0}
    target_animals = {"GOOSE": 0, "COW": 0, "SHEEP": 0}
    for p in state.tile_plan.values():
        if p.get("kind") == "STRUCT":
            st = p.get("structure")
            planned[st] = planned.get(st, 0) + 1
            a = p.get("animal")
            if a in target_animals:
                target_animals[a] += 1

    built = {"COOP": 0, "PASTURE": 0}
    animals = {"GOOSE": 0, "COW": 0, "SHEEP": 0}
    for x, y in ctx.unlocked:
        t = get_tile(ctx.tiles, x, y)
        if isinstance(t, dict) and t.get("kind") in built:
            built[t["kind"]] += 1
            a = t.get("animal")
            if a in animals:
                animals[a] += 1
    for a in animals:
        animals[a] += int(_get(ctx.shed, a, 0))
    return {"planned":planned, "built":built, "animals":animals, "target_animals":target_animals}

def _wheat_feed_reserve(ctx, counts=None, state=None):
    """Wheat to hold in the shed as animal feed.  Front-loaded: sized to the
    animals we intend to own (planned + built structures), not just placed ones,
    so we stock feed before the animals even arrive."""
    if counts is None:
        counts = _struct_animal_counts(ctx, state)
    intended = sum(counts["planned"].get(s, 0) + counts["built"].get(s, 0)
                   for s in ("COOP", "PASTURE"))
    if intended <= 0:
        return 0
    return intended * int(PARAMS["feed_days"]) + 3


# ---------------------------------------------------------------------------
# MARKET ORDERS  — hires + buys + paced sells, capped at MAX_MARKET_ORDERS.
# ---------------------------------------------------------------------------

def decide_market_orders(ctx, state):
    """Economic controller: protect daily labor, unlock land before it is too
    late, buy only the next needed inputs, and liquidate aggressively at the end."""
    hires = decide_hiring(ctx, state)
    orders = list(hires)
    slots = MAX_MARKET_ORDERS - len(orders)
    money = ctx.money
    counts = _struct_animal_counts(ctx, state)
    feed_reserve = _wheat_feed_reserve(ctx, counts, state)

    # Measure productive utilization once per day. Weeds are gaps, not useful
    # occupancy. A two-day streak avoids expanding because of a single lucky
    # frame between harvest and replant operations.
    if state.utilization_day != ctx.day:
        productive = 0
        for x, y in ctx.unlocked:
            tile = get_tile(ctx.tiles, x, y)
            if isinstance(tile, dict) and tile.get("kind") in ("PLANT", "COOP", "PASTURE"):
                productive += 1
        ratio = productive / max(1, len(ctx.unlocked))
        state.near_full_days = state.near_full_days + 1 if ratio >= 0.94 else 0
        state.utilization_day = ctx.day

    # 1) Land is a capacity investment. Do not wait for the current quadrant to
    # become full: production lost while waiting is worth more than the tile cost.
    q = len(ctx.unlocked_quadrants)
    land_costs = [1000, 2000, 4000]
    land_ready = False
    if q < 4:
        cost = land_costs[q - 1]
        # Keep enough cash for a modest next-wave seed purchase.
        reserve = PARAMS["land_reserves"][q - 1]
        min_day = PARAMS["land_days"][q - 1]
        # Finish servicing the land we already own before buying another
        # quadrant.  Otherwise the new capacity creates permanent seed/build
        # backlogs and visible empty patches.
        current_land_ready = state.near_full_days >= 2
        season_time_ready = ctx.day <= int(PARAMS["land_last_day"])
        if (current_land_ready and season_time_ready
                and ctx.day >= min_day and money >= cost + reserve):
            land_ready = True
    if land_ready and slots > 0:
        orders.append(["BUY_LAND"])
        money -= land_costs[q - 1]
        slots -= 1

    # 2) Buy seeds for the currently unlocked empty/planned crop tiles.
    seed_need = {}
    for coord, p in state.tile_plan.items():
        if p.get("kind") == "CROP" and get_tile(ctx.tiles, *coord) is None:
            seed_need[p["crop"]] = seed_need.get(p["crop"], 0) + 1
    # Premium first because later planting has less time to mature.
    for crop in ("MELON", "STRAWBERRY", "TOMATO", "CARROT", "WHEAT"):
        if slots <= 0:
            break
        need = seed_need.get(crop, 0) - int(_get(ctx.seeds_remaining, crop, 0))
        if need <= 0:
            continue
        # Do not buy a seed that cannot reach a useful harvest before the
        # 30-day season ends. Existing seeds may still be planted; this only
        # prevents turning final-day cash into worthless new inventory.
        days_left_after_today = 29 - ctx.day
        if days_left_after_today < CROPS[crop]["max_day"]:
            continue
        unit = CROPS[crop]["seed"]
        affordable = int(max(0, money - WORKING_CASH) // unit)
        n = min(need, affordable)
        if n > 0:
            orders.append(["BUY_SEED", crop, n])
            money -= n * unit
            slots -= 1

    # 3) Buy animals only for currently BUILT empty structures. Planned capacity
    # is deliberately ignored: animals in the shed are dead capital.
    target = counts["target_animals"]
    current = counts["animals"]
    for animal in ("SHEEP", "COW", "GOOSE"):
        if slots <= 0:
            break
        structure = ANIMALS[animal]["structure"]
        occupied_for_structure = sum(
            1 for x, y in ctx.unlocked
            if isinstance(get_tile(ctx.tiles, x, y), dict)
            and get_tile(ctx.tiles, x, y).get("kind") == structure
            and get_tile(ctx.tiles, x, y).get("animal") is not None
        )
        animals_waiting = sum(int(_get(ctx.shed, a, 0)) for a, info in ANIMALS.items()
                              if info["structure"] == structure)
        built_free = max(0, counts["built"].get(structure, 0) - occupied_for_structure - animals_waiting)
        need = min(built_free, max(0, target.get(animal, 0) - current.get(animal, 0)))
        if need <= 0:
            continue
        days_left_after_today = 29 - ctx.day
        if days_left_after_today < {"GOOSE": 4, "COW": 8, "SHEEP": 6}[animal]:
            continue
        # One animal per turn prevents a huge opening cash sink.
        cost = ANIMALS[animal]["cost"]
        if money >= cost + WORKING_CASH:
            orders.append(["BUY_ANIMAL", animal, 1])
            money -= cost
            current[animal] += 1
            slots -= 1

    # 4) Feed. Only maintain the reserve actually needed by existing/near-term animals.
    if slots > 0:
        wheat_have = int(_get(ctx.shed, "WHEAT", 0))
        animals_now = sum(current.values())
        desired_feed = animals_now * 3 + 3
        if wheat_have < desired_feed:
            price = int(_get(ctx.prices, "WHEAT", 25))
            n = min(desired_feed - wheat_have, int(max(0, money - WORKING_CASH) // max(1, price)))
            if n > 0 and price <= PARAMS["wheat_buy_price"]:
                orders.append(["BUY_PRODUCT", "WHEAT", n])
                money -= n * price
                slots -= 1

    # 5) Fertilizer: buy small batches, then rely on animal fertilizer.
    if slots > 0 and ctx.day < 22:
        fert = int(_get(ctx.shed, "FERTILIZER", 0))
        price = int(_get(ctx.prices, "FERTILIZER", 100))
        if fert < 2 and money >= price + WORKING_CASH and price <= PARAMS["fertilizer_buy_price"]:
            orders.append(["BUY_PRODUCT", "FERTILIZER", 1])
            slots -= 1

    # 6) Selling. Endgame overrides price discipline because inventory has no final value.
    sells = decide_sells(ctx, state, feed_reserve, endgame=(ctx.day >= ENDGAME_DAY))
    orders.extend(sells[:slots])

    return orders[:MAX_MARKET_ORDERS]


# ---------------------------------------------------------------------------
# SELLING  — price-ratio-reactive volume via sell_qty().
# ---------------------------------------------------------------------------

def sell_qty(good, have, price, base, endgame=False):
    if have <= 0:
        return 0
    if endgame:
        return have
    ratio = price / max(1, base)
    curve = PARAMS["premium_sell"] if good in PREMIUM_GLUT else PARAMS["normal_sell"]
    for threshold, fraction in curve:
        if ratio >= threshold:
            return max(1, int(have * fraction)) if fraction > 0 else 0
    return 0


def decide_sells(ctx, state, feed_reserve, endgame=False):
    orders = []
    demand_goods = set()
    for shop in ctx.unlocked_shops:
        demand_goods.update(TOWN_DEMAND.get(shop, set()))

    goods = [g for g in ctx.shed if g in BASE_PRICES]
    # In normal play, sell demand-supported goods first. In endgame, highest-value
    # goods first so the market receives large but orderly liquidation batches.
    goods.sort(key=lambda g: (g not in demand_goods, -BASE_PRICES.get(g, 0)))

    for good in goods:
        have = int(_get(ctx.shed, good, 0))
        if good == "WHEAT" and not endgame:
            have = max(0, have - feed_reserve)
        if have <= 0:
            continue
        price = int(_get(ctx.prices, good, BASE_PRICES.get(good, 1)))
        n = sell_qty(good, have, price, BASE_PRICES.get(good, 1), endgame=endgame)
        overflow = int(PARAMS["shed_overflow_force"])
        if ctx.shed_total >= overflow and not endgame and good in GLUT_SENSITIVE:
            n = max(n, min(have, ctx.shed_total - overflow))
        if n > 0:
            orders.append(["SELL", good, n])
    return orders


TOWN_DEMAND = {
    "BAKERY": {"EGG", "WHEAT"},
    "PIZZA_SHOP": {"MILK", "TOMATO", "WHEAT"},
    "BRUNCH_SPOT": {"EGG", "WHEAT", "STRAWBERRY"},
    "YARN_STORE": {"WOOL"},
    "ICE_CREAM_SHOP": {"STRAWBERRY", "MILK", "WHEAT"},
    "PET_CAFE": {"CARROT"},
    "SMOOTHIE_SHOP": {"STRAWBERRY", "MILK"},
    "FARMERS_MARKET": {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY"},
}


# ---------------------------------------------------------------------------
# UNIT DISPATCH  — one action per farmer / hand, priority checklist.
# ---------------------------------------------------------------------------

def decide_unit_action(ctx, state, idx):
    """Return exactly one action for unit `idx` (0 = farmer, 1+ = hands)."""
    pos = ctx.unit_pos(idx)
    inv = ctx.unit_inv(idx)
    tile = get_tile(ctx.tiles, pos[0], pos[1])
    on_shed = pos in SHED_TILES

    # === 1) Farm-wide emergency dispatch.  Exactly one nearest free unit claims
    # each unfed animal before anyone gets distracted by its current tile.
    urgent = _nearest_unclaimed(ctx, pos, ctx.unfed_tiles)
    if urgent is not None:
        if urgent == pos and int(_get(inv, "WHEAT", 0)) > 0:
            ctx.claimed.add(urgent)
            return ["FEED"]
        if int(_get(inv, "WHEAT", 0)) > 0:
            ctx.claimed.add(urgent)
            return move_toward(pos, urgent)
        if on_shed and int(_get(ctx.shed_ledger, "WHEAT", 0)) > 0:
            ctx.shed_ledger["WHEAT"] -= 1
            return ["PICKUP", "WHEAT", 1]
        shed = nearest_shed_tile(pos, ctx.unlocked)
        if pos != shed:
            return move_toward(pos, shed)

    # === 2) Act on the tile we're standing on, if there's something to do =====
    act = _on_tile_action(ctx, state, idx, pos, tile, inv)
    if act is not None:
        return act

    # === 3) At/near the shed with nothing to do here: pick up something useful
    if on_shed:
        pk = _shed_pickup_action(ctx, state, idx, inv)
        if pk is not None:
            return pk

    # === 4) Move toward the nearest highest-priority pending task =============
    mv = _move_to_task(ctx, state, idx, pos, inv)
    if mv is not None:
        return mv

    # === 5) Carry produce back to the shed to enable selling =================
    carry_produce = sum(int(v) for k, v in inv.items()
                         if k not in ANIMALS and isinstance(v, (int, float)))
    if carry_produce > 0:
        if on_shed:
            return ["DROP"]
        return move_toward(pos, nearest_shed_tile(pos, ctx.unlocked))

    # === 6) Nothing to do ====================================================
    return ["PASS"]


def _on_tile_action(ctx, state, idx, pos, tile, inv):
    """The core priority checklist for the tile a unit occupies."""
    day = ctx.day

    if isinstance(tile, dict):
        kind = tile.get("kind")

        # ---- PLANT tiles ----
        if kind == "PLANT":
            crop = tile.get("crop")
            info = CROPS.get(crop)
            if info is not None:
                if crop_ready(tile, info, day, ctx.step):
                    ctx.claimed.add(pos)
                    return ["HARVEST"]
                if not tile.get("watered_today", False):
                    ctx.claimed.add(pos)
                    return ["WATER"]
                if fert_eligible(tile, info, day) and int(_get(inv, "FERTILIZER", 0)) > 0:
                    ctx.claimed.add(pos)
                    return ["FERTILIZE"]
            return None

        # ---- WEED ----
        if kind == "WEED":
            ctx.claimed.add(pos)
            return ["DIG"]

        # ---- Animal structures ----
        if kind in ("COOP", "PASTURE"):
            animal = tile.get("animal")
            if animal is None:
                for aname, ainfo in ANIMALS.items():
                    if ainfo["structure"] == kind and int(_get(inv, aname, 0)) > 0:
                        ctx.claimed.add(pos)
                        return ["PLACE", aname]
                return None
            if not tile.get("fed_today", False):
                if int(_get(inv, "WHEAT", 0)) > 0:
                    ctx.claimed.add(pos)
                    return ["FEED"]
                return None  # no wheat to feed with (reserve should prevent this)
            if not tile.get("cared_today", False):
                ctx.claimed.add(pos)
                return ["CARE"]
            if tile.get("fertilizer_available", False):
                ctx.claimed.add(pos)
                return ["COLLECT_FERTILIZER"]
            if int(_get(tile, "yield_units", 0)) > 0:
                ctx.claimed.add(pos)
                return ["HARVEST"]
            return None

        return None

    # ---- Empty unlocked tile: build a planned structure, or plant a seed ----
    if tile is None:
        plan = state.tile_plan.get(pos)
        if plan is not None:
            if plan["kind"] == "STRUCT":
                if plan["structure"] == "COOP":
                    ctx.claimed.add(pos)
                    return ["BUILD_COOP"]
                if plan["structure"] == "PASTURE":
                    ctx.claimed.add(pos)
                    return ["BUILD_PASTURE"]
            elif plan["kind"] == "CROP":
                crop = plan["crop"]
                # Only plant if we hold the seed AND there is time left today for a
                # unit to water it (fresh plants weed at end of day if left dry).
                if int(_get(ctx.seeds_remaining, crop, 0)) > 0 and ctx.hour < ctx.tpd - 2:
                    ctx.seeds_remaining[crop] = int(ctx.seeds_remaining[crop]) - 1
                    ctx.claimed.add(pos)
                    return ["PLANT", crop]
        return None

    return None  # "LOCKED" or unknown -> can't act here


def _shed_pickup_action(ctx, state, idx, inv):
    """When standing at the shed with nothing better to do: grab an animal that
    needs placing, or fertilizer for a pending high-value crop."""
    if ctx.unfed_tiles and int(_get(ctx.shed_ledger, "WHEAT", 0)) > 0:
        ctx.shed_ledger["WHEAT"] -= 1
        return ["PICKUP", "WHEAT", 1]

    for (x, y, kind, tile) in ctx.empty_structs:
        for aname, ainfo in ANIMALS.items():
            if ainfo["structure"] == kind:
                if int(_get(ctx.shed_ledger, aname, 0)) > 0 and int(_get(inv, aname, 0)) == 0:
                    ctx.shed_ledger[aname] = int(ctx.shed_ledger.get(aname, 0)) - 1
                    return ["PICKUP", aname, 1]

    if ctx.fertilize_tiles:
        if int(_get(ctx.shed_ledger, "FERTILIZER", 0)) > 0 and int(_get(inv, "FERTILIZER", 0)) == 0:
            ctx.shed_ledger["FERTILIZER"] = int(ctx.shed_ledger.get("FERTILIZER", 0)) - 1
            return ["PICKUP", "FERTILIZER", 1]
    return None


def _nearest_unclaimed(ctx, pos, tiles):
    candidates = [t for t in tiles if t not in ctx.claimed]
    return min(candidates, key=lambda t: manhattan(pos, t)) if candidates else None


def _move_to_task(ctx, state, idx, pos, inv):
    """V7 production scheduler.

    V6 had a structural starvation bug: every unit used the same priority
    (harvest -> water -> animals -> build -> plant). Once a few plants existed,
    five/eight workers could spend nearly all turns maintaining the existing
    field while newly unlocked quadrants remained empty.

    V7 makes expansion/planting a first-class workload.  Two workers are
    expansion specialists, three are crop-maintenance specialists, and the
    remaining workers handle animals/logistics.  The priorities also become
    more expansion-heavy whenever a large number of planned tiles are still
    empty.
    """
    carrying_fert = int(_get(inv, "FERTILIZER", 0)) > 0
    carrying_animal = any(int(_get(inv, a, 0)) > 0 for a in ANIMALS)

    plant_tiles = []
    if ctx.hour < ctx.tpd - 2:
        for coord, p in state.tile_plan.items():
            if p.get("kind") == "CROP" and int(_get(ctx.seeds_remaining, p["crop"], 0)) > 0:
                if get_tile(ctx.tiles, coord[0], coord[1]) is None:
                    plant_tiles.append(coord)

    build_tiles = []
    for coord, p in state.tile_plan.items():
        if p.get("kind") == "STRUCT" and get_tile(ctx.tiles, coord[0], coord[1]) is None:
            build_tiles.append(coord)

    place_tiles = [(x, y) for (x, y, k, t) in ctx.empty_structs]

    # How much of the planned farm is still physically empty?
    expansion_backlog = sum(
        1 for coord, plan in state.tile_plan.items()
        if get_tile(ctx.tiles, coord[0], coord[1]) is None
    )

    # idx 0 = farmer. Hands 1-2 are expansion workers, 3-5 are crop workers,
    # and 6+ are animal/logistics workers.
    if idx == 0:
        role = "farmer"
    elif idx <= 2:
        role = "expansion"
    elif idx <= 5:
        role = "crop"
    else:
        role = "animal"

    if expansion_backlog >= 8 and ctx.day < 25:
        if role == "expansion":
            categories = [
                ("build", build_tiles, True),
                ("plant", plant_tiles, True),
                ("harvest", ctx.harvest_tiles, True),
                ("water", ctx.water_tiles, True),
                ("animal", ctx.animal_tiles, True),
                ("place", place_tiles, carrying_animal),
                ("fert", ctx.fertilize_tiles, carrying_fert),
                ("weed", ctx.weed_tiles, True),
            ]
        elif role == "crop":
            categories = [
                ("harvest", ctx.harvest_tiles, True),
                ("water", ctx.water_tiles, True),
                ("plant", plant_tiles, True),
                ("build", build_tiles, True),
                ("fert", ctx.fertilize_tiles, carrying_fert),
                ("weed", ctx.weed_tiles, True),
                ("animal", ctx.animal_tiles, True),
                ("place", place_tiles, carrying_animal),
            ]
        elif role == "animal":
            categories = [
                ("animal", ctx.animal_tiles, True),
                ("place", place_tiles, carrying_animal),
                ("harvest", ctx.harvest_tiles, True),
                ("water", ctx.water_tiles, True),
                ("fert", ctx.fertilize_tiles, carrying_fert),
                ("plant", plant_tiles, True),
                ("build", build_tiles, True),
                ("weed", ctx.weed_tiles, True),
            ]
        else:
            categories = [
                ("harvest", ctx.harvest_tiles, True),
                ("animal", ctx.animal_tiles, True),
                ("place", place_tiles, carrying_animal),
                ("water", ctx.water_tiles, True),
                ("plant", plant_tiles, True),
                ("build", build_tiles, True),
                ("fert", ctx.fertilize_tiles, carrying_fert),
                ("weed", ctx.weed_tiles, True),
            ]
    else:
        # Once most of the farm is established, switch capacity toward
        # maintenance and harvesting.
        if role == "animal":
            categories = [
                ("animal", ctx.animal_tiles, True),
                ("place", place_tiles, carrying_animal),
                ("harvest", ctx.harvest_tiles, True),
                ("water", ctx.water_tiles, True),
                ("fert", ctx.fertilize_tiles, carrying_fert),
                ("plant", plant_tiles, True),
                ("build", build_tiles, True),
                ("weed", ctx.weed_tiles, True),
            ]
        elif role == "expansion":
            categories = [
                ("plant", plant_tiles, True),
                ("build", build_tiles, True),
                ("harvest", ctx.harvest_tiles, True),
                ("water", ctx.water_tiles, True),
                ("fert", ctx.fertilize_tiles, carrying_fert),
                ("animal", ctx.animal_tiles, True),
                ("place", place_tiles, carrying_animal),
                ("weed", ctx.weed_tiles, True),
            ]
        else:
            categories = [
                ("harvest", ctx.harvest_tiles, True),
                ("water", ctx.water_tiles, True),
                ("plant", plant_tiles, True),
                ("animal", ctx.animal_tiles, True),
                ("fert", ctx.fertilize_tiles, carrying_fert),
                ("build", build_tiles, True),
                ("place", place_tiles, carrying_animal),
                ("weed", ctx.weed_tiles, True),
            ]

    for _name, tiles, gate in categories:
        if not gate:
            continue
        best = None
        best_key = None
        for t in tiles:
            if t in ctx.claimed or t == pos:
                continue
            d = manhattan(pos, t)
            if role == "expansion" and _name in ("plant", "build"):
                # Reserve expansion hands for the remote backlog. Without this,
                # freshly emptied near tiles continually win nearest-task
                # selection and the outer NE row is starved despite seeds being
                # available in the shared seed inventory.
                shed_distance = min(manhattan(t, s) for s in SHED_TILES)
                key = (-shed_distance, d, -_tile_value(ctx, t, state))
            else:
                key = (d, -_tile_value(ctx, t, state))
            if best_key is None or key < best_key:
                best_key = key
                best = t
        if best is not None:
            ctx.claimed.add(best)
            return move_toward(pos, best)

    return None


def _tile_value(ctx, coord, state):
    """Rough value score so units prefer premium crops/animals over staples."""
    t = get_tile(ctx.tiles, coord[0], coord[1])
    if isinstance(t, dict):
        if t.get("kind") == "PLANT":
            return BASE_PRICES.get(t.get("crop"), 0)
        if t.get("kind") in ("COOP", "PASTURE"):
            a = t.get("animal")
            if a in ANIMALS:
                return ANIMALS[a]["base"] + 100  # animals are ongoing -> value them
            return 40
    plan = state.tile_plan.get(coord)
    if plan and plan.get("kind") == "CROP":
        return BASE_PRICES.get(plan.get("crop"), 0)
    return 0


# ---------------------------------------------------------------------------
# TOP-LEVEL AGENT  — wrapped so any bug degrades to a safe PASS, never a crash.
# ---------------------------------------------------------------------------

def agent(obs):
    try:
        _reset_if_new_episode(obs)
        ctx = TurnContext(obs)
        state = _state_for_player(ctx.player)

        global TURNS_PER_DAY
        if ctx.tpd and ctx.tpd != TURNS_PER_DAY:
            TURNS_PER_DAY = ctx.tpd

        update_phase(ctx, state=state)
        assign_plans(ctx, state)

        market = decide_market_orders(ctx, state)

        farmer_action = decide_unit_action(ctx, state, 0)
        hand_actions = [decide_unit_action(ctx, state, i + 1) for i in range(len(ctx.hands))]

        return {
            "farmer": farmer_action,
            "hands": hand_actions,
            "market": market,
        }
    except Exception:
        # Defensive fallback: never throw (a runtime error fails the episode).
        n_hands = 0
        try:
            farms = obs.get("farms", []) or []
            p = int(obs.get("player", 0))
            if p < len(farms):
                n_hands = len(farms[p].get("hands", []) or [])
        except Exception:
            n_hands = 0
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in range(n_hands)],
            "market": [],
        }


# ---------------------------------------------------------------------------
# LOCAL TEST HARNESS  — runs only when executed directly, not on Kaggle import.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        from kaggle_environments import make
    except Exception as e:
        print("kaggle_environments not installed:", e)
        raise SystemExit(1)

    STATES[0].reset()
    STATES[1].reset()

    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
    env.run([agent, agent])

    print("\nDaily checkpoints (Agent vs Agent):")
    for step_no, step in enumerate(env.steps):
        if not isinstance(step, list) or step_no % 24 != 1:
            continue
        source = step[0] if step and isinstance(step[0], dict) else {}
        obs = source.get("observation", {}) if isinstance(source, dict) else {}
        farms = obs.get("farms", []) or []
        for player_idx, farm in enumerate(farms[:2]):
            tiles = farm.get("tiles", []) or []
            unlocked = 0
            occupied = 0
            for row in tiles:
                for tile in row:
                    if tile != "LOCKED":
                        unlocked += 1
                        if tile is not None:
                            occupied += 1
            print(
                f"DAY {step_no // 24:02d} | PLAYER {player_idx + 1} | "
                f"money={farm.get('money')} | "
                f"hands={len(farm.get('hands', []) or [])} | "
                f"land={farm.get('unlocked_quadrants')} | "
                f"occupied={occupied}/{unlocked}"
            )

    final = env.steps[-1]
    print("\nFINAL RESULT")
    for i, s in enumerate(final):
        if isinstance(s, dict):
            print(f"PLAYER {i + 1} (AGENT): reward={s.get('reward')} status={s.get('status')}")

    with open("replay.json", "w", encoding="utf-8") as f:
        json.dump(env.toJSON(), f)

    print("Replay saved: replay.json")