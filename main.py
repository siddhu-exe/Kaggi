"""
Kaggriculture competitive agent — main.py
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

Architecture (see Docs/GAME_MODEL.md for the full write-up):
    * GameState            module-global object, persists across turns.
    * TurnContext          per-turn parsed view of the observation + task lists.
    * decide_unit_action   one action per farmer / hand (priority checklist).
    * decide_market_orders BUY_SEED / BUY_ANIMAL / BUY_PRODUCT / SELL / BUY_LAND.
    * decide_hiring        Fibonacci-priced HIRE orders at day start (5/day).
    * agent(obs)           top-level dispatcher, wrapped so it NEVER throws.

Key spec facts baked in (verified against README.md / AGENTS.md):
    * tiles are indexed tiles[y][x]; positions are [x, y].
    * SELL pulls from private["shed"]; HARVEST/PICKUP fill a unit's *inventory*.
      Produce reaches the shed via DROP (on a center tile) or the automatic
      end-of-day inventory drop.  So selling naturally lags harvest by ~a day.
    * A freshly planted seed has consecutive_unwatered = 1 -> it MUST be watered
      the same day or it weeds that night.  We only plant when there is time left
      in the day for a unit to water it.
    * FEED is assumed to consume wheat from the shed (documented assumption; we
      buy market wheat as feed stock so animals can be bought early).
    * Glut-sensitive goods (carrot, strawberry, melon, milk, wool) crash toward
      the $1 floor -> steep sell throttle.  Wheat / eggs absorb gluts -> shallow
      throttle, sell freely (keeping a wheat feed reserve).
"""

import math

# ---------------------------------------------------------------------------
# CONSTANTS  (tune these — every phase threshold is marked with a PHASE comment)
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
# Goods that absorb gluts gently — sell freely (wheat kept above feed reserve).
GLUT_TOLERANT = {"WHEAT", "EGG"}

# Crops worth spending fertilizer on (high value / good return on the bonus).
FERTILIZE_CROPS = {"MELON", "STRAWBERRY", "TOMATO", "CARROT"}

# Price-reactive selling.  Every turn we compute ratio = price / base and scale
# sell volume between `throttle_floor` (sell nothing below it) and 1.0 (sell all
# at/above base).  Glut-tolerant staples (wheat/eggs) have a LOW floor so we sell
# them freely even when cheap; premium goods have a HIGH floor so we hold and wait
# for price recovery instead of dumping into a crash.
SELL_THROTTLE = {
    "WHEAT":      {"floor": 0.25},   # absorbs gluts — sell surplus above feed reserve
    "EGG":        {"floor": 0.25},
    "TOMATO":     {"floor": 0.40},
    "CARROT":     {"floor": 0.45},   # crashes on gluts
    "STRAWBERRY": {"floor": 0.55},   # premium — crashes hard
    "MELON":      {"floor": 0.55},
    "MILK":       {"floor": 0.55},
    "WOOL":       {"floor": 0.55},
    "FERTILIZER": {"floor": 0.35},
}

SHED_CAPACITY = 100
SHED_OVERFLOW_FORCE = 85    # when shed non-seed items exceed this, dump paced goods
MAX_MARKET_ORDERS = 10

# Tiny working-cash buffer so we never fail mid-order — we otherwise spend it all.
WORKING_CASH = 50

# Hands to hire every day.  5 hands ~= $1+1+2+3+5 = $12/day under Fibonacci — the
# labor is near-free, so there is no reason to hold back.
HANDS_PER_DAY = 5

# Single aggressive day-0 allocation for the opening (25-tile NW quadrant).  No
# phases: this is the target from turn 1.  Ongoing crops (tomato/strawberry) and
# animals are represented immediately so they start their long maturation early.
BASE_ALLOCATION = [
    ("WHEAT", 8),        # staple cash + animal-feed backstop
    ("TOMATO", 4),       # ongoing daily production
    ("MELON", 3),        # premium one-time
    ("CARROT", 3),       # fast staple
    ("STRAWBERRY", 2),   # premium ongoing
    ("COOP", 1),         # goose
    ("PASTURE", 1),      # cow / sheep
]


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
        self.last_day_hired = -1     # guard: hire only once per day
        # Day-0 front-load bookkeeping (see decide_market_orders).
        self.animals_queued = set()  # animals we committed to buy on day 0
        self.feed_bought = False     # bought the initial market-wheat feed stock
        self.notes = {}              # scratch for debugging / future tuning


STATE = GameState()


def _reset_if_new_episode(obs):
    """Re-init globals at the very start of an episode (step 0)."""
    try:
        if int(obs.get("step", 0)) == 0 and int(obs.get("day", 0)) == 0:
            STATE.reset()
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
        self.claimed = set()                      # tile coords already handled/targeted

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
        self.empty_structs = []       # (x,y, structure, animal_dict) coop/pasture w/o animal
        self.animal_tiles = []        # (x,y) occupied animal structures needing attention
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
                    # Needs attention if unfed / uncared / has produce or fertilizer.
                    unfed = not tile.get("fed_today", False)
                    uncared = tile.get("fed_today", False) and not tile.get("cared_today", False)
                    has_fert = tile.get("fertilizer_available", False)
                    has_yield = int(_get(tile, "yield_units", 0)) > 0
                    if unfed or uncared or has_fert or has_yield:
                        self.animal_tiles.append((x, y))

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
# PHASE MANAGEMENT
# ---------------------------------------------------------------------------

def update_phase(ctx, state):
    """No-op.  The phased strategy was replaced by day-0 capital deployment (see
    module docstring).  Kept as a stub so the agent() dispatcher is untouched."""
    return


# ---------------------------------------------------------------------------
# TILE PLANNING  — decide what each empty tile *should* become.
# ---------------------------------------------------------------------------

def desired_allocation(ctx, state):
    """Ordered [(item, target_count)] the farm wants — a single aggressive plan
    used from day 0 (no phases).  Filled greedily against available tiles, so
    earlier entries win the land.  Scales up as land is unlocked."""
    plan = list(BASE_ALLOCATION)

    # More land unlocked -> scale staple + ongoing crops to fill it (each extra
    # quadrant is 25 more tiles).  Premium/animals grow more slowly on purpose.
    extra_quads = max(0, len(ctx.unlocked_quadrants) - 1)
    if extra_quads:
        plan += [("WHEAT", 6 * extra_quads), ("TOMATO", 4 * extra_quads),
                 ("MELON", 3 * extra_quads), ("STRAWBERRY", 3 * extra_quads),
                 ("CARROT", 2 * extra_quads), ("PASTURE", 1)]

    # Opponent diversification: their farm is public.  Back off glut-sensitive
    # crops they're flooding (shared price curve), lean into ones they've left open.
    opp = _opp_crop_counts(ctx)
    adj = []
    for item, cnt in plan:
        if item in GLUT_SENSITIVE:
            n = opp.get(item, 0)
            if n >= 4:
                cnt = max(1, cnt // 2)      # they're flooding it -> avoid the price war
            elif n == 0:
                cnt += 1                    # open lane -> lean in slightly
        adj.append((item, cnt))
    return adj


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
    """Persist stable per-tile plans: keep existing plans on still-empty tiles,
    drop plans on tiles that got filled/locked, and add plans to cover deficits."""
    # 1) Prune plans whose tile is no longer an empty unlocked tile.
    for coord in list(state.tile_plan.keys()):
        if coord not in ctx.unlocked:
            del state.tile_plan[coord]
            continue
        if get_tile(ctx.tiles, coord[0], coord[1]) is not None:
            # Tile became a plant / structure / weed — plan fulfilled or obstructed.
            del state.tile_plan[coord]

    # 2) Count what already exists or is already planned.
    planned_counts = {}
    for p in state.tile_plan.values():
        key = p["crop"] if p["kind"] == "CROP" else p["structure"]
        planned_counts[key] = planned_counts.get(key, 0) + 1

    existing = dict(ctx.growing_counts)
    # Count existing structures too.
    struct_counts = {"COOP": 0, "PASTURE": 0}
    for (x, y) in ctx.unlocked:
        t = get_tile(ctx.tiles, x, y)
        if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE"):
            struct_counts[t["kind"]] += 1
    existing.update(struct_counts)

    # 3) Candidate empty tiles with no plan yet, nearest-to-shed first (compact).
    free = [c for c in ctx.empty_tiles if c not in state.tile_plan]
    free.sort(key=lambda c: manhattan(c, (SHED_HALF - 1, SHED_HALF - 1)))

    # 4) Fill deficits in priority order.
    for item, target in desired_allocation(ctx, state):
        have = existing.get(item, 0) + planned_counts.get(item, 0)
        need = target - have
        while need > 0 and free:
            coord = free.pop(0)
            if item in ("COOP", "PASTURE"):
                animal = "GOOSE" if item == "COOP" else None  # cow/sheep chosen at buy time
                state.tile_plan[coord] = {"kind": "STRUCT", "structure": item, "animal": animal}
            else:
                state.tile_plan[coord] = {"kind": "CROP", "crop": item}
            planned_counts[item] = planned_counts.get(item, 0) + 1
            need -= 1


# ---------------------------------------------------------------------------
# HIRING  — Fibonacci-priced HIRE orders, once per day at hour 0.
# ---------------------------------------------------------------------------

def decide_hiring(ctx, state):
    """Hire HANDS_PER_DAY hands every day at hour 0.  Fibonacci pricing makes this
    near-free (5 hands ~= $12), so labor is deployed to the max from day 0."""
    if ctx.hour != 0:
        return []
    if state.last_day_hired == ctx.day:
        return []
    state.last_day_hired = ctx.day

    cost = 0
    a, b = 1, 1
    orders = []
    for _ in range(HANDS_PER_DAY):
        fib = a
        if ctx.money - (cost + fib) < 0:
            break  # can't afford the next hire this early — stop
        cost += fib
        orders.append(["HIRE"])
        a, b = b, a + b
    return orders


# ---------------------------------------------------------------------------
# MARKET ORDERS  — hires + buys + paced sells, capped at MAX_MARKET_ORDERS.
# ---------------------------------------------------------------------------

def decide_market_orders(ctx, state):
    """Front-loaded buys + price-reactive sells, capped at MAX_MARKET_ORDERS.
    The whole point: deploy nearly the full $3000 on day 0."""
    orders = []
    orders.extend(decide_hiring(ctx, state))

    money = ctx.money
    counts = _struct_animal_counts(ctx)          # how many structures / animals exist
    feed_reserve = _wheat_feed_reserve(ctx, counts)

    # ---- 1) BUY_ANIMAL (front-loaded: buy as soon as a coop/pasture is planned) ----
    #        Animals take days to reach first production, so every day of delay is a
    #        lost cycle.  We buy the animal now; a unit builds the structure and
    #        PLACEs it.  We buy the market wheat feed (step 3) so it never starves.
    for want, struct in (("GOOSE", "COOP"), ("COW", "PASTURE")):
        desired = counts["planned"].get(struct, 0) + counts["built"].get(struct, 0)
        have = counts["animals"].get(want, 0)   # in shed + placed
        need = desired - have
        cost = ANIMALS[want]["cost"]
        # Buy at most one of each per turn (keeps cash flow smooth); WORKING_CASH
        # buffer only — we intentionally spend almost everything.
        if need > 0 and money - cost >= WORKING_CASH:
            orders.append(["BUY_ANIMAL", want, 1])
            money -= cost

    # ---- 2) BUY_SEED across ALL crop types to fill every planned tile ----
    seed_need = {}
    for coord, p in state.tile_plan.items():
        if p.get("kind") == "CROP":
            seed_need[p["crop"]] = seed_need.get(p["crop"], 0) + 1
    # Cheap staples first so they always land; then premium (still bought day 0).
    for crop in ("WHEAT", "CARROT", "TOMATO", "MELON", "STRAWBERRY"):
        buy = seed_need.get(crop, 0) - int(_get(ctx.seeds_remaining, crop, 0))
        if buy <= 0:
            continue
        unit = CROPS[crop]["seed"]
        affordable = int(max(0, (money - WORKING_CASH)) // unit)
        n = min(buy, affordable)
        if n > 0:
            orders.append(["BUY_SEED", crop, n])
            money -= n * unit

    # ---- 3) BUY_PRODUCT WHEAT as feed stock (don't wait to grow our own) ----
    #        Enables buying animals early: we top the shed to the feed reserve.
    wheat_have = int(_get(ctx.shed, "WHEAT", 0))
    total_animals_planned = sum(counts["planned"].get(s, 0) + counts["built"].get(s, 0)
                                for s in ("COOP", "PASTURE"))
    if total_animals_planned > 0 and wheat_have < feed_reserve:
        wprice = int(_get(ctx.prices, "WHEAT", BASE_PRICES["WHEAT"]))
        if wprice <= BASE_PRICES["WHEAT"] * 1.8 and money > WORKING_CASH:
            n = min(feed_reserve - wheat_have,
                    int(max(0, (money - WORKING_CASH)) // max(1, wprice)))
            if n > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", n])
                money -= n * wprice
                state.feed_bought = True

    # ---- 4) BUY_PRODUCT FERTILIZER (small stock for high-value crops) ----
    fert_have = int(_get(ctx.shed, "FERTILIZER", 0))
    if fert_have < 3 and money >= 600:
        fprice = int(_get(ctx.prices, "FERTILIZER", BASE_PRICES["FERTILIZER"]))
        n = min(3 - fert_have, 2)
        if fprice <= BASE_PRICES["FERTILIZER"] * 1.5 and money - n * fprice > WORKING_CASH:
            orders.append(["BUY_PRODUCT", "FERTILIZER", n])
            money -= n * fprice

    # ---- 5) BUY_LAND (ungated: buy when NW is nearly full AND cash is healthy) ----
    extras = max(0, len(ctx.unlocked_quadrants) - 1)
    land_costs = [1000, 2000, 4000]
    if extras < len(land_costs):
        land_cost = land_costs[extras]
        nearly_full = len(ctx.empty_tiles) <= 3
        # Keep a working buffer beyond the land cost so we can seed the new land.
        if nearly_full and money >= land_cost + 800:
            orders.append(["BUY_LAND"])
            money -= land_cost

    # ---- 6) SELL (price-reactive volume; wheat feed reserve carved out) ----
    orders.extend(decide_sells(ctx, state, feed_reserve))

    # Respect the per-turn order cap (extras are silently dropped by the engine).
    return orders[:MAX_MARKET_ORDERS]


def _struct_animal_counts(ctx):
    """Count planned vs built structures and owned (shed + placed) animals so we
    can front-load animal purchases without re-buying after placement."""
    planned = {"COOP": 0, "PASTURE": 0}
    for p in STATE.tile_plan.values():
        if p.get("kind") == "STRUCT":
            planned[p["structure"]] = planned.get(p["structure"], 0) + 1

    built = {"COOP": 0, "PASTURE": 0}
    animals = {"GOOSE": 0, "COW": 0, "SHEEP": 0}
    for (x, y) in ctx.unlocked:
        t = get_tile(ctx.tiles, x, y)
        if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE"):
            built[t["kind"]] += 1
            a = t.get("animal")
            if a in animals:
                animals[a] += 1                 # placed animals
    for a in animals:                            # animals sitting in the shed
        animals[a] += int(_get(ctx.shed, a, 0))
    return {"planned": planned, "built": built, "animals": animals}


def _wheat_feed_reserve(ctx, counts=None):
    """Wheat to hold in the shed as animal feed.  Front-loaded: sized to the
    animals we intend to own (planned + built structures), not just placed ones,
    so we stock feed before the animals even arrive."""
    if counts is None:
        counts = _struct_animal_counts(ctx)
    intended = sum(counts["planned"].get(s, 0) + counts["built"].get(s, 0)
                   for s in ("COOP", "PASTURE"))
    if intended <= 0:
        return 0
    return intended * 4 + 3   # ~4 days' feed buffer per animal + slack


def decide_sells(ctx, state, feed_reserve):
    """Price-reactive SELL orders.  Volume scales with the live price relative to
    base: dump near/above base, throttle hard toward the $1 floor.  Because BOTH
    players sell into the same shared price curve, reacting to the real price is
    the correct response to 'the market moved' — no fixed cap schedule."""
    sells = []
    shed = ctx.shed
    prices = ctx.prices
    overflow = ctx.shed_total > SHED_OVERFLOW_FORCE

    # Bias order: town-demanded goods first, then glut-tolerant, then higher price.
    demanded = _town_demanded_goods(ctx)
    sellable = [g for g in SELL_THROTTLE if int(_get(shed, g, 0)) > 0]

    def sort_key(g):
        return (0 if g in demanded else 1,
                0 if g in GLUT_TOLERANT else 1,
                -int(_get(prices, g, 0)))
    sellable.sort(key=sort_key)

    for good in sellable:
        have = int(_get(shed, good, 0))
        # Carve out the wheat feed reserve before selling any wheat.
        if good == "WHEAT":
            have = max(0, have - feed_reserve)
        if have <= 0:
            continue
        n = _sell_quantity(good, have, int(_get(prices, good, BASE_PRICES.get(good, 1))),
                           overflow)
        if n > 0:
            sells.append(["SELL", good, n])

    return sells


def _sell_quantity(good, have, price, overflow):
    """Price-reactive sell volume for `have` units of `good` at the live `price`.
    Scales linearly between the per-good throttle floor and base price; dumps at/
    above base or on overflow; holds near the floor.  Premium (glut-sensitive)
    goods are clamped to <= half the holding per turn so one order can't tank the
    shared price curve."""
    base = BASE_PRICES.get(good, 1) or 1
    ratio = price / base
    floor = SELL_THROTTLE.get(good, {"floor": 0.4})["floor"]

    if overflow:
        n = have                                     # forced dump: shed near cap
    elif ratio < floor:
        n = 0                                         # too cheap -> hold, wait
    elif ratio >= 1.0:
        n = have                                      # at/above base -> sell all
    else:
        frac = (ratio - floor) / (1.0 - floor)        # 0..1 linear ramp
        n = max(1, int(have * frac))

    if good in GLUT_SENSITIVE and not overflow:
        n = min(n, max(2, have // 2))
    return n


def _town_demanded_goods(ctx):
    """Approximate the set of goods currently demanded by unlocked town shops."""
    shop_demand = {
        "BAKERY": {"EGG", "WHEAT"},
        "PIZZA_SHOP": {"MILK", "TOMATO", "WHEAT"},
        "BRUNCH_SPOT": {"EGG", "WHEAT", "STRAWBERRY"},
        "YARN_STORE": {"WOOL"},
        "ICE_CREAM_SHOP": {"STRAWBERRY", "MILK", "WHEAT"},
        "PET_CAFE": {"CARROT"},
        "SMOOTHIE_SHOP": {"STRAWBERRY", "MILK"},
        "FARMERS_MARKET": {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY"},
    }
    demanded = set()
    for shop in ctx.unlocked_shops:
        key = str(shop).upper().replace(" ", "_")
        demanded |= shop_demand.get(key, set())
    return demanded


# ---------------------------------------------------------------------------
# PER-UNIT ACTION  — priority checklist, first match wins, one action returned.
# ---------------------------------------------------------------------------

def decide_unit_action(ctx, state, idx):
    """Return a single op list for farmer (idx 0) or hand (idx>=1)."""
    pos = ctx.unit_pos(idx)
    x, y = pos
    tile = get_tile(ctx.tiles, x, y)
    inv = ctx.unit_inv(idx)
    on_shed = pos in SHED_TILES

    # === 1) Act on the tile we're standing on (survival + income first) =======
    action = _on_tile_action(ctx, state, idx, pos, tile, inv)
    if action is not None:
        return action

    # === 2) If shed-adjacent and idle, pick up an animal / fertilizer we need ==
    if on_shed:
        pk = _shed_pickup_action(ctx, state, idx, inv)
        if pk is not None:
            return pk

    # === 3) Move toward the nearest highest-priority pending task =============
    mv = _move_to_task(ctx, state, idx, pos, inv)
    if mv is not None:
        return mv

    # === 4) Carry produce back to the shed to enable selling =================
    carry_produce = sum(int(v) for k, v in inv.items()
                        if k not in ANIMALS and isinstance(v, (int, float)))
    if carry_produce > 0:
        if on_shed:
            return ["DROP"]
        return move_toward(pos, nearest_shed_tile(pos, ctx.unlocked))

    # === 5) Nothing to do ====================================================
    return ["PASS"]


def _on_tile_action(ctx, state, idx, pos, tile, inv):
    """The core priority checklist for the tile a unit occupies."""
    x, y = pos
    day = ctx.day

    if isinstance(tile, dict):
        kind = tile.get("kind")

        # ---- PLANT tiles ----
        if kind == "PLANT":
            crop = tile.get("crop")
            info = CROPS.get(crop)
            if info is not None:
                # (1) harvest-ready
                if crop_ready(tile, info, day, ctx.step):
                    ctx.claimed.add(pos)
                    return ["HARVEST"]
                # (2) unwatered -> water (survival + one-time yield bonus)
                if not tile.get("watered_today", False):
                    ctx.claimed.add(pos)
                    return ["WATER"]
                # (3) fertilize during window if this unit carries fertilizer
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
                # Place an animal we're carrying that matches this structure.
                for aname, ainfo in ANIMALS.items():
                    if ainfo["structure"] == kind and int(_get(inv, aname, 0)) > 0:
                        ctx.claimed.add(pos)
                        return ["PLACE", aname]
                return None
            # Occupied: daily must-dos first, then collect, then harvest.
            if not tile.get("fed_today", False):
                if int(_get(ctx.shed_ledger, "WHEAT", 0)) > 0:
                    ctx.shed_ledger["WHEAT"] = int(ctx.shed_ledger.get("WHEAT", 0)) - 1
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
    # Animal pickup: an empty structure exists and the matching animal is in shed.
    for (x, y, kind, tile) in ctx.empty_structs:
        for aname, ainfo in ANIMALS.items():
            if ainfo["structure"] == kind:
                if int(_get(ctx.shed_ledger, aname, 0)) > 0 and int(_get(inv, aname, 0)) == 0:
                    ctx.shed_ledger[aname] = int(ctx.shed_ledger.get(aname, 0)) - 1
                    return ["PICKUP", aname, 1]

    # Fertilizer pickup: candidates exist, shed has fertilizer, we carry none.
    if ctx.fertilize_tiles:
        if int(_get(ctx.shed_ledger, "FERTILIZER", 0)) > 0 and int(_get(inv, "FERTILIZER", 0)) == 0:
            ctx.shed_ledger["FERTILIZER"] = int(ctx.shed_ledger.get("FERTILIZER", 0)) - 1
            return ["PICKUP", "FERTILIZER", 1]
    return None


def _move_to_task(ctx, state, idx, pos, inv):
    """Move toward the nearest unclaimed pending task, by priority category.
    Returns a movement op, or None if no pending task is worth moving to."""
    carrying_fert = int(_get(inv, "FERTILIZER", 0)) > 0
    carrying_animal = any(int(_get(inv, a, 0)) > 0 for a in ANIMALS)

    # Tiles this unit can PLANT on (planned crop we hold a seed for).
    plant_tiles = []
    if ctx.hour < ctx.tpd - 2:
        for coord, p in state.tile_plan.items():
            if p.get("kind") == "CROP" and int(_get(ctx.seeds_remaining, p["crop"], 0)) > 0:
                if get_tile(ctx.tiles, coord[0], coord[1]) is None:
                    plant_tiles.append(coord)

    # Structure build tiles (planned, still empty).
    build_tiles = []
    for coord, p in state.tile_plan.items():
        if p.get("kind") == "STRUCT" and get_tile(ctx.tiles, coord[0], coord[1]) is None:
            build_tiles.append(coord)

    # Place tiles (empty structures) — only worth moving to if we carry an animal.
    place_tiles = [(x, y) for (x, y, k, t) in ctx.empty_structs]

    # Priority categories (name, tiles, gate).  High -> low.
    categories = [
        ("harvest", ctx.harvest_tiles, True),
        ("water",   ctx.water_tiles, True),
        ("animal",  ctx.animal_tiles, True),
        ("place",   place_tiles, carrying_animal),
        ("build",   build_tiles, True),
        ("plant",   plant_tiles, True),
        ("weed",    ctx.weed_tiles, True),
        ("fert",    ctx.fertilize_tiles, carrying_fert),
    ]

    for name, tiles, gate in categories:
        if not gate:
            continue
        # Prefer high-value crop/animal tiles when several tasks tie in distance.
        best = None
        best_key = None
        for t in tiles:
            if t in ctx.claimed:
                continue
            if t == pos:
                continue  # handled by on-tile logic already
            d = manhattan(pos, t)
            val = _tile_value(ctx, t)
            key = (d, -val)  # nearest first, then most valuable
            if best_key is None or key < best_key:
                best_key = key
                best = t
        if best is not None:
            ctx.claimed.add(best)  # claim so other units diverge
            return move_toward(pos, best)
    return None


def _tile_value(ctx, coord):
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
    # Empty planned tile: value by its intended crop.
    plan = STATE.tile_plan.get(coord)
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

        # Keep the global TURNS_PER_DAY in sync if the config differs from default.
        global TURNS_PER_DAY
        if ctx.tpd and ctx.tpd != TURNS_PER_DAY:
            TURNS_PER_DAY = ctx.tpd

        update_phase(ctx, state=STATE)
        assign_plans(ctx, STATE)

        market = decide_market_orders(ctx, STATE)

        farmer_action = decide_unit_action(ctx, STATE, 0)
        hand_actions = [decide_unit_action(ctx, STATE, i + 1) for i in range(len(ctx.hands))]

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
    except Exception as e:  # pragma: no cover
        print("kaggle_environments not installed:", e)
        print("Install with:  pip install -U kaggle-environments")
        raise SystemExit(0)

    for opp in ["random", "starter"]:
        env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
        env.run([agent, opp])
        final = env.steps[-1]
        results = []
        for i, s in enumerate(final):
            if isinstance(s, dict):
                results.append((i, s.get("reward"), s.get("status")))
            else:
                results.append((i, getattr(s, "reward", None), getattr(s, "status", None)))
        print(f"vs {opp:8s} ->", results)
