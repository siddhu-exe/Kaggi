"""
Kaggriculture competitive agent — main.py
==========================================

A single-file, defensive agent for the Kaggle "Kaggriculture" two-player farming
simulation.  Implements a phased economic strategy:

    Phase 1  (Foundation, day 0..~4)   wheat + carrot staple engine, sell wheat
                                        aggressively, pace carrot.
    Phase 2  (Transition, day ~5..~9)  add tomato, buy a goose + coop, start
                                        fertilizing high-value crops.
    Phase 3  (Expansion,  day 10+)     buy land when full & flush, add
                                        strawberry / melon / cow / sheep, pace
                                        premium sells, watch the town + opponent.

Architecture (see Docs/GAME_MODEL.md for the full write-up):
    * GameState            module-global object, persists across turns.
    * TurnContext          per-turn parsed view of the observation + task lists.
    * decide_unit_action   one action per farmer / hand (priority checklist).
    * decide_market_orders BUY_SEED / BUY_ANIMAL / BUY_PRODUCT / SELL / BUY_LAND.
    * decide_hiring        Fibonacci-priced HIRE orders at day start.
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
      always hold a wheat feed reserve so it works regardless of source).
    * Glut-sensitive goods (carrot, strawberry, melon, milk, wool) crash toward
      the $1 floor -> paced, capped sells.  Wheat / eggs absorb gluts -> sell
      freely (keeping a wheat feed reserve).
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

# Per-turn SELL caps + a "don't sell below this fraction of base" pacing gate for
# glut-sensitive goods.  Wheat / eggs are uncapped (cap 9999).
SELL_RULES = {
    "WHEAT":      {"cap": 9999, "min_frac": 0.0},   # sell surplus above feed reserve
    "EGG":        {"cap": 9999, "min_frac": 0.0},
    "TOMATO":     {"cap": 6,    "min_frac": 0.45},  # moderate
    "CARROT":     {"cap": 3,    "min_frac": 0.45},
    "STRAWBERRY": {"cap": 2,    "min_frac": 0.50},
    "MELON":      {"cap": 2,    "min_frac": 0.50},
    "MILK":       {"cap": 2,    "min_frac": 0.50},
    "WOOL":       {"cap": 2,    "min_frac": 0.50},
    "FERTILIZER": {"cap": 4,    "min_frac": 0.40},
}

SHED_CAPACITY = 100
SHED_OVERFLOW_FORCE = 85    # when shed non-seed items exceed this, dump paced goods
MAX_MARKET_ORDERS = 10

# Cash we refuse to spend *seed money* below, per phase (big buys have own buffers).
RESERVE_FLOOR = {1: 800, 2: 600, 3: 500}


# ---------------------------------------------------------------------------
# GAME STATE  (module-global; Kaggle keeps our script globals alive per episode)
# ---------------------------------------------------------------------------

class GameState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = 1
        # Intended purpose per tile: {(x,y): {"kind":"CROP","crop":X}} or
        # {(x,y): {"kind":"STRUCT","structure":"COOP","animal":"GOOSE"}}
        self.tile_plan = {}
        self.last_day_hired = -1     # guard: hire only once per day
        self.want_coop = False       # PHASE 2 gate for the first goose coop
        self.want_pastures = False   # PHASE 3 gate for cow/sheep pastures
        self.expansion_planned = False
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
    """Advance the phase.  Day thresholds are soft — real transitions are gated
    on cash / infrastructure inside the buy logic, these just open the door."""
    day = ctx.day
    # --- PHASE 1 -> 2 : after ~day 5, once the staple engine is turning over. ---
    if state.phase < 2 and day >= 5:
        state.phase = 2
    # --- PHASE 2 -> 3 : after day 10. ---
    if state.phase < 3 and day >= 10:
        state.phase = 3

    # PHASE 2 gate: want a goose coop once we're in phase 2 (wheat feed sustainable).
    state.want_coop = state.phase >= 2
    # PHASE 3 gate: pastures for cow/sheep once expanded & cash is healthy.
    state.want_pastures = state.phase >= 3 and ctx.money >= 1200


# ---------------------------------------------------------------------------
# TILE PLANNING  — decide what each empty tile *should* become.
# ---------------------------------------------------------------------------

def desired_allocation(ctx, state):
    """Ordered [(item, target_count)] the current phase wants on the farm.
    Filled greedily against available tiles, so earlier entries win the land."""
    phase = state.phase
    if phase == 1:
        plan = [("WHEAT", 12), ("CARROT", 5)]                       # --- PHASE 1 ---
    elif phase == 2:
        plan = [("WHEAT", 10), ("TOMATO", 5), ("CARROT", 3)]        # --- PHASE 2 ---
        if state.want_coop:
            plan.append(("COOP", 1))
    else:
        plan = [("WHEAT", 12), ("TOMATO", 6), ("MELON", 4),         # --- PHASE 3 ---
                ("STRAWBERRY", 4), ("CARROT", 3)]
        if state.want_coop:
            plan.append(("COOP", 1))
        if state.want_pastures:
            plan.append(("PASTURE", 2))

    # Opponent-overproduction guard: if the opponent is flooding a glut-sensitive
    # crop, cut our target for it so we don't crash the price together.
    opp_counts = _opp_crop_counts(ctx)
    adj = []
    for item, cnt in plan:
        if item in GLUT_SENSITIVE and opp_counts.get(item, 0) >= 4:
            cnt = max(1, cnt // 2)
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
    if ctx.hour != 0:
        return []
    if state.last_day_hired == ctx.day:
        return []
    state.last_day_hired = ctx.day

    # --- Hand count by phase (Fibonacci keeps these cheap: 3 hands = $4/day). ---
    n_quads = len(ctx.unlocked_quadrants)
    if state.phase == 1:
        target = 3                                   # --- PHASE 1 ---
    elif state.phase == 2:
        target = 3                                   # --- PHASE 2 ---
    else:
        target = min(6, 2 + n_quads)                 # --- PHASE 3: scale w/ land ---

    # Never let hiring alone eat the reserve (fib cost is tiny, but be safe).
    reserve = RESERVE_FLOOR.get(state.phase, 500)
    cost = 0
    a, b = 1, 1
    orders = []
    for i in range(target):
        fib = a
        if ctx.money - (cost + fib) < 0:
            break
        cost += fib
        orders.append(["HIRE"])
        a, b = b, a + b
    # Keep at least a little cash even after hiring (cheap so rarely binds).
    if ctx.money - cost < reserve and len(orders) > 1 and state.phase >= 2:
        pass  # hiring is cheap enough that we allow it; reserve guards big buys
    return orders


# ---------------------------------------------------------------------------
# MARKET ORDERS  — hires + buys + paced sells, capped at MAX_MARKET_ORDERS.
# ---------------------------------------------------------------------------

def decide_market_orders(ctx, state):
    orders = []
    orders.extend(decide_hiring(ctx, state))

    reserve = RESERVE_FLOOR.get(state.phase, 500)
    money = ctx.money

    # ---- 1) BUY_LAND (PHASE 3 only, deliberate: full land + healthy cash) ----
    if state.phase >= 3:                                      # --- PHASE 3 ---
        extras = max(0, len(ctx.unlocked_quadrants) - 1)
        land_costs = [1000, 2000, 4000]
        if extras < len(land_costs):
            land_cost = land_costs[extras]
            nearly_full = len(ctx.empty_tiles) <= 3
            if nearly_full and money >= land_cost + 1200:
                orders.append(["BUY_LAND"])
                money -= land_cost

    # ---- 2) BUY_ANIMAL (buy the animal once its empty structure exists) ----
    #        Structure is *built* by a unit; we buy the animal so a unit can PLACE it.
    for (x, y, kind, tile) in ctx.empty_structs:
        want_animal = "GOOSE" if kind == "COOP" else None
        if want_animal is None:
            # Pasture: prefer cow first, then sheep (PHASE 3), if we can afford it.
            want_animal = "COW" if state.phase >= 3 else None
        if want_animal is None:
            continue
        info = ANIMALS.get(want_animal, {})
        cost = info.get("cost", 99999)
        buffer = 400 if want_animal == "GOOSE" else 600
        already = _get(ctx.shed, want_animal, 0)
        if already <= 0 and money >= cost + buffer:
            orders.append(["BUY_ANIMAL", want_animal, 1])
            money -= cost
            break  # one animal purchase per turn keeps cash flow smooth

    # ---- 3) BUY_SEED (cover planned-but-unseeded tiles, respect reserve) ----
    seed_need = {}
    for coord, p in state.tile_plan.items():
        if p["kind"] != "CROP":
            continue
        crop = p["crop"]
        seed_need[crop] = seed_need.get(crop, 0) + 1
    # Order: cheap staples first so they always plant; premium only in later phases.
    for crop in ("WHEAT", "CARROT", "TOMATO", "MELON", "STRAWBERRY"):
        want = seed_need.get(crop, 0)
        have = int(_get(ctx.seeds_remaining, crop, 0))
        buy = want - have
        if buy <= 0:
            continue
        unit = CROPS[crop]["seed"]
        # Buy as many as we can afford above the reserve floor.
        affordable = int(max(0, (money - reserve)) // unit)
        n = min(buy, affordable)
        if n > 0:
            orders.append(["BUY_SEED", crop, n])
            money -= n * unit

    # ---- 4) BUY_PRODUCT WHEAT (top up feed reserve if short & we own animals) ----
    feed_reserve = _wheat_feed_reserve(ctx)
    wheat_have = int(_get(ctx.shed, "WHEAT", 0))
    if ctx.placed_animals > 0 and wheat_have < feed_reserve:
        wprice = int(_get(ctx.prices, "WHEAT", BASE_PRICES["WHEAT"]))
        if wprice <= BASE_PRICES["WHEAT"] * 1.6 and money > reserve:
            n = min(feed_reserve - wheat_have,
                    int(max(0, (money - reserve)) // max(1, wprice)))
            if n > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", n])
                money -= n * wprice

    # ---- 5) BUY_PRODUCT FERTILIZER (PHASE 2+, small top-up for high-value crops) ----
    if state.phase >= 2 and ctx.fertilize_tiles:                # --- PHASE 2 ---
        fert_have = int(_get(ctx.shed, "FERTILIZER", 0))
        if fert_have < 3 and money >= 900:
            fprice = int(_get(ctx.prices, "FERTILIZER", BASE_PRICES["FERTILIZER"]))
            n = min(3 - fert_have, 2)
            if fprice <= BASE_PRICES["FERTILIZER"] * 1.4 and money - n * fprice > reserve + 500:
                orders.append(["BUY_PRODUCT", "FERTILIZER", n])
                money -= n * fprice

    # ---- 6) SELL (paced for glut-sensitive goods; wheat/eggs freely) ----
    orders.extend(decide_sells(ctx, state, feed_reserve))

    # Respect the per-turn order cap (extras are silently dropped by the engine).
    return orders[:MAX_MARKET_ORDERS]


def _wheat_feed_reserve(ctx):
    """Wheat to hold in the shed as animal feed: a few days' buffer per animal."""
    if ctx.placed_animals <= 0:
        return 0
    return ctx.placed_animals * 3 + 2


def decide_sells(ctx, state, feed_reserve):
    """Build SELL orders honoring caps, price floors, and the wheat feed reserve."""
    sells = []
    shed = ctx.shed
    prices = ctx.prices
    overflow = ctx.shed_total > SHED_OVERFLOW_FORCE

    # Bias order: town-demanded goods first, then glut-tolerant, then the rest.
    demanded = _town_demanded_goods(ctx)
    sellable = [g for g in SELL_RULES.keys() if int(_get(shed, g, 0)) > 0]

    def sort_key(g):
        return (0 if g in demanded else 1,          # demanded first
                0 if g in GLUT_TOLERANT else 1,     # then gluttolerant
                -int(_get(prices, g, 0)))           # then higher price
    sellable.sort(key=sort_key)

    for good in sellable:
        have = int(_get(shed, good, 0))
        rule = SELL_RULES[good]
        price = int(_get(prices, good, BASE_PRICES.get(good, 1)))
        base = BASE_PRICES.get(good, price)

        # Reserve wheat as feed before selling any.
        if good == "WHEAT":
            have = max(0, have - feed_reserve)
            if have <= 0:
                continue

        # Pacing gate for glut-sensitive goods: wait for price recovery unless the
        # shed is about to overflow (then dump to avoid end-of-day discard).
        if good in GLUT_SENSITIVE or rule["min_frac"] > 0:
            if price < base * rule["min_frac"] and not overflow:
                continue

        n = min(have, rule["cap"])
        if overflow:
            n = have  # dump everything when overflow-threatened
        if n > 0:
            sells.append(["SELL", good, n])

    return sells


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
    if state.phase >= 2 and ctx.fertilize_tiles:
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
