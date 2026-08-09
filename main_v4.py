"""
Kaggriculture v4 - aggressive economic agent

Design goal: break out of the ~10k plateau by optimizing BANK CASH, not farm
appearance. This version fixes the main strategic errors in v3:

* no arbitrary 20-melon cap / tiny premium portfolio
* land is a first-class investment and is bought as soon as cash permits
* animals are scaled to physical capacity and continuously cared for
* selling is based on the actual market curve, not "1-3 premium items"
* inventory is kept below the shed danger zone
* final-day liquidation is only a safety valve, not the main exit strategy
* production targets are sized to use essentially the whole 100-tile farm

Public Kaggriculture discussion also identifies an all-melon "melon rush" as a
known strong meta. We therefore keep melon as the largest crop, while retaining
animals and a few lower-throughput goods to diversify market pressure.
"""
import math

TPD=24; N=10; H=5
SHED={(4,4),(5,4),(4,5),(5,5)}
MAX_ORDERS=10; SHED_CAP=100; I0=10000
BASE={"WHEAT":25,"CARROT":35,"TOMATO":60,"STRAWBERRY":120,"MELON":250,
      "EGG":50,"MILK":160,"WOOL":200,"FERTILIZER":100}
CROPS={
 "WHEAT":{"seed":10,"max":4,"ongoing":False},
 "CARROT":{"seed":20,"max":3,"ongoing":False},
 "TOMATO":{"seed":50,"max":11,"ongoing":True},
 "STRAWBERRY":{"seed":100,"max":16,"ongoing":True},
 "MELON":{"seed":80,"max":10,"ongoing":False},
}
ANIMALS={
 "GOOSE":{"cost":300,"structure":"COOP","product":"EGG"},
 "COW":{"cost":400,"structure":"PASTURE","product":"MILK"},
 "SHEEP":{"cost":500,"structure":"PASTURE","product":"WOOL"},
}
LAND=[1000,2000,4000]
# Market parameters from the supplied README.
MP={
 "WHEAT":(25,400,"sqrt",.80,"log",.20),
 "CARROT":(35,450,"log",.20,"sqrt",.70),
 "TOMATO":(60,200,"linear",.40,"sqrt",.60),
 "STRAWBERRY":(120,100,"sqrt",.70,"linear",1.60),
 "MELON":(250,300,"log",.20,"sq",3.60),
 "EGG":(50,332,"linear",.40,"log",.20),
 "MILK":(160,122,"sqrt",.60,"linear",1.60),
 "WOOL":(200,105,"log",.20,"sq",3.20),
 "FERTILIZER":(100,200,"linear",.40,"linear",.40),
}
DEMAND={"BAKERY":{"EGG","WHEAT"},"PIZZA_SHOP":{"MILK","TOMATO","WHEAT"},
"BRUNCH_SPOT":{"EGG","WHEAT","STRAWBERRY"},"YARN_STORE":{"WOOL"},
"ICE_CREAM_SHOP":{"STRAWBERRY","MILK","WHEAT"},"PET_CAFE":{"CARROT"},
"SMOOTHIE_SHOP":{"STRAWBERRY","MILK"},"FARMERS_MARKET":{"WHEAT","CARROT","TOMATO","STRAWBERRY"}}


def g(d,k,default=0):
    try:
        v=d.get(k,default); return default if v is None else v
    except Exception: return default

def tile(ts,x,y):
    try:return ts[y][x]
    except Exception:return "LOCKED"

def dist(a,b):return abs(a[0]-b[0])+abs(a[1]-b[1])

def move(a,b):
    dx=b[0]-a[0]; dy=b[1]-a[1]
    if abs(dx)>=abs(dy) and dx:return ["EAST" if dx>0 else "WEST"]
    if dy:return ["SOUTH" if dy>0 else "NORTH"]
    return ["PASS"]

def fn(x,name):
    if name=="linear":return x
    if name=="sq":return x*x
    if name=="sqrt":return math.sqrt(x)
    if name=="log":return math.log(1+x)
    return math.log10(1+x)

def market_price(good,inv):
    p=MP.get(good)
    if not p:return BASE.get(good,1)
    base,T,bf,bt,af,at=p
    d=inv-I0
    if d<0:
        amp=bt*base/fn(T,bf)
        return int(round(base+amp*fn(-d,bf)))
    amp=at*base/fn(T,af)
    return max(1,int(round(base-amp*fn(d,af))))

def max_sell_qty(good,inv,have,target_ratio):
    """Largest quantity whose *last* unit stays >= target_ratio*base."""
    if have<=0:return 0
    base=BASE[good]; target=max(1,int(base*target_ratio))
    # Binary search is cheap and avoids hard-coding the curve.
    lo,hi=0,min(have,100)
    while lo<hi:
        mid=(lo+hi+1)//2
        if market_price(good,inv+mid)>=target:lo=mid
        else:hi=mid-1
    return lo

class S:
    def __init__(self):self.reset()
    def reset(self):self.plan={};self.day=-1
STATE=S()

class C:
    def __init__(self,o):
        self.o=o; self.player=int(o.get("player",0)); self.step=int(o.get("step",0))
        self.day=int(o.get("day",self.step//TPD)); self.hour=int(o.get("hour",self.step%TPD))
        fs=o.get("farms",[]) or []; self.f=fs[self.player] if self.player<len(fs) else {}
        self.opp=fs[1-self.player] if len(fs)>1 else {}
        self.ts=self.f.get("tiles",[]) or []; self.money=float(g(self.f,"money",0))
        self.hands=self.f.get("hands",[]) or []; self.quads=self.f.get("unlocked_quadrants",["NW"]) or ["NW"]
        self.hires=int(g(self.f,"hires_today",0)); pr=o.get("private",{}) or {}
        self.shed=dict(pr.get("shed",{}) or {}); self.seeds=dict(pr.get("seeds",{}) or {})
        self.inv=pr.get("inventories",[]) or []; m=o.get("market",{}) or {}
        self.prices=dict(m.get("prices",{}) or {}); self.mi=dict(m.get("inventory",{}) or {})
        self.shops=(o.get("town",{}) or {}).get("unlocked_shops",[]) or []
        self.unlocked={(x,y) for y,r in enumerate(self.ts) for x,v in enumerate(r) if v!="LOCKED"}
        self.ledger=dict(self.shed); self.claim=set(); self.feedclaim=set(); self.scan()
    def pos(self,i):
        if i==0:p=self.f.get("farmer",[4,4])
        else:p=self.hands[i-1] if i-1<len(self.hands) else [4,4]
        return int(p[0]),int(p[1])
    def ui(self,i):return self.inv[i] if i<len(self.inv) and isinstance(self.inv[i],dict) else {}
    def scan(self):
        self.empty=[];self.weeds=[];self.water=[];self.harv=[];self.fert=[];self.es=[]
        self.unfed=[];self.uncared=[];self.afert=[];self.aharv=[];self.crops={};self.struct={"COOP":0,"PASTURE":0};self.placed={"GOOSE":0,"COW":0,"SHEEP":0}
        for x,y in self.unlocked:
            t=tile(self.ts,x,y)
            if t is None:self.empty.append((x,y));continue
            if not isinstance(t,dict):continue
            k=t.get("kind")
            if k=="WEED":self.weeds.append((x,y));continue
            if k=="PLANT":
                c=t.get("crop");self.crops[c]=self.crops.get(c,0)+1
                if c in CROPS:
                    age=self.day-int(g(t,"planted_day",self.day)); ready=g(t,"yield_units",0)>0 and (CROPS[c]["ongoing"] or age>=CROPS[c]["max"] or (g(t,"max_lifespan_step",-1)!=-1 and self.step>=g(t,"max_lifespan_step",-1)))
                    if ready:self.harv.append((x,y))
                    elif not t.get("watered_today",False):self.water.append((x,y))
                    if c in {"MELON","STRAWBERRY","TOMATO"} and g(t,"fertilized_until_day",-1)<self.day:
                        start=math.ceil(CROPS[c]["max"]/2) if not CROPS[c]["ongoing"] else 8 if c=="TOMATO" else 10
                        if start<=age<=CROPS[c]["max"]:self.fert.append((x,y))
                continue
            if k in ("COOP","PASTURE"):
                self.struct[k]+=1;a=t.get("animal")
                if not a:self.es.append((x,y,k))
                else:
                    self.placed[a]=self.placed.get(a,0)+1
                    if not t.get("fed_today",False):self.unfed.append((x,y))
                    elif not t.get("cared_today",False):self.uncared.append((x,y))
                    if t.get("fertilizer_available",False):self.afert.append((x,y))
                    if g(t,"yield_units",0)>0:self.aharv.append((x,y))
        self.shed_total=sum(int(v) for v in self.shed.values() if isinstance(v,(int,float)))
        self.owned={a:self.placed.get(a,0)+int(g(self.shed,a,0)) for a in ANIMALS}
        for a in ANIMALS:
            for z in self.inv:self.owned[a]+=int(g(z,a,0))

def opp_counts(c):
    out={}
    for r in c.opp.get("tiles",[]) or []:
        for t in r:
            if isinstance(t,dict):
                if t.get("kind")=="PLANT":out[t.get("crop")]=out.get(t.get("crop"),0)+1
                elif t.get("animal"):out[t.get("animal")]=out.get(t.get("animal"),0)+1
    return out

def targets(c):
    q=len(c.quads)
    # Deliberately aggressive. This is a near-full-farm production portfolio.
    crop={
      1:{"MELON":18,"TOMATO":2,"STRAWBERRY":1,"CARROT":1,"WHEAT":0},
      2:{"MELON":36,"TOMATO":5,"STRAWBERRY":2,"CARROT":2,"WHEAT":1},
      3:{"MELON":54,"TOMATO":10,"STRAWBERRY":5,"CARROT":5,"WHEAT":2},
      4:{"MELON":60,"TOMATO":14,"STRAWBERRY":7,"CARROT":5,"WHEAT":3},
    }[min(4,q)].copy()
    # If melon is already badly depressed, redirect only NEW land to tomato/carrot.
    mp=c.prices.get("MELON",250)/250
    if mp<0.45:
        cut=max(0,crop["MOMELON"] if "MOMELON" in crop else 0)
        crop["MELON"]=max(25,crop["MELON"]-15)
        crop["TOMATO"]+=8;crop["CARROT"]+=5
    return crop

def animal_targets(c):
    q=len(c.quads)
    return [{"GOOSE":1,"COW":1,"SHEEP":0},{"GOOSE":2,"COW":2,"SHEEP":1},{"GOOSE":4,"COW":4,"SHEEP":2},{"GOOSE":6,"COW":5,"SHEEP":3}][q-1]

def assign(c):
    for p in list(STATE.plan):
        if p not in c.unlocked or tile(c.ts,*p) is not None:del STATE.plan[p]
    pc={x:0 for x in CROPS};ps={"COOP":0,"PASTURE":0}
    for v in STATE.plan.values():
        if v["kind"]=="CROP":pc[v["crop"]]+=1
        else:ps[v["structure"]]+=1
    at=animal_targets(c); st={"COOP":at["GOOSE"],"PASTURE":at["COW"]+at["SHEEP"]}
    free=[p for p in c.empty if p not in STATE.plan];free.sort(key=lambda p:dist(p,(4,4)))
    for s in ("COOP","PASTURE"):
        need=max(0,st[s]-c.struct[s]-ps[s])
        while need and free:
            p=free.pop(0);STATE.plan[p]={"kind":"STRUCT","structure":s};ps[s]+=1;need-=1
    ct=targets(c)
    # Fill by economic value; melon first unless its price is badly depressed.
    score=[]
    for crop,n in ct.items():
        have=c.crops.get(crop,0)+pc[crop]; need=max(0,n-have)
        if need:
            y={"MELON":6,"TOMATO":4,"STRAWBERRY":4,"CARROT":3,"WHEAT":4}[crop]
            s=y*BASE[crop]/CROPS[crop]["seed"]*(c.prices.get(crop,BASE[crop])/BASE[crop])
            if crop=="MELON":s*=1.6
            score.append((s,crop,need))
    score.sort(reverse=True)
    for _,crop,n in score:
        while n and free:
            p=free.pop(0);STATE.plan[p]={"kind":"CROP","crop":crop};n-=1

def fib(n):
    a,b=1,1
    for _ in range(n):a,b=b,a+b
    return a

def hire(c):
    if c.hour!=0:return []
    desired=6 if len(c.quads)<=2 else 8
    if len(c.quads)>=4:desired=10
    out=[];spent=0
    for n in range(desired):
        x=fib(n)
        if c.money-spent>=x:out.append(["HIRE"]);spent+=x
        else:break
    return out

def sell_orders(c):
    # Keep premium prices above ~70% base. For staples allow 55%.
    out=[];demand=set()
    for s in c.shops:demand |= DEMAND.get(s,set())
    for good,raw in c.shed.items():
        if good not in BASE or good=="FERTILIZER":continue
        have=int(raw)
        if good=="WHEAT":have=max(0,have-(sum(c.placed.values())*2+3))
        if not have:continue
        p=int(c.prices.get(good,BASE[good])); ratio=p/BASE[good]
        target=.70 if good in {"MELON","STRAWBERRY","MILK","WOOL"} else .55
        # Shed pressure can override price protection.
        if c.shed_total>88:target=.35
        qty=max_sell_qty(good,int(c.mi.get(good,I0)),have,target)
        if c.day>=29:qty=have
        if qty:
            score=(p/BASE[good])+(2 if good in demand else 0)
            out.append((score,good,qty))
    out.sort(reverse=True)
    return [["SELL",g,n] for _,g,n in out]

def market(c):
    out=hire(c); cash=c.money-sum(fib(i) for i in range(len(out)))
    # Land is mandatory for scaling. Buy one per turn as soon as we can fund it.
    q=len(c.quads)
    if q<4:
        cost=LAND[q-1]
        # Minimal operating reserve. The point is to unlock productive tiles quickly.
        if cash>=cost+350 and c.day<=20:
            out.append(["BUY_LAND"]);cash-=cost
    # Structures -> animals. No animal may exist without a physical slot.
    at=animal_targets(c); empty={"COOP":0,"PASTURE":0}
    for _,_,s in c.es:empty[s]+=1
    for a in ("GOOSE","COW","SHEEP"):
        species=c.owned.get(a,0); placed=c.placed.get(a,0)
        cap=placed+empty[ANIMALS[a]["structure"]]
        if species>=at[a] or cap<=species:continue
        if cash>=ANIMALS[a]["cost"]+100:
            out.append(["BUY_ANIMAL",a,1]);cash-=ANIMALS[a]["cost"]
    # Seed exactly what the persistent plan needs.
    need={x:0 for x in CROPS}
    for p in STATE.plan.values():
        if p["kind"]=="CROP":need[p["crop"]]+=1
    for crop in ("MELON","TOMATO","STRAWBERRY","CARROT","WHEAT"):
        n=max(0,need[crop]-int(g(c.seeds,crop,0)))
        if n:
            cost=CROPS[crop]["seed"];buy=min(n,max(0,int((cash-100)//cost)))
            if buy:out.append(["BUY_SEED",crop,buy]);cash-=buy*cost
    # Buy fertilizer only as a bridge. Animals should provide the rest.
    fert=int(g(c.shed,"FERTILIZER",0))
    if fert<4 and cash>500:
        fp=int(c.prices.get("FERTILIZER",100));n=min(4-fert,int((cash-200)//max(1,fp)))
        if fp<=110 and n:out.append(["BUY_PRODUCT","FERTILIZER",n]);cash-=n*fp
    # Wheat feed reserve. We do not grow much wheat because melon is the main engine.
    reserve=sum(c.placed.values())*2+3;have=int(g(c.shed,"WHEAT",0))
    wp=int(c.prices.get("WHEAT",25))
    if have<reserve and cash>wp+100:
        n=min(reserve-have,int((cash-100)//max(1,wp)))
        if n:out.append(["BUY_PRODUCT","WHEAT",n]);cash-=n*wp
    out.extend(sell_orders(c))
    return out[:MAX_ORDERS]

def on_tile(c,idx,pos,t,inv):
    if isinstance(t,dict) and t.get("kind") in ("COOP","PASTURE") and t.get("animal"):
        if not t.get("fed_today",False):return ["FEED"] if g(inv,"WHEAT",0)>0 else None
        if not t.get("cared_today",False):return ["CARE"]
        if t.get("fertilizer_available",False):return ["COLLECT_FERTILIZER"]
        if g(t,"yield_units",0)>0:return ["HARVEST"]
        return None
    if isinstance(t,dict):
        k=t.get("kind")
        if k=="WEED":return ["DIG"]
        if k=="PLANT":
            crop=t.get("crop")
            if crop in CROPS:
                age=c.day-int(g(t,"planted_day",c.day))
                if g(t,"yield_units",0)>0 and (CROPS[crop]["ongoing"] or age>=CROPS[crop]["max"] or (g(t,"max_lifespan_step",-1)!=-1 and c.step>=g(t,"max_lifespan_step",-1))):return ["HARVEST"]
                if not t.get("watered_today",False):return ["WATER"]
                if c.fert and g(inv,"FERTILIZER",0)>0:return ["FERTILIZE"]
        if k in ("COOP","PASTURE") and not t.get("animal"):
            for a in ANIMALS:
                if ANIMALS[a]["structure"]==k and g(inv,a,0)>0:return ["PLACE",a]
    if t is None:
        p=STATE.plan.get(pos)
        if p:
            if p["kind"]=="STRUCT":return ["BUILD_COOP"] if p["structure"]=="COOP" else ["BUILD_PASTURE"]
            crop=p["crop"]
            if g(c.seeds,crop,0)>0 and c.hour<TPD-2:return ["PLANT",crop]
    return None

def nearest(tasks,pos,claim):
    z=[p for p in tasks if p not in claim]
    return min(z,key=lambda p:dist(pos,p)) if z else None

def pickup(c,idx,inv):
    pos=c.pos(idx)
    if c.unfed and g(inv,"WHEAT",0)==0 and g(c.ledger,"WHEAT",0)>0:
        c.ledger["WHEAT"]-=1;return ["PICKUP","WHEAT",1]
    # Animals first when there is an empty matching structure.
    for x,y,s in c.es:
        for a in ANIMALS:
            if ANIMALS[a]["structure"]==s and g(c.ledger,a,0)>0 and g(inv,a,0)==0:
                c.ledger[a]-=1;return ["PICKUP",a,1]
    if c.fert and g(c.ledger,"FERTILIZER",0)>0 and g(inv,"FERTILIZER",0)==0:
        c.ledger["FERTILIZER"]-=1;return ["PICKUP","FERTILIZER",1]
    return None

def action(c,idx):
    pos=c.pos(idx);inv=c.ui(idx);t=tile(c.ts,*pos);on=pos in SHED
    # Animal survival always wins.
    if c.unfed:
        if pos in c.unfed and g(inv,"WHEAT",0)>0:return ["FEED"]
        if g(inv,"WHEAT",0)>0:
            q=nearest(c.unfed,pos,c.feedclaim)
            if q:c.feedclaim.add(q);return move(pos,q)
        if on:
            a=pickup(c,idx,inv)
            if a:return a
        if g(c.ledger,"WHEAT",0)>0:return move(pos,min(SHED,key=lambda p:dist(pos,p)))
    a=on_tile(c,idx,pos,t,inv)
    if a:return a
    carried=[a for a in ANIMALS if g(inv,a,0)>0]
    if carried:
        q=nearest([p[:2] for p in c.es if p[2]==ANIMALS[carried[0]]["structure"]],pos,set())
        if q:return move(pos,q)
    if g(inv,"FERTILIZER",0)>0 and c.fert:
        q=nearest(c.fert,pos,set())
        if q:return move(pos,q)
    produce=sum(int(v) for k,v in inv.items() if k in BASE and k!="FERTILIZER")
    if produce:
        if on:return ["DROP"]
        return move(pos,min(SHED,key=lambda p:dist(pos,p)))
    if on:
        a=pickup(c,idx,inv)
        if a:return a
    queues=[c.uncared,c.afert,c.aharv,c.harv,c.water,c.weeds,
            [p for p,v in STATE.plan.items() if v["kind"]=="STRUCT" and tile(c.ts,*p) is None],
            [p for p,v in STATE.plan.items() if v["kind"]=="CROP" and tile(c.ts,*p) is None and g(c.seeds,v["crop"],0)>0],c.fert]
    for q in queues:
        z=nearest(q,pos,c.claim)
        if z:c.claim.add(z);return move(pos,z)
    return ["PASS"]

def agent(obs):
    try:
        if int(obs.get("step",0))==0:STATE.reset()
        c=C(obs);assign(c)
        m=market(c); farmer=action(c,0);hands=[action(c,i+1) for i in range(len(c.hands))]
        return {"farmer":farmer,"hands":hands,"market":m}
    except Exception:
        try:n=len((obs.get("farms",[]) or [])[int(obs.get("player",0))].get("hands",[]) or [])
        except Exception:n=0
        return {"farmer":["PASS"],"hands":[["PASS"] for _ in range(n)],"market":[]}

if __name__ == "__main__":
    from kaggle_environments import make
    import json

    print("=" * 70)
    print("KAGGRICULTURE V4 DIAGNOSTIC")
    print("=" * 70)

    env = make(
        "kaggriculture",
        configuration={
            "episodeSteps": 720
        },
        debug=True
    )

    print("Starting simulation...")
    env.run([agent, "starter"])

    print("\nSIMULATION FINISHED")
    print("=" * 70)

    # Print every day
    for step_no, step in enumerate(env.steps):

        if step_no % 24 != 0:
            continue

        if not isinstance(step, list):
            continue

        player = step[0]

        if not isinstance(player, dict):
            continue

        print(
            f"DAY {step_no // 24:02d} | "
            f"reward={player.get('reward')} | "
            f"status={player.get('status')}"
        )

        farms = player.get("observation", {}).get("farms", [])

        if farms:
            farm = farms[0]

            print(
                f"    money={farm.get('money')} "
                f"land={farm.get('unlocked_quadrants')} "
                f"hands={len(farm.get('hands', []))}"
            )

    print("\nFINAL RESULT")
    print("=" * 70)

    final = env.steps[-1]

    for i, p in enumerate(final):
        if isinstance(p, dict):
            print(
                f"PLAYER {i}: "
                f"reward={p.get('reward')} "
                f"status={p.get('status')}"
            )

    # Save complete replay
    try:
        with open("v4_replay.json", "w") as f:
            json.dump(env.toJSON(), f)

        print("\nReplay saved to: v4_replay.json")
    except Exception as e:
        print("Could not save replay:", e)
