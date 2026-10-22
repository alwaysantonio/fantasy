"""
Squad optimiser — integer programme over the next few gameweeks.

Prediction is done (predictions table). This is the decision: which 15, which
XI, which captain, and which transfers, given rules a greedy pick can't handle
(budget, 3-per-club, formation, the bench having to be useful).

One model covers gameweeks t = 0..H:

    squad[p,t]  ∈ {0,1}   in the 15
    xi[p,t]     ≤ squad   starts
    cap[p,t]    ≤ xi      captain (points counted twice)
    buy/sell[p,t]         squad[t] = squad[t-1] + buy − sell, for t ≥ 1

    maximise Σ_t decay^t · ( Σ xp·xi + Σ xp·cap + bench_weight · Σ xp·(squad − xi) )
             − 4 · hits

At t = 0 the squad is either free (wildcard) or the current squad plus your
banked free transfers. Each week adds one free transfer, banked up to five,
plus hits if `max_hits` allows them. Only the t = 0 decisions are a recommendation. Later
weeks are a plan that shows why t = 0 looks the way it does, and it will be
re-solved next week with new predictions.

PRICES. Players already owned cost their selling price, everyone else costs
the current price. Future price changes are ignored, so the plan is slightly
optimistic about budget for anyone likely to rise.

    uv run python -m fantasy_26.optimise.squad --wildcard
    uv run python -m fantasy_26.optimise.squad              # free transfers only
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pulp

from fantasy_26.db import SessionLocal, engine
from fantasy_26.models.schema import OptimisationRun, Recommendation
from fantasy_26.predict.aggregate import MODEL_NAME as XP_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("optimise")

SQUAD_SHAPE = {1: 2, 2: 5, 3: 5, 4: 3}
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
MAX_PER_CLUB = 3
HIT_COST = 4
MAX_BANKED_FT = 5
POS_NAME = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass
class Config:
    wildcard: bool = False
    horizon: int = 5  # weeks after the first, so H+1 weeks in total
    decay: float = 0.85
    bench_weight: float = 0.1  # value of a bench point (autosubs, injuries)
    max_hits: int = 0  # per week
    transfers: int | None = None  # force exactly this many transfers this week
    # Friction per planned transfer after the first week. Predictions are
    # noisy, and without this the solver will swap equal-value players back
    # and forth just because a banked transfer is free.
    transfer_friction: float = 0.3
    pool_per_position: int = 60  # prune to the best N per position by horizon xP


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def load_state() -> dict:
    """Your latest squad, bank and free transfers, from the squad ingest."""
    eg = pd.read_sql(
        """
        SELECT eg.id, eg.entry_id, eg.bank, g.number AS gw, e.fpl_entry_id
        FROM entry_gameweeks eg
        JOIN entries e   ON e.id = eg.entry_id AND e.is_mine
        JOIN gameweeks g ON g.id = eg.gameweek_id
        ORDER BY g.deadline DESC LIMIT 1
        """,
        engine,
    )
    if eg.empty:
        raise SystemExit("no squad loaded — run `make ingest-squad` first")
    eg = eg.iloc[0]
    picks = pd.read_sql(
        "SELECT player_season_id, selling_price FROM entry_picks WHERE entry_gameweek_id = %(id)s",
        engine,
        params={"id": int(eg["id"])},
    )
    return {
        "entry_id": int(eg["entry_id"]),
        "fpl_entry_id": int(eg["fpl_entry_id"]),
        "last_gw": int(eg["gw"]),
        "bank": int(eg["bank"]),
        "squad": dict(zip(picks["player_season_id"], picks["selling_price"])),
        "free_transfers": free_transfers(int(eg["fpl_entry_id"])),
    }


def free_transfers(fpl_entry_id: int) -> int:
    """Free transfers for the next gameweek, replayed from the public history.

    One per week, banked up to five. A wildcard or free hit week neither uses
    nor adds to the bank; GW1 has unlimited transfers so nothing carries.
    """
    hist = pd.read_sql(
        """
        SELECT payload::jsonb AS p FROM raw_ingests
        WHERE endpoint = %(ep)s ORDER BY fetched_at DESC LIMIT 1
        """,
        engine,
        params={"ep": f"entry/{fpl_entry_id}/history/"},
    )
    if hist.empty:
        return 1
    payload = hist.iloc[0]["p"]
    chips = {c["event"]: c["name"] for c in payload.get("chips", [])}
    ft = 0
    for row in sorted(payload["current"], key=lambda r: r["event"]):
        if row["event"] == 1:
            ft = 1
            continue
        if chips.get(row["event"]) in ("wildcard", "freehit"):
            continue
        used = row["event_transfers"]
        ft = min(MAX_BANKED_FT, max(ft - used, 0) + 1)
    return max(ft, 1)


def load_pool(state: dict, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, int, pd.Series]:
    """Players with price, club and xP per gameweek from the latest xP run."""
    run_id = int(
        pd.read_sql(
            "SELECT max(id) AS id FROM model_runs WHERE name = %(n)s",
            engine,
            params={"n": XP_MODEL},
        ).iloc[0]["id"]
    )
    xp = pd.read_sql(
        """
        SELECT p.player_season_id, p.horizon, g.number AS gw, p.expected_points
        FROM predictions p JOIN gameweeks g ON g.id = p.gameweek_id
        WHERE p.model_run_id = %(r)s AND p.horizon <= %(h)s
        """,
        engine,
        params={"r": run_id, "h": cfg.horizon},
    )
    players = pd.read_sql(
        """
        SELECT DISTINCT ON (ps.id)
               ps.id AS player_season_id, pl.web_name, ps.team_season_id,
               t.short_name AS team, sn.now_cost,
               CASE sn.position WHEN 'GK' THEN 1 WHEN 'DEF' THEN 2
                                WHEN 'MID' THEN 3 WHEN 'FWD' THEN 4 END AS position
        FROM player_seasons ps
        JOIN seasons s        ON s.id = ps.season_id AND s.is_current
        JOIN players pl       ON pl.id = ps.player_id
        JOIN team_seasons ts  ON ts.id = ps.team_season_id
        JOIN teams t          ON t.id = ts.team_id
        JOIN player_snapshots sn ON sn.player_season_id = ps.id
        ORDER BY ps.id, sn.captured_at DESC
        """,
        engine,
    )
    players = players[players["position"].isin(SQUAD_SHAPE)].set_index("player_season_id")

    grid = xp.pivot_table(
        index="player_season_id", columns="horizon", values="expected_points", aggfunc="sum"
    ).reindex(columns=range(cfg.horizon + 1), fill_value=0.0)
    grid = grid.reindex(players.index).fillna(0.0)  # blank gameweek = 0

    # Owned players cost their selling price; everyone else the market price.
    players["cost"] = players["now_cost"]
    for ps_id, sell in state["squad"].items():
        if ps_id in players.index:
            players.loc[ps_id, "cost"] = sell
    players["owned"] = players.index.isin(list(state["squad"]))

    # Prune: the solver doesn't need the 400th-best defender. Owned players
    # always stay so selling them is an explicit decision.
    weights = cfg.decay ** np.arange(cfg.horizon + 1)
    players["horizon_xp"] = grid.to_numpy() @ weights
    keep = players.groupby("position")["horizon_xp"].rank(ascending=False) <= cfg.pool_per_position
    players = players[keep | players["owned"]]
    gws = xp.drop_duplicates("horizon").set_index("horizon")["gw"].sort_index()
    return players, grid.loc[players.index], run_id, gws


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def solve(state: dict, cfg: Config) -> dict:
    players, xp, xp_run, gws = load_pool(state, cfg)
    P, T = list(players.index), list(range(cfg.horizon + 1))
    pos, club = players["position"], players["team_season_id"]
    cost = players["cost"]
    budget = state["bank"] + sum(state["squad"].values())
    owned = set(state["squad"])

    m = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    B = pulp.LpVariable.dicts
    squad = B("squad", (P, T), cat="Binary")
    xi = B("xi", (P, T), cat="Binary")
    cap = B("cap", (P, T), cat="Binary")
    buy = B("buy", (P, T), cat="Binary")
    sell = B("sell", (P, T), cat="Binary")
    hits = B("hits", T, lowBound=0, upBound=cfg.max_hits, cat="Integer")
    if cfg.transfers is not None:
        if cfg.wildcard:
            raise SystemExit("--transfers doesn't apply to a wildcard week")
        # forcing N transfers may need more hits than max_hits allows
        hits[0].upBound = max(cfg.max_hits, cfg.transfers - state["free_transfers"])
        m += pulp.lpSum(buy[p][0] for p in P) == cfg.transfers

    obj = []
    for t in T:
        w = cfg.decay**t
        for p in P:
            v = float(xp.loc[p, t])
            obj += [w * v * xi[p][t], w * v * cap[p][t], w * cfg.bench_weight * v * (squad[p][t] - xi[p][t])]
        obj.append(-HIT_COST * w * hits[t])
        if t > 0:
            obj.append(-cfg.transfer_friction * w * pulp.lpSum(buy[p][t] for p in P))
    m += pulp.lpSum(obj)

    for t in T:
        m += pulp.lpSum(cost[p] * squad[p][t] for p in P) <= budget
        for k, n in SQUAD_SHAPE.items():
            m += pulp.lpSum(squad[p][t] for p in P if pos[p] == k) == n
            m += pulp.lpSum(xi[p][t] for p in P if pos[p] == k) >= XI_MIN[k]
            m += pulp.lpSum(xi[p][t] for p in P if pos[p] == k) <= XI_MAX[k]
        m += pulp.lpSum(xi[p][t] for p in P) == 11
        m += pulp.lpSum(cap[p][t] for p in P) == 1
        for c in club.unique():
            m += pulp.lpSum(squad[p][t] for p in P if club[p] == c) <= MAX_PER_CLUB
        for p in P:
            m += xi[p][t] <= squad[p][t]
            m += cap[p][t] <= xi[p][t]

        # transfers: t = 0 moves from the current squad, later weeks from t-1
        for p in P:
            prev = (1 if p in owned else 0) if t == 0 else squad[p][t - 1]
            m += squad[p][t] == prev + buy[p][t] - sell[p][t]

    # Free transfers bank up to five, so rolling one has value. A wildcard
    # week neither spends nor adds to the bank's existing transfers.
    ft = B("ft", T, lowBound=0, upBound=MAX_BANKED_FT, cat="Integer")
    m += ft[0] == state["free_transfers"]
    for t in T:
        n = pulp.lpSum(buy[p][t] for p in P)
        if t == 0 and cfg.wildcard:
            m += hits[t] == 0
            used = 0
        else:
            m += n <= ft[t] + hits[t]
            used = n - hits[t]
        if t + 1 in T:
            m += ft[t + 1] <= ft[t] - used + 1

    t0 = time.time()
    status = m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=300, gapRel=0.001))
    elapsed = time.time() - t0
    logger.info("solver: %s in %.1fs, objective %.2f", pulp.LpStatus[status], elapsed, pulp.value(m.objective))

    val = lambda v: (v.value() or 0) > 0.5  # noqa: E731
    weeks = []
    for t in T:
        rows = players.loc[[p for p in P if val(squad[p][t])]].copy()
        rows["xp"] = [float(xp.loc[p, t]) for p in rows.index]
        rows["start"] = [val(xi[p][t]) for p in rows.index]
        rows["captain"] = [val(cap[p][t]) for p in rows.index]
        weeks.append(
            {
                "gw": int(gws.get(t, t)),
                "squad": rows,
                "buys": [p for p in P if val(buy[p][t])],
                "sells": [p for p in P if val(sell[p][t])],
                "hits": int(round(hits[t].value() or 0)),
                "xi_xp": float((rows["xp"] * rows["start"]).sum() + rows.loc[rows["captain"], "xp"].sum()),
            }
        )
    return {
        "status": pulp.LpStatus[status],
        "objective": float(pulp.value(m.objective)),
        "seconds": elapsed,
        "budget": budget,
        "weeks": weeks,
        "players": players,
        "xp_run": xp_run,
    }


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def lineup(week: dict) -> pd.DataFrame:
    """XI first in formation order, then the bench: GK, then by xP."""
    sq = week["squad"].copy()
    starters = sq[sq["start"]].sort_values(["position", "xp"], ascending=[True, False])
    bench = sq[~sq["start"]]
    bench = pd.concat(
        [bench[bench["position"] == 1], bench[bench["position"] != 1].sort_values("xp", ascending=False)]
    )
    out = pd.concat([starters, bench])
    vice = starters[~starters["captain"]]["xp"].idxmax()
    out["role"] = ["XI"] * len(starters) + [f"bench {i + 1}" for i in range(len(bench))]
    out.loc[out["captain"], "role"] = "XI (C)"
    out.loc[vice, "role"] = "XI (V)"
    out["pos"] = out["position"].map(POS_NAME)
    out["price"] = out["cost"] / 10
    return out


def persist(state: dict, cfg: Config, result: dict) -> int:
    w0 = result["weeks"][0]
    gw_id = int(
        pd.read_sql(
            """SELECT g.id FROM gameweeks g JOIN seasons s ON s.id = g.season_id
               WHERE s.is_current AND g.number = %(n)s""",
            engine,
            params={"n": w0["gw"]},
        ).iloc[0]["id"]
    )
    lu = lineup(w0)
    with SessionLocal() as session:
        run = OptimisationRun(
            entry_id=state["entry_id"],
            gameweek_id=gw_id,
            model_run_id=result["xp_run"],
            created_at=datetime.now(timezone.utc),
            horizon_gws=cfg.horizon + 1,
            decay=cfg.decay,
            max_hits=cfg.max_hits,
            objective_value=result["objective"],
            solver_status=result["status"],
            solve_seconds=result["seconds"],
        )
        session.add(run)
        session.flush()
        recs = []
        for p, r in lu.iterrows():
            kind = "captain" if r["captain"] else ("start" if r["start"] else "bench")
            recs.append(Recommendation(run_id=run.id, kind=kind, player_season_id=int(p),
                                       target_gameweek_id=gw_id, xp_gain=float(r["xp"]),
                                       rationale=r["role"]))
        for p in w0["buys"]:
            recs.append(Recommendation(run_id=run.id, kind="buy", player_season_id=int(p),
                                       target_gameweek_id=gw_id,
                                       rationale="wildcard" if cfg.wildcard else "transfer"))
        for p in w0["sells"]:
            recs.append(Recommendation(run_id=run.id, kind="sell", player_season_id=int(p),
                                       target_gameweek_id=gw_id,
                                       rationale="wildcard" if cfg.wildcard else "transfer"))
        session.add_all(recs)
        session.commit()
        return run.id


def report(state: dict, cfg: Config, result: dict) -> None:
    players = result["players"]
    w0 = result["weeks"][0]
    lu = lineup(w0)
    spent = int(w0["squad"]["cost"].sum())
    print(f"\n=== GW{w0['gw']} {'WILDCARD' if cfg.wildcard else 'transfers'} — "
          f"spend £{spent / 10:.1f}m of £{result['budget'] / 10:.1f}m, "
          f"XI xP {w0['xi_xp']:.2f} (incl. captain) ===")
    print(lu[["role", "pos", "web_name", "team", "price", "xp"]].round(2).to_string(index=False))
    if not cfg.wildcard:
        ft = state["free_transfers"]
        if w0["buys"]:
            print(f"\nTransfers ({len(w0['buys'])} of {ft} free"
                  f"{', ' + str(w0['hits']) + ' hit(s)' if w0['hits'] else ''}):",
                  ", ".join(players.loc[w0["sells"], "web_name"]), "→",
                  ", ".join(players.loc[w0["buys"], "web_name"]))
        else:
            print(f"\nNo transfer this week: roll it "
                  f"(you'll have {min(ft + 1, MAX_BANKED_FT)} free next week).")

    print("\nPlan for later weeks (re-solved each week, not a commitment):")
    for w in result["weeks"][1:]:
        moves = (
            f"{', '.join(players.loc[w['sells'], 'web_name'])} → {', '.join(players.loc[w['buys'], 'web_name'])}"
            if w["buys"] else "roll transfer"
        )
        cap = w["squad"].loc[w["squad"]["captain"], "web_name"].iloc[0]
        print(f"  GW{w['gw']}: XI xP {w['xi_xp']:5.2f}  captain {cap:<12} {moves}")
    net = [w["xi_xp"] - HIT_COST * w["hits"] for w in result["weeks"]]
    total = sum(cfg.decay**t * v for t, v in enumerate(net))
    n_hits = sum(w["hits"] for w in result["weeks"])
    print(f"\nXI xP over GW{w0['gw']}-{result['weeks'][-1]['gw']}, net of {n_hits} hit(s): "
          f"{sum(net):.2f}   (discounted {total:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Squad optimiser")
    parser.add_argument("--wildcard", action="store_true")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--decay", type=float, default=0.85)
    parser.add_argument("--max-hits", type=int, default=0,
                        help="hits allowed per week; taken only if worth more than 4 points")
    parser.add_argument("--transfers", type=int, default=None,
                        help="make exactly this many transfers this week, hits included")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        wildcard=args.wildcard,
        horizon=args.horizon,
        decay=args.decay,
        max_hits=args.max_hits,
        transfers=args.transfers,
    )
    state = load_state()
    logger.info("entry %d after GW%d: bank £%.1fm, %d free transfer(s)",
                state["fpl_entry_id"], state["last_gw"], state["bank"] / 10, state["free_transfers"])
    result = solve(state, cfg)
    report(state, cfg, result)
    if not args.no_save:
        logger.info("saved optimisation run %d", persist(state, cfg, result))


if __name__ == "__main__":
    from fantasy_26.optimise import squad

    squad.main()
