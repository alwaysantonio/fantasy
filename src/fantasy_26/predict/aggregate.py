"""
Expected points: sub-models in, one row per player per gameweek out.

Every component is a probability or an expectation from a model that has
already been validated on its own. This module only multiplies them by FPL's
point values and adds them up, so when a total is wrong you can see which
component was wrong.

    appearance   p_sub · 1 + p_60 · 2                           minutes model
    goals        goals90 · E[min]/90 · pts[pos]                 attacking × minutes
    assists      assists90 · E[min]/90 · 3                      attacking × minutes
    clean sheet  p_60 · P(team CS) · pts[pos]                   minutes × Dixon-Coles
    conceded     E[⌊GC/2⌋] given λ_opp · min/90                 GK/DEF only
    saves        E[⌊S/3⌋], S ~ Poisson(SAVES_PER_GOAL · λ_opp)  GK only
    defcon       P(count ≥ threshold), count ~ Poisson(defcon90 · min/90)
    bonus        bonus90 · E[min]/90                            shrunk season rate
    negatives    yellows90 · E[min]/90 + red / own-goal priors

Point values come from FPL's own `game_config.scoring` in the latest
bootstrap payload, so a rule change mid-season is picked up without a code
change.

Expected values are additive, so a double gameweek is just two fixture rows
summed. Probabilities (p_60, p_clean_sheet) combine as 1 − Π(1 − p).

    # honest check on gameweeks no sub-model has seen
    uv run python -m fantasy_26.predict.aggregate evaluate --season 2026-27

    # write predictions for the next gameweek + 5 ahead
    uv run python -m fantasy_26.predict.aggregate run
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sqlalchemy import insert

from fantasy_26.db import SessionLocal, engine
from fantasy_26.features.builder import build_training_frame
from fantasy_26.models.schema import ModelRun, Prediction
from fantasy_26.predict import attacking, minutes
from fantasy_26.predict.team_strength import MODEL_NAME as STRENGTH_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("aggregate")

MODEL_NAME = "xp_aggregate"
VERSION = "0.1"

POS_KEY = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# Saves per goal conceded for keepers playing 60+, 2023-24..2025-26.
SAVES_PER_GOAL = 2.04
DEFCON_THRESHOLD = {2: 10, 3: 12, 4: 12}  # CBIT for DEF, CBIRT for MID/FWD
# Defensive action counts are overdispersed: variance ≈ 2 × mean in 2025-26
# for every outfield position. A Poisson tail underestimated DefCon hits by
# more than half, so use a negative binomial with that variance instead.
DEFCON_DISPERSION = 2.0
# Per-90 rates by position, 2023-24..2025-26. Too rare to model per player.
RED90 = {1: 0.001, 2: 0.008, 3: 0.006, 4: 0.004}
OWN_GOAL90 = {1: 0.006, 2: 0.009, 3: 0.002, 4: 0.001}

FALLBACK_SCORING = {
    "short_play": 1,
    "long_play": 2,
    "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
    "assists": 3,
    "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "goals_conceded": {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0},
    "saves": 1,
    "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2},
    "bonus": 1,
    "yellow_cards": -1,
    "red_cards": -3,
    "own_goals": -2,
}


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def load_scoring() -> dict:
    row = pd.read_sql(
        """
        SELECT payload::jsonb -> 'game_config' -> 'scoring' AS scoring
        FROM raw_ingests WHERE endpoint = 'bootstrap-static/'
        ORDER BY fetched_at DESC LIMIT 1
        """,
        engine,
    )
    if row.empty or row.iloc[0]["scoring"] is None:
        logger.warning("no scoring config in raw_ingests; using fallback values")
        return FALLBACK_SCORING
    s = row.iloc[0]["scoring"]
    return json.loads(s) if isinstance(s, str) else s


def load_projections() -> pd.DataFrame:
    """Latest Dixon-Coles projection for every fixture."""
    return pd.read_sql(
        """
        SELECT DISTINCT ON (fp.fixture_id)
               fp.fixture_id, fp.lambda_home, fp.lambda_away,
               fp.p_home_clean_sheet, fp.p_away_clean_sheet
        FROM fixture_projections fp
        JOIN model_runs mr ON mr.id = fp.model_run_id
        WHERE mr.name = %(name)s
        ORDER BY fp.fixture_id, fp.model_run_id DESC
        """,
        engine,
        params={"name": STRENGTH_MODEL},
    )


def _per_position(pos: pd.Series, table: dict) -> np.ndarray:
    return pos.map(lambda p: table.get(POS_KEY.get(p, "MID"), 0)).to_numpy(float)


def _overdispersed_sf(threshold: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """P(count ≥ threshold), count ~ NegBin with variance DEFCON_DISPERSION × mean."""
    p = 1.0 / DEFCON_DISPERSION
    n = np.clip(mean, 1e-6, None) / (DEFCON_DISPERSION - 1.0)
    return nbinom.sf(threshold - 1, n, p)


def _expected_floor(mu: np.ndarray, k: int, terms: int = 15) -> np.ndarray:
    """E[⌊X/k⌋] for X ~ Poisson(mu), via Σ_j P(X ≥ j·k)."""
    j = np.arange(1, terms + 1)
    return poisson.sf(j[None, :] * k - 1, mu[:, None]).sum(axis=1)


# --------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------


def fixture_components(
    df: pd.DataFrame,
    projections: pd.DataFrame,
    scoring: dict,
    minutes_model: minutes.MinutesModel,
    attacking_model: attacking.AttackingModel,
) -> pd.DataFrame:
    """Expected points by component for each player-fixture row.

    `df` must have been through `attacking.prepare` (or `prepare_upcoming`).
    """
    df = df[df["position"].isin(list(POS_KEY))]  # managers score differently
    df = df.merge(projections, on="fixture_id", how="left", validate="many_to_one")
    missing = df["lambda_home"].isna().sum()
    if missing:
        logger.warning("%d rows have no fixture projection; dropping them", missing)
        df = df[df["lambda_home"].notna()]
    df = df.reset_index(drop=True)

    home = df["was_home"].astype(bool).to_numpy()
    lam_against = np.where(home, df["lambda_away"], df["lambda_home"])
    p_team_cs = np.where(home, df["p_home_clean_sheet"], df["p_away_clean_sheet"])
    pos = df["position"]

    mins = minutes_model.predict(df)
    rates = attacking_model.predict(df)
    m_sub = minutes_model.class_minutes[1] / 90.0
    m_full = minutes_model.class_minutes[2] / 90.0
    p_sub, p_60 = mins["p_sub"].to_numpy(), mins["p_60"].to_numpy()
    e90 = mins["expected_minutes"].to_numpy() / 90.0

    def by_class(f):
        """E[f(minutes share)] over the two playing classes."""
        return p_sub * f(m_sub) + p_60 * f(m_full)

    out = df[
        ["player_season_id", "fixture_id", "gameweek_id", "gameweek", "web_name", "position"]
    ].copy()
    out["p_appear"] = mins["p_appear"].to_numpy()
    out["p_60"] = p_60
    out["expected_minutes"] = mins["expected_minutes"].to_numpy()
    out["expected_goals"] = rates["goals90"].to_numpy() * e90
    out["expected_assists"] = rates["assists90"].to_numpy() * e90
    out["p_clean_sheet"] = p_60 * p_team_cs
    out["expected_goals_conceded"] = lam_against * e90

    gk, dfn = (pos == 1).to_numpy(), (pos == 2).to_numpy()
    out["expected_saves"] = np.where(gk, SAVES_PER_GOAL * lam_against * e90, 0.0)

    defcon90 = df["defcon90"].to_numpy()
    threshold = pos.map(DEFCON_THRESHOLD).fillna(99).to_numpy()
    out["p_defcon"] = np.where(
        gk, 0.0, by_class(lambda m: _overdispersed_sf(threshold, defcon90 * m))
    )

    out["expected_bps"] = 0.0  # not modelled yet; bonus comes from bonus90
    bonus = df["bonus90"].to_numpy() * e90
    yellows = df["yellows90"].to_numpy() * e90
    reds = pos.map(RED90).fillna(0).to_numpy() * e90
    own_goals = pos.map(OWN_GOAL90).fillna(0).to_numpy() * e90

    out["pts_appearance"] = scoring["short_play"] * p_sub + scoring["long_play"] * p_60
    out["pts_goals"] = _per_position(pos, scoring["goals_scored"]) * out["expected_goals"]
    out["pts_assists"] = scoring["assists"] * out["expected_assists"]
    out["pts_clean_sheet"] = _per_position(pos, scoring["clean_sheets"]) * out["p_clean_sheet"]
    out["pts_conceded"] = np.where(
        gk | dfn,
        _per_position(pos, scoring["goals_conceded"])
        * by_class(lambda m: _expected_floor(lam_against * m, 2)),
        0.0,
    )
    out["pts_saves"] = np.where(
        gk,
        scoring["saves"]
        * by_class(lambda m: _expected_floor(SAVES_PER_GOAL * lam_against * m, 3, 20)),
        0.0,
    )
    out["pts_defcon"] = _per_position(pos, scoring["defensive_contribution"]) * out["p_defcon"]
    out["pts_bonus"] = scoring["bonus"] * bonus
    out["pts_negatives"] = (
        scoring["yellow_cards"] * yellows
        + scoring["red_cards"] * reds
        + scoring["own_goals"] * own_goals
    )
    out["expected_negatives"] = yellows + reds + own_goals

    pts_cols = [c for c in out.columns if c.startswith("pts_")]
    out["expected_points"] = out[pts_cols].sum(axis=1)

    # Rough spread for captaincy: Poisson for counts, Bernoulli for the rest.
    gpts = _per_position(pos, scoring["goals_scored"])
    cspts = _per_position(pos, scoring["clean_sheets"])
    app2 = p_sub * scoring["short_play"] ** 2 + p_60 * scoring["long_play"] ** 2
    out["points_var"] = (
        gpts**2 * out["expected_goals"]
        + scoring["assists"] ** 2 * out["expected_assists"]
        + cspts**2 * out["p_clean_sheet"] * (1 - out["p_clean_sheet"])
        + (app2 - out["pts_appearance"] ** 2)
        + 2.0 * bonus  # bonus arrives in 1-3 point lumps
    )
    return out


def to_gameweek(fx: pd.DataFrame) -> pd.DataFrame:
    """Sum fixture rows into one row per player per gameweek (doubles add up)."""
    keys = ["player_season_id", "gameweek_id"]
    sum_cols = [
        "expected_minutes",
        "expected_goals",
        "expected_assists",
        "expected_goals_conceded",
        "expected_saves",
        "expected_bps",
        "expected_negatives",
        "expected_points",
        "points_var",
        *[c for c in fx.columns if c.startswith("pts_")],
    ]
    prob_cols = ["p_appear", "p_60", "p_clean_sheet", "p_defcon"]

    g = fx.groupby(keys)
    out = g[sum_cols].sum()
    for c in prob_cols:
        out[c] = 1 - g[c].apply(lambda s: np.prod(1 - s))
    out[["gameweek", "web_name", "position"]] = g[["gameweek", "web_name", "position"]].first()
    out["n_fixtures"] = g.size()
    out["points_std"] = np.sqrt(out.pop("points_var").clip(lower=0))
    return out.reset_index()


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def actual_components(df: pd.DataFrame, scoring: dict) -> pd.DataFrame:
    """The same breakdown computed from what actually happened."""
    pos = df["position"]
    gk, dfn = (pos == 1).to_numpy(), (pos == 2).to_numpy()
    m = df["minutes"].to_numpy()
    a = df[["player_season_id", "gameweek_id"]].copy()
    a["pts_appearance"] = np.select(
        [m >= 60, m > 0], [scoring["long_play"], scoring["short_play"]], 0
    )
    a["pts_goals"] = _per_position(pos, scoring["goals_scored"]) * df["goals_scored"]
    a["pts_assists"] = scoring["assists"] * df["assists"]
    a["pts_clean_sheet"] = _per_position(pos, scoring["clean_sheets"]) * df["clean_sheets"]
    a["pts_conceded"] = np.where(
        gk | dfn,
        _per_position(pos, scoring["goals_conceded"]) * (df["goals_conceded"] // 2),
        0,
    )
    a["pts_saves"] = np.where(gk, scoring["saves"] * (df["saves"] // 3), 0)
    a["pts_defcon"] = (
        _per_position(pos, scoring["defensive_contribution"]) * df["defcon_hit"]
    )
    a["pts_bonus"] = scoring["bonus"] * df["bonus"]
    a["pts_negatives"] = (
        scoring["yellow_cards"] * df["yellow_cards"]
        + scoring["red_cards"] * df["red_cards"]
        + scoring["own_goals"] * df["own_goals"]
    )
    a["total_points"] = df["total_points"]
    return a.groupby(["player_season_id", "gameweek_id"]).sum().reset_index()


def evaluate(season: str) -> dict:
    """Score finished gameweeks of a season none of the sub-models trained on."""
    scoring = load_scoring()
    df = attacking.prepare(build_training_frame(engine, [season], min_appearances=0))
    fx = fixture_components(
        df, load_projections(), scoring, minutes.load_model(), attacking.load_model()
    )
    pred = to_gameweek(fx)
    act = actual_components(df, scoring)
    both = pred.merge(act, on=["player_season_id", "gameweek_id"], suffixes=("", "_act"))
    form = (
        df.groupby(["player_season_id", "gameweek_id"])["points_mean_5"].first().fillna(0)
    )
    both = both.join(form, on=["player_season_id", "gameweek_id"])

    print(f"\nComponent totals, {season} GW{both.gameweek.min()}-{both.gameweek.max()}"
          f" ({len(both)} player-gameweeks):")
    comps = [c for c in pred.columns if c.startswith("pts_")]
    t = pd.DataFrame(
        {
            "predicted": [both[c].sum() for c in comps],
            "actual": [both[f"{c}_act"].sum() for c in comps],
        },
        index=comps,
    )
    t.loc["TOTAL"] = [both["expected_points"].sum(), both["total_points"].sum()]
    t["ratio"] = t["actual"] / t["predicted"]
    print(t.round(2).to_string())

    y = both["total_points"]
    metrics = {
        "n": int(len(both)),
        "mae_model": float((both["expected_points"] - y).abs().mean()),
        "mae_form": float((both["points_mean_5"] - y).abs().mean()),
        "spearman_model": float(both.groupby("gameweek")[["expected_points", "total_points"]]
                                .corr("spearman").xs("total_points", level=1)["expected_points"].mean()),
        "spearman_form": float(both.groupby("gameweek")[["points_mean_5", "total_points"]]
                               .corr("spearman").xs("total_points", level=1)["points_mean_5"].mean()),
    }

    # The optimiser only cares about the top of the list.
    top = both.sort_values("expected_points", ascending=False).groupby("gameweek").head(20)
    top_form = both.sort_values("points_mean_5", ascending=False).groupby("gameweek").head(20)
    metrics["top20_actual_model"] = float(top["total_points"].mean())
    metrics["top20_actual_form"] = float(top_form["total_points"].mean())
    metrics["top20_predicted_model"] = float(top["expected_points"].mean())

    logger.info("MAE          model %.3f   form %.3f", metrics["mae_model"], metrics["mae_form"])
    logger.info("Spearman/GW  model %.3f   form %.3f", metrics["spearman_model"], metrics["spearman_form"])
    logger.info(
        "Top-20/GW actual points  model %.2f (predicted %.2f)   form %.2f",
        metrics["top20_actual_model"],
        metrics["top20_predicted_model"],
        metrics["top20_actual_form"],
    )

    both["decile"] = pd.qcut(both["expected_points"].rank(method="first"), 10, labels=False) + 1
    cal = both.groupby("decile").agg(
        n=("total_points", "size"),
        predicted=("expected_points", "mean"),
        actual=("total_points", "mean"),
    )
    print("\nCalibration by xP decile (mean points per player-gameweek):")
    print(cal.round(2).to_string())
    return metrics


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def next_gameweek() -> tuple[str, int]:
    row = pd.read_sql(
        """
        SELECT s.label, g.number FROM gameweeks g JOIN seasons s ON s.id = g.season_id
        WHERE s.is_current AND g.deadline > now()
        ORDER BY g.deadline LIMIT 1
        """,
        engine,
    )
    if row.empty:
        raise SystemExit("no upcoming gameweek in the current season")
    return row.iloc[0]["label"], int(row.iloc[0]["number"])


def run(gameweek: int | None, horizon: int) -> int:
    season, first = next_gameweek()
    first = gameweek or first
    last = min(first + horizon, 38)

    scoring = load_scoring()
    projections = load_projections()
    mm, am = minutes.load_model(), attacking.load_model()

    frames = []
    for gw in range(first, last + 1):
        df = attacking.prepare_upcoming([season], gw)
        gwp = to_gameweek(fixture_components(df, projections, scoring, mm, am))
        frames.append(gwp.assign(horizon=gw - first))
        logger.info("GW%d: %d players, top xP %.2f", gw, len(gwp), gwp["expected_points"].max())
    pred = pd.concat(frames, ignore_index=True)

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        sha = None

    # Columns not modelled yet (the p_bonus_* split) keep their schema default.
    skip = {"id", "model_run_id", "created_at", "actual_points"}
    cols = [
        c.key
        for c in Prediction.__table__.columns
        if c.key not in skip and c.key in pred.columns
    ]
    now = datetime.now(timezone.utc)
    with SessionLocal() as session:
        mr = ModelRun(
            name=MODEL_NAME,
            version=VERSION,
            trained_at=now,
            config={
                "season": season,
                "gameweeks": [first, last],
                "minutes_model": minutes.VERSION,
                "attacking_model": attacking.VERSION,
                "scoring": scoring,
            },
            git_sha=sha,
            note="sum of component expectations; see predict/aggregate.py",
        )
        session.add(mr)
        session.flush()
        rows = pred[cols].assign(model_run_id=mr.id, created_at=now)
        session.execute(insert(Prediction), rows.to_dict("records"))
        session.commit()
        run_id = mr.id

    logger.info("model run %d: %d predictions, GW%d-%d", run_id, len(pred), first, last)
    top = pred[pred["horizon"] == 0].nlargest(15, "expected_points")
    print(f"\nTop 15 for GW{first}:")
    print(
        top[["web_name", "position", "n_fixtures", "p_60", "expected_goals",
             "expected_assists", "p_clean_sheet", "expected_points", "points_std"]]
        .round(2).to_string(index=False)
    )
    return run_id


def backfill_actuals() -> int:
    """Fill `actual_points` on predictions for finished gameweeks."""
    with engine.begin() as conn:
        res = conn.exec_driver_sql(
            """
            UPDATE predictions p SET actual_points = a.pts
            FROM (
                SELECT player_season_id, gameweek_id, SUM(total_points) AS pts
                FROM player_fixture_stats GROUP BY 1, 2
            ) a, gameweeks g
            WHERE a.player_season_id = p.player_season_id
              AND a.gameweek_id = p.gameweek_id
              AND g.id = p.gameweek_id AND g.finished
              AND p.actual_points IS NULL
            """
        )
    logger.info("filled actual_points on %d predictions", res.rowcount)
    return res.rowcount


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Expected points aggregator")
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--season", default="2026-27")
    r = sub.add_parser("run")
    r.add_argument("--gameweek", type=int, default=None)
    r.add_argument("--horizon", type=int, default=5)
    sub.add_parser("actuals")
    args = parser.parse_args()
    if args.cmd == "evaluate":
        evaluate(args.season)
    elif args.cmd == "run":
        run(args.gameweek, args.horizon)
    elif args.cmd == "actuals":
        backfill_actuals()


if __name__ == "__main__":
    # Run via the package module, not __main__ (see predict/minutes.py).
    from fantasy_26.predict import aggregate

    aggregate.main()
