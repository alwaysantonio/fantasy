"""
Attacking rates — expected goals and assists per 90 for each player-fixture.

    goals90   = exp(b0 + b1·log(xg90_open) + b2·log(mult)) + PEN_XG · taker · mult
    assists90 = exp(c0 + c1·log(xa90)      + c2·log(mult))

`xg90` / `xa90` are the builder's shrunk season-to-date rates. `mult` is the
fixture multiplier from the Dixon-Coles fit:

    mult = exp(defence[opponent] + home_advantage · (is_home − ½))

A player's rate is already measured against a spread of opponents, so the
multiplier only has to describe how this opponent and venue differ from
average. The player's own team's attack is already in his rate.

The Poisson GLM is there to calibrate, not to discover. If the raw
`xg90 × mult` were perfectly calibrated, b0=0, b1=1 and b2=1. Early in a season
the rate is mostly prior, and the fitted b1 < 1 measures how much further to
shrink it.

The aggregator multiplies by expected minutes / 90. Evaluation here conditions
on ACTUAL minutes, so a minutes error can't hide an attacking error.

PENALTIES. FPL's xG includes penalty xG, and set-piece orders exist only in
this season's live snapshots, so the penalty effect can't be learned from
history. It is handled explicitly instead:

  1. Work out who held penalty duty at every past deadline this season: your
     set_piece_duties table first, FPL's penalties_order otherwise.
  2. Remove the penalty xG that duty already put into xg90 (`xg90_open`).
  3. Add PEN_XG back for whoever holds duty at this deadline.

So when duty changes, the new taker gains roughly 0.09 xG/90 straight away
instead of waiting a month for his rolling rate to catch up. For past seasons
nothing is known about duty, so both terms are zero and the GLM sees the
xg90 it was trained on.

    uv run python -m fantasy_26.predict.attacking evaluate
    uv run python -m fantasy_26.predict.attacking train
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sqlalchemy import Engine

from fantasy_26.db import SessionLocal, engine
from fantasy_26.features.builder import (
    CONFIG,
    assert_no_leakage,
    build_prediction_frame,
    build_training_frame,
)
from fantasy_26.models.schema import ModelRun
from fantasy_26.predict.team_strength import MODEL_NAME as STRENGTH_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("attacking")

MODEL_DIR = Path("models")
MODEL_NAME = "attacking_glm"
VERSION = "0.1"

# ≈0.12 penalties per team-match × 0.78 xG each. FPL doesn't publish penalties
# taken, so this is an assumption, not a fit. Revisit if you source pen data.
PEN_XG = 0.09

RATE_FLOOR = 0.005
UTC_NS = "datetime64[ns, UTC]"  # merge_asof needs both keys at the same resolution  # keeps log() finite for players whose rate is ~0

# 2022-23 is excluded: FPL started publishing xG partway through it, so its
# early xg90 values are deflated by zeros.
TRAIN_SEASONS = ["2023-24", "2024-25"]
VALID_SEASON = "2025-26"


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

STRENGTHS_SQL = """
SELECT ts.team_season_id, ts.as_of, ts.defence, ts.home_advantage, ts.model_run_id
FROM team_strengths ts
JOIN model_runs mr ON mr.id = ts.model_run_id
WHERE mr.name = %(name)s
"""

DUTIES_SQL = """
SELECT d.player_season_id, ps.team_season_id, d.effective_from, d.effective_to,
       d.penalty_order
FROM set_piece_duties d
JOIN player_seasons ps ON ps.id = d.player_season_id
WHERE d.penalty_order IS NOT NULL
"""


def attach_multiplier(df: pd.DataFrame, eng: Engine = engine) -> pd.DataFrame:
    """Opponent defence + venue, from the latest strength fit before the deadline.

    Several model runs can cover the same date (the full walk-forward, then the
    daily current-season refreshes). The latest run wins.
    """
    st = pd.read_sql(STRENGTHS_SQL, eng, params={"name": STRENGTH_MODEL})
    if st.empty:
        raise ValueError("no team strengths — run `make strength` first")
    st = (
        st.sort_values("model_run_id")
        .drop_duplicates(["team_season_id", "as_of"], keep="last")
        .assign(as_of=lambda x: pd.to_datetime(x["as_of"]).dt.tz_localize("UTC").astype(UTC_NS))
        .rename(columns={"team_season_id": "opponent_team_season_id"})
        .sort_values("as_of")
    )

    left = df.reset_index().assign(deadline=lambda x: x["deadline"].astype(UTC_NS))
    left = left.sort_values("deadline")
    merged = pd.merge_asof(
        left,
        st[["opponent_team_season_id", "as_of", "defence", "home_advantage"]],
        left_on="deadline",
        right_on="as_of",
        by="opponent_team_season_id",
        direction="backward",
    ).set_index("index")
    merged["deadline"] = df["deadline"]

    missing = merged["defence"].isna().sum()
    if missing:
        logger.warning("%d rows have no strength fit; using a neutral multiplier", missing)
    merged["opp_defence"] = merged["defence"].fillna(0.0)
    merged["mult"] = np.exp(
        merged["opp_defence"]
        + merged["home_advantage"].fillna(0.0) * (merged["is_home"] - 0.5)
    )
    return merged.drop(columns=["as_of", "defence", "home_advantage"]).loc[df.index]


def attach_penalty_duty(df: pd.DataFrame, eng: Engine = engine) -> pd.DataFrame:
    """`pen_taker` at each deadline, and the penalty xG it already put into xg90.

    A manual first-choice entry for a team replaces FPL's ordering for every
    player on that team, so naming a new taker also takes duty away from
    the old one.
    """
    df = df.sort_values(["player_season_id", "kickoff_time"]).copy()

    # FPL leaves penalties_order null both for "not on the list" and, in
    # backfilled snapshots, for "never recorded". A gameweek counts as observed
    # if anyone has an order. Unobserved gameweeks take the player's first
    # observed duty that season. Otherwise an established taker looks like a
    # non-taker before live capture began, his penalty xG never gets removed
    # from xg90, and PEN_XG is added on top of it.
    observed = df.groupby("gameweek_id")["penalties_order"].transform(
        lambda s: s.notna().any()
    )
    first_choice = (df["penalties_order"] == 1).astype(float).where(observed)
    df["pen_taker"] = (
        first_choice.groupby(df["player_season_id"]).bfill().fillna(0.0)
    )

    duties = pd.read_sql(DUTIES_SQL, eng)
    if not duties.empty:
        day = df["deadline"].dt.date
        for d in duties.itertuples():
            active = day >= d.effective_from
            if d.effective_to is not None and not pd.isna(d.effective_to):
                active &= day <= d.effective_to
            team = active & (df["team_season_id"] == d.team_season_id)
            if d.penalty_order == 1:
                df.loc[team, "pen_taker"] = 0.0
                df.loc[team & (df["player_season_id"] == d.player_season_id), "pen_taker"] = 1.0
            else:
                df.loc[team & (df["player_season_id"] == d.player_season_id), "pen_taker"] = 0.0

    # Same shrinkage arithmetic as the builder's xg90, so the subtraction
    # removes exactly the penalty share the rate contains.
    g = df.groupby("player_season_id", sort=False)
    taker_minutes = (df["minutes"] * df["pen_taker"]).groupby(df["player_season_id"])
    cum_taker = taker_minutes.cumsum().groupby(df["player_season_id"]).shift(1).fillna(0)
    cum_min = g["minutes"].cumsum().groupby(df["player_season_id"]).shift(1).fillna(0)
    df["pen_xg90_hist"] = PEN_XG * cum_taker / (cum_min + CONFIG.prior_minutes)
    df["xg90_open"] = (df["xg90"] - df["pen_xg90_hist"]).clip(lower=RATE_FLOOR)
    return df.sort_index()


def prepare(df: pd.DataFrame, eng: Engine = engine) -> pd.DataFrame:
    """Everything the model needs on top of the builder's frame."""
    df = attach_multiplier(df, eng)
    df = attach_penalty_duty(df, eng)
    return df


def prepare_upcoming(seasons: list[str], gameweek: int, eng: Engine = engine) -> pd.DataFrame:
    """`prepare` for an upcoming gameweek.

    build_prediction_frame returns only the target gameweek's rows, but penalty
    duty history needs the player's earlier matches. So run `prepare` over the
    season so far plus the upcoming rows, then keep the upcoming ones.
    """
    upcoming = build_prediction_frame(eng, seasons, gameweek)
    history = build_training_frame(eng, seasons, min_appearances=0)
    history = history[history["deadline"] < upcoming["deadline"].min()]
    both = pd.concat(
        [history.assign(_upcoming=False), upcoming.assign(_upcoming=True)],
        ignore_index=True,
    )
    both = prepare(both, eng)
    return both[both["_upcoming"]].drop(columns="_upcoming")


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def _design(df: pd.DataFrame, rate_col: str) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(df)),
            np.log(df[rate_col].clip(lower=RATE_FLOOR)),
            np.log(df["mult"]),
        ]
    )


@dataclass
class AttackingModel:
    goals_coef: np.ndarray  # [b0, b1 (log xg90_open), b2 (log mult)]
    assists_coef: np.ndarray  # [c0, c1 (log xa90), c2 (log mult)]
    pen_xg: float
    metrics: dict
    trained_at: datetime

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-90 rates. `df` must have been through `prepare`."""
        out = pd.DataFrame(index=df.index)
        open_play = np.exp(_design(df, "xg90_open") @ self.goals_coef)
        out["pen_xg90"] = self.pen_xg * df["pen_taker"] * df["mult"]
        out["goals90"] = open_play + out["pen_xg90"]
        out["assists90"] = np.exp(_design(df, "xa90") @ self.assists_coef)
        out["mult"] = df["mult"]
        # keepers almost never score or assist; don't let a prior invent it
        gk = df["position"] == 1
        out.loc[gk, ["goals90", "assists90", "pen_xg90"]] = 0.0
        return out


def _outfield_played(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["position"].isin([2, 3, 4])) & (df["minutes"] > 0)]


def _fit_glm(df: pd.DataFrame, rate_col: str, target: str) -> np.ndarray:
    res = sm.GLM(
        df[target].to_numpy(float),
        _design(df, rate_col),
        family=sm.families.Poisson(),
        offset=np.log(df["minutes"] / 90.0),
    ).fit()
    return np.asarray(res.params)


def fit(train: pd.DataFrame) -> AttackingModel:
    train = _outfield_played(train)
    gc = _fit_glm(train, "xg90_open", "y_goals")
    ac = _fit_glm(train, "xa90", "y_assists")
    logger.info("goals   b0=%+.3f  b1(xg90)=%.3f  b2(mult)=%.3f", *gc)
    logger.info("assists c0=%+.3f  c1(xa90)=%.3f  c2(mult)=%.3f", *ac)
    return AttackingModel(gc, ac, PEN_XG, {}, datetime.now(timezone.utc))


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def _poisson_ll(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean Poisson log-likelihood without the log(y!) constant."""
    mu = np.clip(mu, 1e-9, None)
    return float((y * np.log(mu) - mu).mean())


def score(valid: pd.DataFrame, model: AttackingModel) -> dict:
    """Conditional on actual minutes. Compared against the raw rates, with and
    without the fixture multiplier, so each piece has to earn its place."""
    v = _outfield_played(valid)
    exposure = v["minutes"].to_numpy() / 90.0
    pred = model.predict(v)
    out = {}
    for target, rate, col in (("y_goals", "xg90", "goals90"), ("y_assists", "xa90", "assists90")):
        y = v[target].to_numpy(float)
        cands = {
            "raw": v[rate].to_numpy() * exposure,
            "raw_x_mult": v[rate].to_numpy() * v["mult"].to_numpy() * exposure,
            "model": pred[col].to_numpy() * exposure,
        }
        out[target] = {
            name: {
                "poisson_ll": _poisson_ll(y, mu),
                "pred_total": float(mu.sum()),
                "actual_total": float(y.sum()),
            }
            for name, mu in cands.items()
        }
        for name, m in out[target].items():
            logger.info(
                "%-10s %-11s ll=%.5f  predicted=%7.1f  actual=%7.1f",
                target,
                name,
                m["poisson_ll"],
                m["pred_total"],
                m["actual_total"],
            )
    return out


def calibration_by_decile(valid: pd.DataFrame, model: AttackingModel) -> pd.DataFrame:
    """Within each predicted-goals decile, do predicted and actual totals agree?"""
    v = _outfield_played(valid)
    mu = model.predict(v)["goals90"] * v["minutes"] / 90.0
    t = pd.DataFrame({"pred": mu, "goals": v["y_goals"]})
    t["decile"] = pd.qcut(t["pred"].rank(method="first"), 10, labels=False) + 1
    out = t.groupby("decile").agg(n=("goals", "size"), predicted=("pred", "sum"), actual=("goals", "sum"))
    out["ratio"] = (out["actual"] / out["predicted"]).round(3)
    return out.round(1)


def load_frame(seasons: list[str]) -> pd.DataFrame:
    df = build_training_frame(engine, seasons)
    assert_no_leakage(df)
    return prepare(df)


def evaluate() -> dict:
    df = load_frame([*TRAIN_SEASONS, VALID_SEASON])
    model = fit(df[df["season_label"].isin(TRAIN_SEASONS)])
    valid = df[df["season_label"] == VALID_SEASON]
    metrics = score(valid, model)
    print(f"\nGoals calibration by predicted decile ({VALID_SEASON}):")
    print(calibration_by_decile(valid, model).to_string())
    return metrics


def train() -> AttackingModel:
    """Production fit on every season with full xG, validation season included."""
    seasons = [*TRAIN_SEASONS, VALID_SEASON]
    df = load_frame(seasons)
    model = fit(df)
    model.metrics = {"in_sample": score(df, model)}

    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{MODEL_NAME}_v{VERSION}.joblib"
    joblib.dump(model, path)
    logger.info("saved %s", path)

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        sha = None
    with SessionLocal() as session:
        session.add(
            ModelRun(
                name=MODEL_NAME,
                version=VERSION,
                trained_at=model.trained_at,
                config={
                    "seasons": seasons,
                    "pen_xg": PEN_XG,
                    "goals_coef": model.goals_coef.tolist(),
                    "assists_coef": model.assists_coef.tolist(),
                },
                metrics=model.metrics,
                git_sha=sha,
                note="Poisson GLM on shrunk xg90/xa90 × Dixon-Coles multiplier",
            )
        )
        session.commit()
    logger.info("registered model run")
    return model


def load_model(version: str = VERSION) -> AttackingModel:
    return joblib.load(MODEL_DIR / f"{MODEL_NAME}_v{version}.joblib")


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Attacking rates")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evaluate")
    sub.add_parser("train")
    args = parser.parse_args()
    if args.cmd == "evaluate":
        evaluate()
    elif args.cmd == "train":
        train()


if __name__ == "__main__":
    # Run via the package module, not __main__, so pickled model classes are
    # saved as fantasy_26.predict.attacking.* and can be loaded from anywhere.
    from fantasy_26.predict import attacking

    attacking.main()
