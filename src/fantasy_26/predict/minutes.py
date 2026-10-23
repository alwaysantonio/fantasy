"""
Minutes model — the single highest-value component in the stack.

Predicts a THREE-CLASS distribution, not a minutes regression:

    0 = did not play
    1 = 1-59 minutes   (1 appearance point, no clean sheet points)
    2 = 60+ minutes    (2 appearance points, clean sheet eligible)

Why classification: FPL scoring is a step function at 0 and 60, and minutes
are violently bimodal — a spike at 0, another at 90, little in between. A
regression predicting "48 minutes" describes a player who barely exists, and
the number cannot be turned into appearance or clean-sheet points.

CALIBRATION IS THE POINT. Everything downstream multiplies these outputs by
points, so a model that says 0.8 and is right 80% of the time is worth more
than one with better top-1 accuracy and meaningless probabilities. The raw
booster came out systematically under-confident in the 0.2-0.7 range, which
would quietly under-value every rotation-risk defender, so an isotonic
calibrator is fitted on a season the booster never saw.

THREE-WAY SPLIT, always by season:

    train     earlier seasons        fits the booster
    calibrate one middle season      fits the isotonic layer (booster unseen)
    validate  the latest season      honest metrics (both unseen)

Fitting the calibrator on the training data would be pointless — the booster
is already well calibrated on data it memorised. It needs unseen data.

    # evaluation run: honest metrics
    uv run python -m fantasy_26.predict.minutes train

    # production run: no holdout, calibrate on the most recent complete season
    uv run python -m fantasy_26.predict.minutes train --calib 2025-26 --no-valid
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from fantasy_26.db import SessionLocal, engine
from fantasy_26.features.builder import (
    FEATURE_COLUMNS,
    build_training_frame,
    assert_no_leakage,
)
from fantasy_26.models.schema import ModelRun

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("minutes")

MODEL_DIR = Path("models")
MODEL_NAME = "minutes_lgbm"
VERSION = "0.2"

# Managers score on an entirely different basis; they are not footballers.
EXCLUDED_POSITIONS = (5,)

# FPL's injury flags, applied on top of the booster. The booster can't learn
# them: past seasons have no flag history (FPL doesn't publish it), so it never
# saw a flagged player and gave chance_next zero importance. Until a season of
# our own snapshots exists to train on, scale P(plays) by FPL's own figure.
#
# Without a return date, the unavailable share shrinks by this factor for each
# week past the first gameweek after the snapshot. UNAVAILABLE (left the club,
# loaned out) is absent, so it never recovers.
RECOVERY_PER_WEEK = {"DOUBTFUL": 0.5, "INJURED": 0.75, "SUSPENDED": 0.75}
_RETURN_DATE = re.compile(r"(?:Expected back|Suspended until) (\d{1,2} [A-Z][a-z]{2})")


def _return_date(news: str, captured_at: pd.Timestamp) -> pd.Timestamp:
    """'Expected back 8 Nov' → 2026-11-08. The news carries no year."""
    m = _RETURN_DATE.search(news) if isinstance(news, str) else None
    if not m:
        return pd.NaT
    date = pd.to_datetime(f"{m.group(1)} {captured_at.year}", format="%d %b %Y", utc=True)
    if date < captured_at - pd.Timedelta(days=60):  # Dec snapshot, "back 3 Jan"
        date += pd.DateOffset(years=1)
    return date


def availability(frame: pd.DataFrame) -> np.ndarray:
    """P(available) for each row, from the snapshot it was built on.

    A stated return date wins: out until then, available from then. Otherwise
    the flag's chance applies in full to the first gameweek after the snapshot
    and fades by RECOVERY_PER_WEEK after that. Rows without a snapshot (past
    seasons) get 1.
    """
    if "status" not in frame or "captured_at" not in frame:
        return np.ones(len(frame))
    status = frame["status"].fillna("AVAILABLE")
    chance = np.where(
        status == "AVAILABLE", 1.0, frame["chance_next"].fillna(100).to_numpy() / 100.0
    )
    weeks = (
        ((frame["deadline"] - frame["captured_at"]).dt.days // 7).fillna(0).clip(lower=0)
    ).to_numpy()
    rate = status.map(RECOVERY_PER_WEEK).fillna(1.0).to_numpy()
    factor = 1.0 - (1.0 - chance) * rate**weeks

    flagged = (status != "AVAILABLE").to_numpy()
    if flagged.any() and "news" in frame:
        sub = frame[flagged]
        back = pd.Series(
            [_return_date(n, c) for n, c in zip(sub["news"], sub["captured_at"])],
            index=sub.index,
            dtype="datetime64[ns, UTC]",
        )
        when = sub["kickoff_time"] if "kickoff_time" in sub else sub["deadline"]
        dated = back.notna().to_numpy()
        stated = np.where(when.to_numpy() >= back.to_numpy(), 1.0, chance[flagged])
        factor[flagged] = np.where(dated, stated, factor[flagged])
    return factor


@dataclass
class MinutesModel:
    """Calibrated classifier plus what's needed to turn classes into minutes."""

    booster: object  # calibrated; exposes predict_proba
    raw_booster: object  # uncalibrated, kept for feature_importances_
    features: list[str]
    class_minutes: dict[int, float]  # E[minutes | class], learned not assumed
    metrics: dict
    trained_at: datetime

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[self.features].astype(float)
        proba = self.booster.predict_proba(X)
        avail = availability(frame)
        proba[:, 1] *= avail
        proba[:, 2] *= avail
        proba[:, 0] = 1.0 - proba[:, 1] - proba[:, 2]

        out = pd.DataFrame(index=frame.index)
        out["p_zero"] = proba[:, 0]
        out["p_sub"] = proba[:, 1]
        out["p_60"] = proba[:, 2]
        out["p_appear"] = 1.0 - proba[:, 0]
        out["expected_minutes"] = (
            proba[:, 0] * self.class_minutes[0]
            + proba[:, 1] * self.class_minutes[1]
            + proba[:, 2] * self.class_minutes[2]
        )
        return out

    def feature_importance(self, top: int = 20) -> pd.Series:
        return (
            pd.Series(self.raw_booster.feature_importances_, index=self.features)
            .sort_values(ascending=False)
            .head(top)
        )


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------


def three_way_split(
    df: pd.DataFrame,
    calib_season: str,
    valid_season: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by season, never randomly.

    A random split leaks badly: consecutive matches for one player share almost
    all their rolling features, so every holdout row has near neighbours in
    training. You get a wonderful score and a model that cannot forecast.
    """
    if "season_label" not in df.columns:
        raise ValueError("build_training_frame must emit season_label")

    held_out = {calib_season} | ({valid_season} if valid_season else set())
    train = df[~df["season_label"].isin(held_out)]
    calib = df[df["season_label"] == calib_season]
    valid = df[df["season_label"] == valid_season] if valid_season else df.iloc[0:0]

    if train.empty:
        raise ValueError("no training rows left — add more seasons")
    if calib.empty:
        raise ValueError(f"calibration season {calib_season} has no rows")

    logger.info(
        "train %6d rows (%s)",
        len(train),
        ", ".join(sorted(train["season_label"].unique())),
    )
    logger.info("calib %6d rows (%s)", len(calib), calib_season)
    if valid_season:
        logger.info("valid %6d rows (%s)", len(valid), valid_season)
    return train, calib, valid


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------


def baseline_predictions(train: pd.DataFrame, valid: pd.DataFrame) -> dict[str, np.ndarray]:
    """Two baselines the model must beat to justify existing.

    'prior'      — the overall class distribution, ignoring everything.
    'last_match' — the player's previous outcome carried forward. Deceptively
                   strong, and the real bar.
    """
    prior = train["y_minutes_class"].value_counts(normalize=True).sort_index()
    prior_pred = np.tile(prior.reindex([0, 1, 2]).fillna(0).values, (len(valid), 1))

    last_class = np.select(
        [valid["minutes_last"].fillna(0) == 0, valid["minutes_last"].fillna(0) < 60],
        [0, 1],
        default=2,
    )
    last_pred = np.full((len(valid), 3), 0.08)
    last_pred[np.arange(len(valid)), last_class] = 0.84

    return {"prior": prior_pred, "last_match": last_pred}


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Mean |predicted - actual| across probability buckets, weighted by count.

    One number to watch across model versions. Lower is better; under ~0.02 is
    good enough that calibration is no longer your limiting factor.
    """
    df = pd.DataFrame({"p": p, "y": y})
    df["bucket"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    g = df.groupby("bucket", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")
    )
    return float((g["n"] * (g["actual"] - g["predicted"]).abs()).sum() / g["n"].sum())


def evaluate(y_true: np.ndarray, proba: np.ndarray, label: str) -> dict:
    """Log loss, Brier, AUC and ECE.

    Accuracy is deliberately absent: predicting the majority class scores well
    on it and tells you nothing about whether 0.7 means 0.7.
    """
    p_appear = 1.0 - proba[:, 0]
    p60 = proba[:, 2]
    y_appear = (y_true > 0).astype(int)
    y60 = (y_true == 2).astype(int)

    m = {
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
        "brier_appear": float(brier_score_loss(y_appear, p_appear)),
        "brier_60": float(brier_score_loss(y60, p60)),
        "ece_60": expected_calibration_error(y60, p60),
        "ece_appear": expected_calibration_error(y_appear, p_appear),
    }
    if len(np.unique(y_appear)) > 1:
        m["auc_appear"] = float(roc_auc_score(y_appear, p_appear))
    if len(np.unique(y60)) > 1:
        m["auc_60"] = float(roc_auc_score(y60, p60))

    logger.info(
        "%-16s logloss=%.4f  brier60=%.4f  auc60=%.4f  ECE60=%.4f",
        label,
        m["log_loss"],
        m["brier_60"],
        m.get("auc_60", float("nan")),
        m["ece_60"],
    )
    return m


def calibration_table(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Of the players given p≈0.8, did ~80% actually do it?"""
    df = pd.DataFrame({"p": p, "y": y_true})
    df["bucket"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    out = (
        df.groupby("bucket", observed=True)
        .agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
        .round(3)
    )
    out["gap"] = (out["actual"] - out["predicted"]).round(3)
    return out


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def _wrap_calibrator(model, X_calib, y_calib):
    """Fit isotonic calibration on top of a pre-fitted booster.

    sklearn moved from cv="prefit" to FrozenEstimator around 1.6; support both
    so this doesn't break on the next dependency bump.

    Isotonic rather than sigmoid: the observed miscalibration wasn't a smooth
    S-curve, it was a roughly constant positive offset across the middle of the
    range, which a monotone step function fixes and a sigmoid only partly does.
    Isotonic wants a few thousand calibration rows, which a full season gives.
    """
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    except ImportError:
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_calib, y_calib)
    return calibrated


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def train_minutes_model(
    seasons: list[str],
    calib_season: str,
    valid_season: Optional[str] = None,
    params: Optional[dict] = None,
    save: bool = True,
) -> MinutesModel:
    import lightgbm as lgb

    df = build_training_frame(engine, seasons)
    assert_no_leakage(df)

    df = df[~df["position"].isin(EXCLUDED_POSITIONS)]
    df = df.dropna(subset=["y_minutes_class"])

    train, calib, valid = three_way_split(df, calib_season, valid_season)

    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    X_train, y_train = train[features].astype(float), train["y_minutes_class"].astype(int).values
    X_calib, y_calib = calib[features].astype(float), calib["y_minutes_class"].astype(int).values

    default_params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 100,  # high: resists overfitting to individuals
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "n_estimators": 2000,
        "verbosity": -1,
    }
    raw = lgb.LGBMClassifier(**{**default_params, **(params or {})})

    # early stopping uses the calibration season, not the validation season —
    # peeking at validation for stopping rounds is a subtle form of leakage
    raw.fit(
        X_train,
        y_train,
        eval_X=X_calib,
        eval_y=y_calib,
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    logger.info("best iteration: %d", raw.best_iteration_)

    calibrated = _wrap_calibrator(raw, X_calib, y_calib)
    logger.info("fitted isotonic calibration on %d rows", len(X_calib))

    metrics: dict = {}
    if valid_season and not valid.empty:
        X_valid = valid[features].astype(float)
        y_valid = valid["y_minutes_class"].astype(int).values

        proba_raw = raw.predict_proba(X_valid)
        proba_cal = calibrated.predict_proba(X_valid)

        metrics["raw"] = evaluate(y_valid, proba_raw, "uncalibrated")
        metrics["calibrated"] = evaluate(y_valid, proba_cal, "calibrated")
        for name, base in baseline_predictions(train, valid).items():
            metrics[name] = evaluate(y_valid, base, name)

        ece_gain = metrics["raw"]["ece_60"] - metrics["calibrated"]["ece_60"]
        ll_gain = metrics["last_match"]["log_loss"] - metrics["calibrated"]["log_loss"]
        logger.info("ECE improvement from calibration: %+.4f", ece_gain)
        logger.info("log loss vs last-match baseline:   %+.4f", ll_gain)

        if ll_gain <= 0:
            logger.warning(
                "model does not beat carrying the previous match forward — "
                "check features before building anything on top of this"
            )
        if ece_gain < 0:
            logger.warning(
                "calibration made things worse — the calibration season may be "
                "unrepresentative, or there are too few rows for isotonic"
            )

        print("\nP(60+) calibration BEFORE:")
        print(calibration_table((y_valid == 2).astype(int), proba_raw[:, 2]).to_string())
        print("\nP(60+) calibration AFTER:")
        print(calibration_table((y_valid == 2).astype(int), proba_cal[:, 2]).to_string())

    class_minutes = (
        train.groupby("y_minutes_class")["minutes"].mean().reindex([0, 1, 2]).fillna(0)
    )
    logger.info(
        "E[minutes|class]: 0=%.1f  1=%.1f  2=%.1f",
        class_minutes[0],
        class_minutes[1],
        class_minutes[2],
    )

    mm = MinutesModel(
        booster=calibrated,
        raw_booster=raw,
        features=features,
        class_minutes={int(k): float(v) for k, v in class_minutes.items()},
        metrics=metrics,
        trained_at=datetime.now(timezone.utc),
    )

    print("\nTop features:")
    print(mm.feature_importance(15).to_string())

    if save:
        _persist(mm, seasons, calib_season, valid_season, default_params)
    return mm


def _persist(
    mm: MinutesModel,
    seasons: list[str],
    calib_season: str,
    valid_season: Optional[str],
    params: dict,
) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{MODEL_NAME}_v{VERSION}.joblib"
    joblib.dump(mm, path)
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
                trained_at=mm.trained_at,
                config={
                    "params": params,
                    "seasons": seasons,
                    "calib_season": calib_season,
                    "valid_season": valid_season,
                    "features": mm.features,
                    "class_minutes": mm.class_minutes,
                    "calibration": "isotonic",
                },
                metrics=mm.metrics,
                git_sha=sha,
                note=f"3-class minutes model, isotonic-calibrated on {calib_season}",
            )
        )
        session.commit()
    logger.info("registered model run")


def load_model(version: str = VERSION) -> MinutesModel:
    return joblib.load(MODEL_DIR / f"{MODEL_NAME}_v{version}.joblib")


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Minutes model")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train")
    p.add_argument(
        "--seasons",
        nargs="+",
        default=["2022-23", "2023-24", "2024-25", "2025-26"],
    )
    p.add_argument("--calib", default="2024-25", help="season for the isotonic layer")
    p.add_argument("--valid", default="2025-26", help="season for honest metrics")
    p.add_argument(
        "--no-valid",
        action="store_true",
        help="production run: no holdout, use every season not used for calibration",
    )
    p.add_argument("--no-save", action="store_true")

    args = parser.parse_args()
    if args.cmd == "train":
        train_minutes_model(
            args.seasons,
            calib_season=args.calib,
            valid_season=None if args.no_valid else args.valid,
            save=not args.no_save,
        )


if __name__ == "__main__":
    # Run via the package module, not __main__, so pickled model classes are
    # saved as fantasy_26.predict.minutes.* and can be loaded from anywhere.
    from fantasy_26.predict import minutes

    minutes.main()
