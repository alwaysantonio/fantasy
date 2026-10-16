"""
Feature engineering.

THE ONE RULE: a feature for gameweek N may only use information that existed
before gameweek N's deadline. Every rolling window here is shifted by one row
within each player, and every snapshot join is an as-of join against the
deadline. Break this and you get a beautiful validation score and a useless
model.

Two entry points:

    build_training_frame(engine, seasons)  -> features + targets, all history
    build_prediction_frame(engine, gw)     -> features only, upcoming gameweek

Both produce one row per player per FIXTURE, not per gameweek. Double
gameweeks are therefore two rows; aggregate to gameweek level at the very end,
after prediction, by summing expected points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import Engine

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureConfig:
    """Window lengths and shrinkage priors.

    Shrinkage matters more than window length. A player with 180 minutes of
    data has an xG90 estimate that is mostly noise; pulling it toward a
    positional prior stops your model chasing small-sample flukes in August.
    """

    short_windows: tuple[int, ...] = (1, 3, 5)
    form_window: int = 6
    rate_window: int = 15  # appearances for xG/xA/defensive rates

    # prior "virtual minutes" of regression toward the positional mean
    prior_minutes: float = 270.0

    # per-90 priors by position id (1 GK, 2 DEF, 3 MID, 4 FWD)
    xg90_prior: dict[int, float] = field(
        default_factory=lambda: {1: 0.00, 2: 0.04, 3: 0.12, 4: 0.30}
    )
    xa90_prior: dict[int, float] = field(
        default_factory=lambda: {1: 0.00, 2: 0.06, 3: 0.14, 4: 0.12}
    )
    # Prior for defcon90, which counts CBIT for DEF but CBIRT (with
    # recoveries) for MID/FWD. Means per 90 for 60+ minute games in 2025-26,
    # the first season FPL published these stats.
    cbit90_prior: dict[int, float] = field(
        default_factory=lambda: {1: 0.0, 2: 7.5, 3: 7.9, 4: 4.1}
    )
    # per-90 rates over 2023-24..2025-26 appearances, by position
    bonus90_prior: dict[int, float] = field(
        default_factory=lambda: {1: 0.23, 2: 0.21, 3: 0.30, 4: 0.62}
    )
    yellows90_prior: dict[int, float] = field(
        default_factory=lambda: {1: 0.08, 2: 0.19, 3: 0.21, 4: 0.16}
    )


CONFIG = FeatureConfig()

POSITION_MAP = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4, "MNG": 5}


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

MATCH_SQL = """
SELECT
    pfs.player_season_id,
    ps.element_id,
    ps.player_id,
    p.web_name,
    s.label                AS season_label,
    pfs.fixture_id,
    pfs.gameweek_id,
    gw.number              AS gameweek,
    gw.deadline,
    gw.is_double,
    f.kickoff_time,
    pfs.was_home,
    ps.team_season_id,
    pfs.opponent_team_season_id,
    pfs.minutes,
    pfs.starts,
    pfs.goals_scored,
    pfs.assists,
    pfs.clean_sheets,
    pfs.goals_conceded,
    pfs.saves,
    pfs.bonus,
    pfs.bps,
    pfs.total_points,
    pfs.expected_goals,
    pfs.expected_assists,
    pfs.expected_goals_conceded,
    pfs.influence,
    pfs.creativity,
    pfs.threat,
    pfs.clearances_blocks_interceptions,
    pfs.tackles,
    pfs.recoveries,
    pfs.defensive_contribution,
    pfs.yellow_cards,
    pfs.red_cards,
    pfs.own_goals,
    pfs.penalties_missed
FROM player_fixture_stats pfs
JOIN player_seasons ps ON ps.id = pfs.player_season_id
JOIN players        p  ON p.id  = ps.player_id
JOIN gameweeks      gw ON gw.id = pfs.gameweek_id
JOIN fixtures       f  ON f.id  = pfs.fixture_id
JOIN seasons        s  ON s.id  = ps.season_id
WHERE s.label = ANY(%(seasons)s)
ORDER BY pfs.player_season_id, f.kickoff_time
"""

SNAPSHOT_SQL = """
SELECT
    snap.player_season_id,
    snap.captured_at,
    snap.position,
    snap.now_cost,
    snap.selected_by_percent,
    snap.transfers_in_event,
    snap.transfers_out_event,
    snap.status,
    snap.chance_of_playing_this_round,
    snap.chance_of_playing_next_round,
    snap.news_added,
    snap.form            AS fpl_form,
    snap.ep_next         AS fpl_ep_next,
    snap.penalties_order,
    snap.direct_freekicks_order,
    snap.corners_and_indirect_freekicks_order
FROM player_snapshots snap
JOIN player_seasons ps ON ps.id = snap.player_season_id
JOIN seasons        s  ON s.id  = ps.season_id
WHERE s.label = ANY(%(seasons)s)
ORDER BY snap.player_season_id, snap.captured_at
"""

UPCOMING_SQL = """
SELECT
    ps.id            AS player_season_id,
    ps.element_id,
    ps.player_id,
    p.web_name,
    s.label          AS season_label,
    f.id             AS fixture_id,
    gw.id            AS gameweek_id,
    gw.number        AS gameweek,
    gw.deadline,
    gw.is_double,
    f.kickoff_time,
    (f.home_team_season_id = ps.team_season_id) AS was_home,
    ps.team_season_id,
    CASE WHEN f.home_team_season_id = ps.team_season_id
         THEN f.away_team_season_id ELSE f.home_team_season_id
    END AS opponent_team_season_id
FROM player_seasons ps
JOIN players   p  ON p.id = ps.player_id
JOIN seasons   s  ON s.id = ps.season_id
JOIN fixtures  f  ON (f.home_team_season_id = ps.team_season_id
                   OR f.away_team_season_id = ps.team_season_id)
JOIN gameweeks gw ON gw.id = f.gameweek_id
WHERE s.label = ANY(%(seasons)s) AND gw.number = %(gw)s
"""


def fetch_matches(engine: Engine, seasons: list[str]) -> pd.DataFrame:
    df = pd.read_sql(MATCH_SQL, engine, params={"seasons": seasons})
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    df["deadline"] = pd.to_datetime(df["deadline"], utc=True)
    return df.sort_values(["player_season_id", "kickoff_time"]).reset_index(drop=True)


def fetch_snapshots(engine: Engine, seasons: list[str]) -> pd.DataFrame:
    df = pd.read_sql(SNAPSHOT_SQL, engine, params={"seasons": seasons})
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    df["news_added"] = pd.to_datetime(df["news_added"], utc=True)
    return df.sort_values(["player_season_id", "captured_at"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# derived columns
# --------------------------------------------------------------------------


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Columns computed from a single match row — no lookback, so no shift
    needed yet."""
    df = df.copy()
    df["cbit"] = df["clearances_blocks_interceptions"] + df["tackles"]
    df["cbirt"] = df["cbit"] + df["recoveries"]
    df["played"] = (df["minutes"] > 0).astype(int)
    df["played_60"] = (df["minutes"] >= 60).astype(int)
    df["started"] = (df["starts"] > 0).astype(int)

    # DefCon thresholds differ by position: 10 CBIT for DEF, 12 CBIRT for MID/FWD
    df["defcon_hit"] = np.where(
        df["position"] == 2,
        (df["cbit"] >= 10).astype(int),
        np.where(df["position"].isin([3, 4]), (df["cbirt"] >= 12).astype(int), 0),
    )
    df["defcon_metric"] = np.where(df["position"] == 2, df["cbit"], df["cbirt"])
    return df


# --------------------------------------------------------------------------
# rolling features (the point-in-time critical section)
# --------------------------------------------------------------------------


def _lagged_roll(
    g: pd.core.groupby.SeriesGroupBy,
    window: int,
    how: str = "mean",
) -> pd.Series:
    """Shift by one THEN roll. Row N sees rows N-window .. N-1, never row N.

    This single shift is the difference between a real model and a leak.
    """
    shifted = g.shift(1)
    roller = shifted.rolling(window, min_periods=1)
    return getattr(roller, how)()


def add_rolling_features(df: pd.DataFrame, cfg: FeatureConfig = CONFIG) -> pd.DataFrame:
    """All lookback features. Assumes df is sorted by player then kickoff.

    Note that grouping on player_season_id (unique per player PER SEASON) means
    rolling windows reset at season boundaries for free — last August's form
    never bleeds into this May's rows.
    """
    df = df.sort_values(["player_season_id", "kickoff_time"]).reset_index(drop=True)
    g = df.groupby("player_season_id", sort=False)

    # --- minutes: the highest-value feature block in the whole model ---
    for w in cfg.short_windows:
        df[f"minutes_mean_{w}"] = _lagged_roll(g["minutes"], w)
        df[f"started_rate_{w}"] = _lagged_roll(g["started"], w)
        df[f"played60_rate_{w}"] = _lagged_roll(g["played_60"], w)

    df[f"minutes_mean_{cfg.form_window}"] = _lagged_roll(g["minutes"], cfg.form_window)
    df["minutes_last"] = g["minutes"].shift(1)
    df["started_last"] = g["started"].shift(1)
    df["appearances_to_date"] = g["played"].cumsum().shift(1).fillna(0)

    # consecutive starts — strong signal of a nailed player
    started = df["started"].fillna(0)
    blocks = (started != started.shift(1)).cumsum()
    df["consecutive_starts"] = (
        started.groupby([df["player_season_id"], blocks]).cumcount() + 1
    ) * started
    df["consecutive_starts"] = g["consecutive_starts"].shift(1).fillna(0)

    # --- per-90 rates, minutes-weighted and shrunk (bonus and cards too) ---
    cum_min = g["minutes"].cumsum().shift(1).fillna(0)
    for src, out, prior_map in (
        ("expected_goals", "xg90", cfg.xg90_prior),
        ("expected_assists", "xa90", cfg.xa90_prior),
        ("bonus", "bonus90", cfg.bonus90_prior),
        ("yellow_cards", "yellows90", cfg.yellows90_prior),
    ):
        cum_val = g[src].cumsum().shift(1).fillna(0)
        prior = df["position"].map(prior_map).fillna(0.0)
        df[out] = (
            (cum_val + prior * cfg.prior_minutes / 90.0)
            / (cum_min + cfg.prior_minutes)
            * 90.0
        )

    df["goals_mean_5"] = _lagged_roll(g["goals_scored"], 5)
    df["assists_mean_5"] = _lagged_roll(g["assists"], 5)
    df["threat_mean_5"] = _lagged_roll(g["threat"], 5)
    df["creativity_mean_5"] = _lagged_roll(g["creativity"], 5)

    # --- defensive contribution ---
    cum_defcon = g["defcon_metric"].cumsum().shift(1).fillna(0)
    prior_cbit = df["position"].map(cfg.cbit90_prior).fillna(0.0)
    df["defcon90"] = (
        (cum_defcon + prior_cbit * cfg.prior_minutes / 90.0)
        / (cum_min + cfg.prior_minutes)
        * 90.0
    )
    df["defcon_hit_rate"] = _lagged_roll(g["defcon_hit"], cfg.rate_window)

    # --- goalkeeping ---
    df["saves_mean_5"] = _lagged_roll(g["saves"], 5)
    df["conceded_mean_5"] = _lagged_roll(g["goals_conceded"], 5)

    # --- bonus / bps ---
    df["bps_mean_5"] = _lagged_roll(g["bps"], 5)
    df["bonus_mean_5"] = _lagged_roll(g["bonus"], 5)

    # --- points ---
    df["points_mean_5"] = _lagged_roll(g["total_points"], 5)
    df["points_std_5"] = _lagged_roll(g["total_points"], 5, how="std")

    # --- discipline ---
    df["yellows_mean_10"] = _lagged_roll(g["yellow_cards"], 10)

    return df


def add_fixture_context(df: pd.DataFrame) -> pd.DataFrame:
    """Rest, congestion, and venue. Rotation risk is largely a function of
    these, so they feed the minutes model directly."""
    df = df.sort_values(["player_season_id", "kickoff_time"]).reset_index(drop=True)
    g = df.groupby("player_season_id", sort=False)

    prev_kickoff = g["kickoff_time"].shift(1)
    df["days_rest"] = (df["kickoff_time"] - prev_kickoff).dt.total_seconds() / 86400
    df["days_rest"] = df["days_rest"].clip(upper=30).fillna(14)
    df["short_rest"] = (df["days_rest"] < 4).astype(int)

    # fixtures in the following 10 days — congestion drives rotation
    def _congestion(sub: pd.DataFrame) -> pd.Series:
        times = sub["kickoff_time"]
        return pd.Series(
            [
                int(((times > t) & (times <= t + pd.Timedelta(days=10))).sum())
                for t in times
            ],
            index=sub.index,
        )

    df["fixtures_next_10d"] = (
        df.groupby("team_season_id", group_keys=False).apply(_congestion).astype(int)
    )

    df["is_home"] = df["was_home"].astype(int)
    df["is_dgw"] = df["is_double"].astype(int)
    return df


def attach_snapshots(matches: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """As-of join: for each match, take the most recent snapshot captured
    strictly BEFORE that gameweek's deadline.

    This is where injury news, price and availability enter the feature set —
    and the single place most likely to leak if done carelessly. merge_asof
    with direction='backward' guarantees we never see the future.
    """
    left = matches.sort_values("deadline").reset_index(drop=True)
    right = snaps.sort_values("captured_at").reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        left_on="deadline",
        right_on="captured_at",
        by="player_season_id",
        direction="backward",
        allow_exact_matches=False,
    )

    merged["snapshot_age_hours"] = (
        merged["deadline"] - merged["captured_at"]
    ).dt.total_seconds() / 3600

    # availability: 100 when FPL says nothing, which is the common case
    merged["chance_next"] = merged["chance_of_playing_next_round"].fillna(100.0)
    merged["is_flagged"] = (merged["status"] != "AVAILABLE").astype(int)
    merged["news_age_days"] = (
        (merged["deadline"] - merged["news_added"]).dt.total_seconds() / 86400
    ).fillna(999)

    merged["net_transfers"] = (
        merged["transfers_in_event"].fillna(0) - merged["transfers_out_event"].fillna(0)
    )
    merged["price"] = merged["now_cost"].fillna(45) / 10.0
    merged["is_penalty_taker"] = (merged["penalties_order"] == 1).astype(int)
    merged["takes_freekicks"] = (
        merged["direct_freekicks_order"].fillna(9) <= 2
    ).astype(int)

    return merged


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale-free versions of the transfer-market signals.

    Raw net transfers is dominated by squad size: a player owned by two million
    managers posts huge numbers whatever the news, so the raw column partly
    measures popularity rather than information. Ranking within the gameweek
    strips the scale out and leaves the ordering, which is the part that
    actually reflects what managers learned from press conferences.

    Must run AFTER attach_snapshots, which is where net_transfers is created.
    """
    df = df.copy()
    df["net_transfers_rank"] = (
        df.groupby("gameweek_id")["net_transfers"].rank(pct=True).fillna(0.5)
    )
    df["transfers_in_rank"] = (
        df.groupby("gameweek_id")["transfers_in_event"]
        .rank(pct=True)
        .fillna(0.5)
    )
    return df


def map_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Position comes from the snapshot, not the registration, because FPL
    reclassifies players mid-season and we want what was true at that deadline.

    Stored as the enum NAME (native_enum=False), so it arrives as a string.
    """
    df = df.copy()
    df["position"] = df["position"].map(POSITION_MAP).fillna(3).astype(int)
    return df


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Targets for each sub-model. These are the CURRENT row's outcomes —
    unshifted, obviously, since that's what we're predicting."""
    df = df.copy()

    # minutes model: three-class, not a regression
    df["y_minutes_class"] = np.select(
        [df["minutes"] == 0, df["minutes"] < 60], [0, 1], default=2
    )
    df["y_played"] = df["played"]
    df["y_played_60"] = df["played_60"]

    df["y_goals"] = df["goals_scored"]
    df["y_assists"] = df["assists"]
    df["y_clean_sheet"] = df["clean_sheets"]
    df["y_defcon"] = df["defcon_hit"]
    df["y_saves"] = df["saves"]
    df["y_bps"] = df["bps"]
    df["y_bonus"] = df["bonus"]
    df["y_points"] = df["total_points"]
    return df


FEATURE_COLUMNS = [
    "position",
    "is_home",
    "is_dgw",
    "price",
    "selected_by_percent",
    "net_transfers",
    "net_transfers_rank",
    "transfers_in_rank",
    # minutes
    "minutes_mean_1",
    "minutes_mean_3",
    "minutes_mean_5",
    "minutes_mean_6",
    "minutes_last",
    "started_last",
    "started_rate_3",
    "started_rate_5",
    "played60_rate_3",
    "played60_rate_5",
    "consecutive_starts",
    "appearances_to_date",
    # availability
    "chance_next",
    "is_flagged",
    "news_age_days",
    # context
    "days_rest",
    "short_rest",
    "fixtures_next_10d",
    # attacking
    "xg90",
    "xa90",
    "goals_mean_5",
    "assists_mean_5",
    "threat_mean_5",
    "creativity_mean_5",
    "is_penalty_taker",
    "takes_freekicks",
    # defensive
    "defcon90",
    "defcon_hit_rate",
    "saves_mean_5",
    "conceded_mean_5",
    # bonus
    "bps_mean_5",
    "bonus_mean_5",
    # form
    "points_mean_5",
    "points_std_5",
    "yellows_mean_10",
]


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def build_training_frame(
    engine: Engine,
    seasons: list[str],
    cfg: FeatureConfig = CONFIG,
    min_appearances: int = 3,
) -> pd.DataFrame:
    """Full historical feature matrix with targets.

    `min_appearances` drops each player's first few matches, where every
    rolling feature is dominated by the prior rather than by data. Keeping
    them adds noise without adding signal.
    """
    matches = fetch_matches(engine, seasons)
    snaps = fetch_snapshots(engine, seasons)

    if matches.empty:
        raise ValueError(f"no match data for seasons {seasons} — run the ingest first")

    df = attach_snapshots(matches, snaps)
    df = map_positions(df)
    df = add_market_features(df)
    df = add_derived(df)
    df = add_rolling_features(df, cfg)
    df = add_fixture_context(df)
    df = add_targets(df)

    df = df[df["appearances_to_date"] >= min_appearances]
    logger.info("training frame: %d rows, %d features", len(df), len(FEATURE_COLUMNS))
    return df


def build_prediction_frame(
    engine: Engine,
    seasons: list[str],
    target_gameweek: int,
    cfg: FeatureConfig = CONFIG,
) -> pd.DataFrame:
    """Features for an upcoming gameweek.

    The awkward bit: no player_fixture_stats rows exist for a match that
    hasn't been played, so we synthesise placeholder rows from the fixture
    list, append them to history, and run the same rolling pipeline. The
    lagged windows then fill correctly because the placeholder is last.
    """
    history = fetch_matches(engine, seasons)
    snaps = fetch_snapshots(engine, seasons)
    upcoming = pd.read_sql(
        UPCOMING_SQL, engine, params={"seasons": seasons, "gw": target_gameweek}
    )

    if upcoming.empty:
        raise ValueError(f"no fixtures found for GW{target_gameweek}")

    upcoming["kickoff_time"] = pd.to_datetime(upcoming["kickoff_time"], utc=True)
    upcoming["deadline"] = pd.to_datetime(upcoming["deadline"], utc=True)

    # zero-fill the stat columns so the shapes line up
    for col in history.columns:
        if col not in upcoming.columns:
            upcoming[col] = 0

    combined = pd.concat([history, upcoming[history.columns]], ignore_index=True)

    df = attach_snapshots(combined, snaps)
    df = map_positions(df)
    df = add_market_features(df)
    df = add_derived(df)
    df = add_rolling_features(df, cfg)
    df = add_fixture_context(df)

    out = df[df["gameweek"] == target_gameweek].copy()
    logger.info(
        "prediction frame: %d player-fixtures for GW%d", len(out), target_gameweek
    )
    return out


# --------------------------------------------------------------------------
# leakage check
# --------------------------------------------------------------------------


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Cheap guard against the mistake that silently ruins everything.

    If a rolling feature correlates almost perfectly with its own target,
    the shift is missing somewhere. Run this every time you add a feature.
    """
    checks = [
        ("minutes_mean_1", "minutes"),
        ("points_mean_5", "total_points"),
        ("bps_mean_5", "bps"),
    ]
    for feat, target in checks:
        if feat not in df or target not in df:
            continue
        sub = df[[feat, target]].dropna()
        if len(sub) < 100:
            continue
        corr = sub[feat].corr(sub[target])
        if corr > 0.95:
            raise AssertionError(
                f"LEAK: {feat} correlates {corr:.3f} with {target} — check the shift"
            )
    # snapshots must predate the deadline
    if "snapshot_age_hours" in df:
        bad = (df["snapshot_age_hours"] < 0).sum()
        if bad:
            raise AssertionError(f"LEAK: {bad} snapshots captured after the deadline")
    logger.info("leakage checks passed")