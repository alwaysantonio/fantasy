"""
Database schema for an FPL prediction + optimisation app.

Design principles
-----------------
1. POINT-IN-TIME CORRECTNESS. Anything that mutates during a season (price,
   availability, news, ownership) is APPENDED to a snapshot table, never
   updated in place. Without this you cannot backtest honestly, because you
   will accidentally train on information that did not exist at the deadline.

2. STABLE IDENTITY. FPL reassigns `id` for players and teams every season.
   The `code` field is stable across seasons, so it is the natural key here.
   Season-specific ids live on the *Season link tables.

3. RAW ARCHIVE. Every ingest writes the untouched JSON to RawIngest. When you
   inevitably realise you needed a field you didn't parse, you can rebuild
   without having lost the history.

Target: SQLAlchemy 2.0 with SQLite (or Postgres — swap JSON for JSONB).
"""

from __future__ import annotations

import enum
from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class Position(enum.IntEnum):
    GK = 1
    DEF = 2
    MID = 3
    FWD = 4
    MNG = 5  # managers, added as a scoring "position"


class Availability(enum.Enum):
    AVAILABLE = "a"
    DOUBTFUL = "d"
    INJURED = "i"
    SUSPENDED = "s"
    UNAVAILABLE = "u"
    NOT_IN_SQUAD = "n"


# --------------------------------------------------------------------------
# Reference / dimension tables
# --------------------------------------------------------------------------


class Season(Base):
    """e.g. '2026-27'. Everything season-scoped hangs off this."""

    __tablename__ = "seasons"

    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(String(9), unique=True)  # "2026-27"
    start_year: Mapped[int] = mapped_column(Integer)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)

    gameweeks: Mapped[list["Gameweek"]] = relationship(back_populates="season")


class Gameweek(Base):
    """One row per GW per season. Deadline is the critical field — it defines
    the information cutoff for every feature you build."""

    __tablename__ = "gameweeks"
    __table_args__ = (UniqueConstraint("season_id", "number"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    season_id: Mapped[int] = mapped_column(ForeignKey("seasons.id"), index=True)
    number: Mapped[int] = mapped_column(Integer)  # 1..38

    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    finished: Mapped[bool] = mapped_column(Boolean, default=False)
    data_checked: Mapped[bool] = mapped_column(Boolean, default=False)

    # Useful context for the minutes model
    is_blank: Mapped[bool] = mapped_column(Boolean, default=False)
    is_double: Mapped[bool] = mapped_column(Boolean, default=False)

    average_entry_score: Mapped[Optional[int]] = mapped_column(Integer)
    highest_score: Mapped[Optional[int]] = mapped_column(Integer)
    chip_plays: Mapped[Optional[dict]] = mapped_column(JSON)

    season: Mapped["Season"] = relationship(back_populates="gameweeks")


class Team(Base):
    """Canonical club identity. `code` is stable across seasons; the per-season
    id (1..20, assigned alphabetically) lives on TeamSeason."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(64))
    short_name: Mapped[str] = mapped_column(String(8))


class TeamSeason(Base):
    __tablename__ = "team_seasons"
    __table_args__ = (
        UniqueConstraint("season_id", "team_id"),
        UniqueConstraint("season_id", "fpl_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    season_id: Mapped[int] = mapped_column(ForeignKey("seasons.id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    fpl_id: Mapped[int] = mapped_column(Integer)  # the 1..20 id for this season

    # FPL's own strength ratings — weak, but free features
    strength_overall_home: Mapped[Optional[int]] = mapped_column(Integer)
    strength_overall_away: Mapped[Optional[int]] = mapped_column(Integer)
    strength_attack_home: Mapped[Optional[int]] = mapped_column(Integer)
    strength_attack_away: Mapped[Optional[int]] = mapped_column(Integer)
    strength_defence_home: Mapped[Optional[int]] = mapped_column(Integer)
    strength_defence_away: Mapped[Optional[int]] = mapped_column(Integer)

    # Set once you know it — drives the rotation features in the minutes model
    in_europe: Mapped[bool] = mapped_column(Boolean, default=False)
    european_competition: Mapped[Optional[str]] = mapped_column(String(32))


class Player(Base):
    """Canonical player identity across seasons. `code` (the photo id) is
    stable; `opta_code` is a useful join key to external data sources."""

    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    opta_code: Mapped[Optional[str]] = mapped_column(String(16), index=True)

    first_name: Mapped[str] = mapped_column(String(64))
    second_name: Mapped[str] = mapped_column(String(64))
    web_name: Mapped[str] = mapped_column(String(64), index=True)

    date_of_birth: Mapped[Optional[date]] = mapped_column(Date)
    # Optional external join keys — worth populating if you pull FBref/Understat
    fbref_id: Mapped[Optional[str]] = mapped_column(String(16))
    understat_id: Mapped[Optional[str]] = mapped_column(String(16))


class PlayerSeason(Base):
    """Player's registration for one season: their season-specific element id,
    club, position and starting price.

    Position can change mid-season (FPL reclassifies players), so treat
    `position` here as the season-opening value and trust the per-GW snapshot
    for what was true at any given deadline.
    """

    __tablename__ = "player_seasons"
    __table_args__ = (
        UniqueConstraint("season_id", "player_id"),
        UniqueConstraint("season_id", "element_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    season_id: Mapped[int] = mapped_column(ForeignKey("seasons.id"), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    element_id: Mapped[int] = mapped_column(Integer)  # FPL id within this season

    team_season_id: Mapped[int] = mapped_column(ForeignKey("team_seasons.id"), index=True)
    position: Mapped[Position] = mapped_column(
        Enum(Position, native_enum=False, length=16)
    )
    start_price: Mapped[int] = mapped_column(Integer)  # tenths of £m: 125 == £12.5m

    squad_number: Mapped[Optional[int]] = mapped_column(Integer)


class Fixture(Base):
    __tablename__ = "fixtures"
    __table_args__ = (UniqueConstraint("season_id", "fpl_fixture_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    season_id: Mapped[int] = mapped_column(ForeignKey("seasons.id"), index=True)
    fpl_fixture_id: Mapped[int] = mapped_column(Integer)
    gameweek_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("gameweeks.id"), index=True
    )  # nullable: postponed fixtures have no GW

    kickoff_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), index=True
    )
    home_team_season_id: Mapped[int] = mapped_column(ForeignKey("team_seasons.id"))
    away_team_season_id: Mapped[int] = mapped_column(ForeignKey("team_seasons.id"))

    home_score: Mapped[Optional[int]] = mapped_column(Integer)
    away_score: Mapped[Optional[int]] = mapped_column(Integer)
    finished: Mapped[bool] = mapped_column(Boolean, default=False)
    minutes_played: Mapped[int] = mapped_column(Integer, default=0)

    # FPL's FDR — keep it for comparison, don't rely on it
    home_difficulty: Mapped[Optional[int]] = mapped_column(Integer)
    away_difficulty: Mapped[Optional[int]] = mapped_column(Integer)

    # Post-match xG, if you source it. Better Dixon-Coles input than goals.
    home_xg: Mapped[Optional[float]] = mapped_column(Float)
    away_xg: Mapped[Optional[float]] = mapped_column(Float)


# --------------------------------------------------------------------------
# Fact table: what actually happened
# --------------------------------------------------------------------------


class PlayerFixtureStat(Base):
    """One row per player per fixture. Keyed on fixture rather than gameweek so
    double gameweeks work naturally — aggregate up to GW level in a query, not
    in the schema.

    This is your training target and your feature source. Populated after each
    match from element-summary/{id}/ once `data_checked` is true for the GW.
    """

    __tablename__ = "player_fixture_stats"
    __table_args__ = (
        UniqueConstraint("player_season_id", "fixture_id"),
        Index("ix_pfs_gw", "gameweek_id", "player_season_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    player_season_id: Mapped[int] = mapped_column(
        ForeignKey("player_seasons.id"), index=True
    )
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id"), index=True)
    gameweek_id: Mapped[int] = mapped_column(ForeignKey("gameweeks.id"), index=True)

    was_home: Mapped[bool] = mapped_column(Boolean)
    opponent_team_season_id: Mapped[int] = mapped_column(ForeignKey("team_seasons.id"))

    # --- minutes ---
    minutes: Mapped[int] = mapped_column(Integer, default=0)
    starts: Mapped[int] = mapped_column(Integer, default=0)

    # --- scoring events ---
    goals_scored: Mapped[int] = mapped_column(Integer, default=0)
    assists: Mapped[int] = mapped_column(Integer, default=0)
    clean_sheets: Mapped[int] = mapped_column(Integer, default=0)
    goals_conceded: Mapped[int] = mapped_column(Integer, default=0)
    own_goals: Mapped[int] = mapped_column(Integer, default=0)
    penalties_saved: Mapped[int] = mapped_column(Integer, default=0)
    penalties_missed: Mapped[int] = mapped_column(Integer, default=0)
    yellow_cards: Mapped[int] = mapped_column(Integer, default=0)
    red_cards: Mapped[int] = mapped_column(Integer, default=0)
    saves: Mapped[int] = mapped_column(Integer, default=0)

    # --- defensive contributions (2025/26 onward) ---
    clearances_blocks_interceptions: Mapped[int] = mapped_column(Integer, default=0)
    tackles: Mapped[int] = mapped_column(Integer, default=0)
    recoveries: Mapped[int] = mapped_column(Integer, default=0)
    defensive_contribution: Mapped[int] = mapped_column(Integer, default=0)  # 0 or 2

    # --- bonus ---
    bonus: Mapped[int] = mapped_column(Integer, default=0)
    bps: Mapped[int] = mapped_column(Integer, default=0)

    total_points: Mapped[int] = mapped_column(Integer, default=0)

    # --- underlying / ICT ---
    influence: Mapped[float] = mapped_column(Float, default=0.0)
    creativity: Mapped[float] = mapped_column(Float, default=0.0)
    threat: Mapped[float] = mapped_column(Float, default=0.0)
    ict_index: Mapped[float] = mapped_column(Float, default=0.0)
    expected_goals: Mapped[float] = mapped_column(Float, default=0.0)
    expected_assists: Mapped[float] = mapped_column(Float, default=0.0)
    expected_goal_involvements: Mapped[float] = mapped_column(Float, default=0.0)
    expected_goals_conceded: Mapped[float] = mapped_column(Float, default=0.0)

    # --- market state at the time of this match ---
    value: Mapped[Optional[int]] = mapped_column(Integer)  # price in tenths
    selected: Mapped[Optional[int]] = mapped_column(Integer)
    transfers_in: Mapped[Optional[int]] = mapped_column(Integer)
    transfers_out: Mapped[Optional[int]] = mapped_column(Integer)

    @property
    def cbit(self) -> int:
        """Defender DefCon metric: threshold 10."""
        return self.clearances_blocks_interceptions + self.tackles

    @property
    def cbirt(self) -> int:
        """Midfielder/forward DefCon metric: threshold 12."""
        return self.cbit + self.recoveries


# --------------------------------------------------------------------------
# Point-in-time snapshots (append-only)
# --------------------------------------------------------------------------


class PlayerSnapshot(Base):
    """Daily append-only capture of everything mutable on a player.

    This is the table that makes honest backtesting possible: to build features
    for GW n, filter to `captured_at < gameweek.deadline`. ~700 rows/day is
    trivial for SQLite.
    """

    __tablename__ = "player_snapshots"
    __table_args__ = (
        UniqueConstraint("player_season_id", "captured_at"),
        Index("ix_snapshot_time_brin", "captured_at", postgresql_using="brin"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    player_season_id: Mapped[int] = mapped_column(
        ForeignKey("player_seasons.id"), index=True
    )
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    gameweek_id: Mapped[Optional[int]] = mapped_column(ForeignKey("gameweeks.id"))

    # Position and club can both change mid-season
    position: Mapped[Position] = mapped_column(Enum(Position, native_enum=False, length=16))
    team_season_id: Mapped[int] = mapped_column(ForeignKey("team_seasons.id"))

    # --- market ---
    now_cost: Mapped[int] = mapped_column(Integer)
    cost_change_event: Mapped[int] = mapped_column(Integer, default=0)
    selected_by_percent: Mapped[float] = mapped_column(Float, default=0.0)
    transfers_in_event: Mapped[int] = mapped_column(Integer, default=0)
    transfers_out_event: Mapped[int] = mapped_column(Integer, default=0)

    # --- availability: the highest-value fields in this table ---
    status: Mapped[Availability] = mapped_column(
        Enum(Availability, native_enum=False, length=16)
    )
    chance_of_playing_this_round: Mapped[Optional[int]] = mapped_column(Integer)
    chance_of_playing_next_round: Mapped[Optional[int]] = mapped_column(Integer)
    news: Mapped[Optional[str]] = mapped_column(Text)
    news_added: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # --- FPL's own numbers, for baseline comparison ---
    form: Mapped[Optional[float]] = mapped_column(Float)
    points_per_game: Mapped[Optional[float]] = mapped_column(Float)
    ep_this: Mapped[Optional[float]] = mapped_column(Float)
    ep_next: Mapped[Optional[float]] = mapped_column(Float)

    # --- FPL-supplied set-piece hints (often stale; see SetPieceDuty) ---
    penalties_order: Mapped[Optional[int]] = mapped_column(Integer)
    corners_and_indirect_freekicks_order: Mapped[Optional[int]] = mapped_column(Integer)
    direct_freekicks_order: Mapped[Optional[int]] = mapped_column(Integer)


class PriceChange(Base):
    """Derived from consecutive snapshots. Materialised because the price-change
    model trains on it directly and the self-join is annoying."""

    __tablename__ = "price_changes"
    __table_args__ = (UniqueConstraint("player_season_id", "changed_on"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    player_season_id: Mapped[int] = mapped_column(
        ForeignKey("player_seasons.id"), index=True
    )
    changed_on: Mapped[date] = mapped_column(Date, index=True)
    old_cost: Mapped[int] = mapped_column(Integer)
    new_cost: Mapped[int] = mapped_column(Integer)
    delta: Mapped[int] = mapped_column(Integer)  # +1 / -1 in tenths

    # Features as they stood immediately before the change
    net_transfers_24h: Mapped[Optional[int]] = mapped_column(Integer)
    ownership_at_change: Mapped[Optional[float]] = mapped_column(Float)


class SetPieceDuty(Base):
    """Manual override table. FPL's own set-piece ordering lags reality by weeks
    and this is one of the highest-value features you have — a nailed penalty
    taker is worth roughly +0.15 xG per match. Curate it by hand.
    """

    __tablename__ = "set_piece_duties"

    id: Mapped[int] = mapped_column(primary_key=True)
    player_season_id: Mapped[int] = mapped_column(
        ForeignKey("player_seasons.id"), index=True
    )
    effective_from: Mapped[date] = mapped_column(Date, index=True)
    effective_to: Mapped[Optional[date]] = mapped_column(Date)

    penalty_order: Mapped[Optional[int]] = mapped_column(Integer)  # 1 = first choice
    direct_fk_order: Mapped[Optional[int]] = mapped_column(Integer)
    corner_share: Mapped[Optional[float]] = mapped_column(Float)  # 0..1

    source: Mapped[Optional[str]] = mapped_column(String(128))
    note: Mapped[Optional[str]] = mapped_column(Text)


# --------------------------------------------------------------------------
# Model outputs
# --------------------------------------------------------------------------


class ModelRun(Base):
    """Every set of predictions is tagged with the run that produced it, so you
    can compare model versions against each other on identical history."""

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), index=True)  # "minutes_lgbm"
    version: Mapped[str] = mapped_column(String(32))
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # Guard against leakage: no training data after this instant
    train_cutoff: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    config: Mapped[Optional[dict]] = mapped_column(JSON)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON)
    git_sha: Mapped[Optional[str]] = mapped_column(String(40))
    note: Mapped[Optional[str]] = mapped_column(Text)


class TeamStrength(Base):
    """Fitted Dixon-Coles / Poisson coefficients. Time-varying, so store the
    as-of date and always read the latest row before the deadline."""

    __tablename__ = "team_strengths"
    __table_args__ = (UniqueConstraint("model_run_id", "team_season_id", "as_of"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"), index=True)
    team_season_id: Mapped[int] = mapped_column(ForeignKey("team_seasons.id"), index=True)
    as_of: Mapped[date] = mapped_column(Date, index=True)

    attack: Mapped[float] = mapped_column(Float)
    defence: Mapped[float] = mapped_column(Float)
    home_advantage: Mapped[Optional[float]] = mapped_column(Float)
    rho: Mapped[Optional[float]] = mapped_column(Float)  # Dixon-Coles correction


class FixtureProjection(Base):
    """Per-fixture team-level output of the strength model. Feeds clean sheet
    probability, expected goals conceded, and the attacking multiplier."""

    __tablename__ = "fixture_projections"
    __table_args__ = (UniqueConstraint("model_run_id", "fixture_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"), index=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id"), index=True)

    lambda_home: Mapped[float] = mapped_column(Float)
    lambda_away: Mapped[float] = mapped_column(Float)
    p_home_clean_sheet: Mapped[float] = mapped_column(Float)
    p_away_clean_sheet: Mapped[float] = mapped_column(Float)
    p_home_win: Mapped[Optional[float]] = mapped_column(Float)
    p_draw: Mapped[Optional[float]] = mapped_column(Float)
    p_away_win: Mapped[Optional[float]] = mapped_column(Float)


class Prediction(Base):
    """Expected points for one player in one gameweek, with the full component
    breakdown retained.

    Store components, not just the total. When your xP is wrong you need to know
    *which* sub-model was wrong, and a single number tells you nothing.

    `horizon` = 0 for the upcoming GW, 1..5 for future GWs (the optimiser needs
    a multi-week horizon to make sensible transfer decisions).
    """

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("model_run_id", "player_season_id", "gameweek_id"),
        Index("ix_pred_gw_xp", "gameweek_id", "expected_points"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"), index=True)
    player_season_id: Mapped[int] = mapped_column(
        ForeignKey("player_seasons.id"), index=True
    )
    gameweek_id: Mapped[int] = mapped_column(ForeignKey("gameweeks.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    horizon: Mapped[int] = mapped_column(Integer, default=0)

    # --- minutes model ---
    p_appear: Mapped[float] = mapped_column(Float)
    p_60: Mapped[float] = mapped_column(Float)
    expected_minutes: Mapped[float] = mapped_column(Float)

    # --- attacking ---
    expected_goals: Mapped[float] = mapped_column(Float, default=0.0)
    expected_assists: Mapped[float] = mapped_column(Float, default=0.0)

    # --- defensive ---
    p_clean_sheet: Mapped[float] = mapped_column(Float, default=0.0)
    expected_goals_conceded: Mapped[float] = mapped_column(Float, default=0.0)
    p_defcon: Mapped[float] = mapped_column(Float, default=0.0)
    expected_saves: Mapped[float] = mapped_column(Float, default=0.0)

    # --- bonus & negatives ---
    expected_bps: Mapped[float] = mapped_column(Float, default=0.0)
    p_bonus_3: Mapped[float] = mapped_column(Float, default=0.0)
    p_bonus_2: Mapped[float] = mapped_column(Float, default=0.0)
    p_bonus_1: Mapped[float] = mapped_column(Float, default=0.0)
    expected_negatives: Mapped[float] = mapped_column(Float, default=0.0)

    # --- points, decomposed then summed ---
    pts_appearance: Mapped[float] = mapped_column(Float, default=0.0)
    pts_goals: Mapped[float] = mapped_column(Float, default=0.0)
    pts_assists: Mapped[float] = mapped_column(Float, default=0.0)
    pts_clean_sheet: Mapped[float] = mapped_column(Float, default=0.0)
    pts_defcon: Mapped[float] = mapped_column(Float, default=0.0)
    pts_saves: Mapped[float] = mapped_column(Float, default=0.0)
    pts_bonus: Mapped[float] = mapped_column(Float, default=0.0)
    pts_conceded: Mapped[float] = mapped_column(Float, default=0.0)
    pts_negatives: Mapped[float] = mapped_column(Float, default=0.0)

    expected_points: Mapped[float] = mapped_column(Float, index=True)
    # Spread matters for captaincy: prefer high-variance players when chasing
    points_std: Mapped[Optional[float]] = mapped_column(Float)

    # Backfilled after the GW completes — this is your scoreboard
    actual_points: Mapped[Optional[int]] = mapped_column(Integer)


# --------------------------------------------------------------------------
# Your team
# --------------------------------------------------------------------------


class Entry(Base):
    """An FPL manager account (yours). Public data is reachable via
    /api/entry/{fpl_entry_id}/ with no auth."""

    __tablename__ = "entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    fpl_entry_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(64))
    player_name: Mapped[Optional[str]] = mapped_column(String(64))
    is_mine: Mapped[bool] = mapped_column(Boolean, default=False)


class EntryGameweek(Base):
    """Your squad state for one GW. Note picks for the *pending* GW are private;
    the last completed GW is public, so this is normally populated from history
    with the current GW patched in manually if you've already transferred."""

    __tablename__ = "entry_gameweeks"
    __table_args__ = (UniqueConstraint("entry_id", "gameweek_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    entry_id: Mapped[int] = mapped_column(ForeignKey("entries.id"), index=True)
    gameweek_id: Mapped[int] = mapped_column(ForeignKey("gameweeks.id"), index=True)

    points: Mapped[Optional[int]] = mapped_column(Integer)
    total_points: Mapped[Optional[int]] = mapped_column(Integer)
    overall_rank: Mapped[Optional[int]] = mapped_column(Integer)

    bank: Mapped[int] = mapped_column(Integer, default=0)  # tenths
    team_value: Mapped[int] = mapped_column(Integer, default=1000)
    free_transfers: Mapped[int] = mapped_column(Integer, default=1)
    transfers_made: Mapped[int] = mapped_column(Integer, default=0)
    transfer_cost: Mapped[int] = mapped_column(Integer, default=0)
    chip_played: Mapped[Optional[str]] = mapped_column(String(16))

    picks: Mapped[list["EntryPick"]] = relationship(back_populates="entry_gameweek")


class EntryPick(Base):
    """One of the 15. `slot` 1-11 are starters, 12-15 the bench in sub order."""

    __tablename__ = "entry_picks"
    __table_args__ = (UniqueConstraint("entry_gameweek_id", "slot"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    entry_gameweek_id: Mapped[int] = mapped_column(
        ForeignKey("entry_gameweeks.id"), index=True
    )
    player_season_id: Mapped[int] = mapped_column(
        ForeignKey("player_seasons.id"), index=True
    )

    slot: Mapped[int] = mapped_column(Integer)  # 1..15
    is_captain: Mapped[bool] = mapped_column(Boolean, default=False)
    is_vice_captain: Mapped[bool] = mapped_column(Boolean, default=False)
    multiplier: Mapped[int] = mapped_column(Integer, default=1)  # 0 benched, 2 capt

    purchase_price: Mapped[Optional[int]] = mapped_column(Integer)
    selling_price: Mapped[Optional[int]] = mapped_column(Integer)

    entry_gameweek: Mapped["EntryGameweek"] = relationship(back_populates="picks")


# --------------------------------------------------------------------------
# Optimiser output
# --------------------------------------------------------------------------


class OptimisationRun(Base):
    """One solve of the ILP. Persisting these lets you show 'here's what I told
    you last week and here's what it scored'."""

    __tablename__ = "optimisation_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    entry_id: Mapped[int] = mapped_column(ForeignKey("entries.id"), index=True)
    gameweek_id: Mapped[int] = mapped_column(ForeignKey("gameweeks.id"), index=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    horizon_gws: Mapped[int] = mapped_column(Integer, default=5)
    decay: Mapped[float] = mapped_column(Float, default=0.85)
    max_hits: Mapped[int] = mapped_column(Integer, default=0)

    objective_value: Mapped[Optional[float]] = mapped_column(Float)
    solver_status: Mapped[Optional[str]] = mapped_column(String(32))
    solve_seconds: Mapped[Optional[float]] = mapped_column(Float)

    recommendations: Mapped[list["Recommendation"]] = relationship(
        back_populates="run"
    )


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("optimisation_runs.id"), index=True)

    kind: Mapped[str] = mapped_column(String(16))  # start|bench|captain|buy|sell|chip
    player_season_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("player_seasons.id")
    )
    replaces_player_season_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("player_seasons.id")
    )
    target_gameweek_id: Mapped[Optional[int]] = mapped_column(ForeignKey("gameweeks.id"))

    xp_gain: Mapped[Optional[float]] = mapped_column(Float)
    rationale: Mapped[Optional[str]] = mapped_column(Text)

    accepted: Mapped[Optional[bool]] = mapped_column(Boolean)  # did you take it?

    run: Mapped["OptimisationRun"] = relationship(back_populates="recommendations")


# --------------------------------------------------------------------------
# Ingest bookkeeping
# --------------------------------------------------------------------------


class RawIngest(Base):
    """Untouched API payloads. Cheap insurance — when you realise in March that
    you needed a field you never parsed, you can replay from here."""

    __tablename__ = "raw_ingests"
    __table_args__ = (Index("ix_raw_endpoint_time", "endpoint", "fetched_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    endpoint: Mapped[str] = mapped_column(String(128))
    params: Mapped[Optional[str]] = mapped_column(String(128))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status_code: Mapped[int] = mapped_column(Integer)
    payload: Mapped[Optional[dict]] = mapped_column(JSON)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), index=True)


class IngestLog(Base):
    """Did the nightly job run, and did it work? Silent ingest failure is the
    most common way these projects quietly rot."""

    __tablename__ = "ingest_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    job: Mapped[str] = mapped_column(String(64), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    success: Mapped[Optional[bool]] = mapped_column(Boolean)
    rows_written: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[Optional[str]] = mapped_column(Text)