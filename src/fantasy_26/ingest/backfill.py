"""
Backfill: reconstruct history that live ingest cannot recover.

Two jobs, sharing the synthetic-snapshot trick:

1. backfill_snapshots_from_stats()
   Current season, for gameweeks that passed before you started snapshotting.
   `player_fixture_stats.value` records each player's price at each match, so
   price is exact. Position and club are taken from today's registration, which
   is an approximation for anyone reclassified or transferred mid-season.

2. backfill_season_from_csv()
   Past seasons from github.com/vaastav/Fantasy-Premier-League.

WHAT CANNOT BE RECOVERED, ever:
    - injury news and `chance_of_playing_*`      -> filled as available/100
    - daily ownership percentage                 -> left null
    - intra-gameweek price movements             -> one value per gameweek

Backfilled snapshots are therefore honest about price and worthless for
availability. `is_flagged` will be 0 for every backfilled row, so the minutes
model learns nothing from it on this data — that feature only becomes useful
once you have live snapshot history. The rolling minutes features, which come
from match stats and are the stronger signal anyway, are unaffected.

Every synthetic snapshot is written with `captured_at = deadline - 2h` so the
as-of join in features/builder.py finds it.

    uv run python -m fantasy_26.ingest.backfill snapshots
    uv run python -m fantasy_26.ingest.backfill csv --archive ~/data/fpl-archive --season 2024-25
    uv run python -m fantasy_26.ingest.backfill csv --archive ~/data/fpl-archive --all
"""

from __future__ import annotations

import argparse
import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from fantasy_26.db import SessionLocal
from fantasy_26.models.schema import (
    Availability,
    Fixture,
    Gameweek,
    Player,
    PlayerFixtureStat,
    PlayerSeason,
    PlayerSnapshot,
    Position,
    Season,
    Team,
    TeamSeason,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("backfill")

SNAPSHOT_OFFSET = timedelta(hours=2)  # before the deadline, so as-of finds it


# --------------------------------------------------------------------------
# 1. synthetic snapshots for the current season
# --------------------------------------------------------------------------

def _as_position(value) -> Position:
    """The column is a VARCHAR holding the enum NAME (native_enum=False), but
    raw SQL can also hand back the int. Accept either."""
    if isinstance(value, Position):
        return value
    if isinstance(value, str):
        return Position[value] if value in Position.__members__ else Position(int(value))
    return Position(int(value))

def backfill_snapshots_from_stats(
    session: Session,
    season_label: str,
    overwrite: bool = False,
) -> int:
    """One synthetic snapshot per player per played gameweek.

    Only fills gameweeks with no real snapshot before their deadline, unless
    `overwrite` is set. Real snapshots always win — they have news and
    ownership that these do not.
    """
    season = session.scalar(select(Season).where(Season.label == season_label))
    if season is None:
        raise ValueError(f"season {season_label} not found")

    stats = pd.read_sql(
        """
        SELECT pfs.player_season_id, pfs.gameweek_id, pfs.value,
               pfs.transfers_in, pfs.transfers_out, pfs.selected,
               gw.deadline, gw.number,
               ps.position, ps.team_season_id
        FROM player_fixture_stats pfs
        JOIN gameweeks      gw ON gw.id = pfs.gameweek_id
        JOIN player_seasons ps ON ps.id = pfs.player_season_id
        WHERE ps.season_id = %(sid)s
        """,
        session.get_bind(),
        params={"sid": season.id},
    )
    if stats.empty:
        logger.warning("no match stats for %s — run ingest-results first", season_label)
        return 0

    stats["deadline"] = pd.to_datetime(stats["deadline"], utc=True)

    # A double gameweek gives two rows per player; keep the first match's price.
    stats = stats.sort_values("number").drop_duplicates(
        ["player_season_id", "gameweek_id"], keep="first"
    )

    if not overwrite:
        real = pd.read_sql(
            """
            SELECT DISTINCT sn.player_season_id, gw.id AS gameweek_id
            FROM player_snapshots sn
            JOIN player_seasons ps ON ps.id = sn.player_season_id
            JOIN gameweeks      gw ON gw.season_id = ps.season_id
            WHERE ps.season_id = %(sid)s AND sn.captured_at < gw.deadline
            """,
            session.get_bind(),
            params={"sid": season.id},
        )
        if not real.empty:
            before = len(stats)
            stats = stats.merge(
                real, on=["player_season_id", "gameweek_id"], how="left", indicator=True
            )
            stats = stats[stats["_merge"] == "left_only"].drop(columns="_merge")
            logger.info("skipping %d rows already covered", before - len(stats))

    if stats.empty:
        logger.info("nothing to backfill")
        return 0

    rows = [
        {
            "player_season_id": int(r.player_season_id),
            "captured_at": r.deadline - SNAPSHOT_OFFSET,
            "gameweek_id": int(r.gameweek_id),
            "position": _as_position(r.position),
            "team_season_id": int(r.team_season_id),
            "now_cost": int(r.value) if pd.notna(r.value) else 45,
            "cost_change_event": 0,
            "selected_by_percent": 0.0,  # manager count, not a percentage — unusable
            "transfers_in_event": int(r.transfers_in or 0),
            "transfers_out_event": int(r.transfers_out or 0),
            "status": Availability.AVAILABLE,  # unknowable; see module docstring
            "chance_of_playing_this_round": None,
            "chance_of_playing_next_round": None,
            "news": None,
            "news_added": None,
            "form": None,
            "points_per_game": None,
            "ep_this": None,
            "ep_next": None,
            "penalties_order": None,
            "corners_and_indirect_freekicks_order": None,
            "direct_freekicks_order": None,
        }
        for r in stats.itertuples()
    ]

    CHUNK = 2000  # 2000 x 22 cols = 44k params, safely under Postgres's 65535
    for i in range(0, len(rows), CHUNK):
        stmt = insert(PlayerSnapshot).values(rows[i : i + CHUNK])
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["player_season_id", "captured_at"]
        )
        session.execute(stmt)

    session.commit()
    logger.info("wrote %d synthetic snapshots for %s", len(rows), season_label)
    return len(rows)


# --------------------------------------------------------------------------
# 2. past seasons from the vaastav archive
# --------------------------------------------------------------------------


def _read_csv(path: Path) -> pd.DataFrame:
    """Archive CSVs carry a mix of encodings across seasons."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"could not decode {path}")


def _col(df: pd.DataFrame, name: str, default=0):
    """Column access tolerant of schema drift across seasons — `starts` and the
    `expected_*` family don't exist before 2022-23."""
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def backfill_season_from_csv(
    session: Session,
    archive_dir: Path,
    season_label: str,
) -> dict[str, int]:
    """Load one historical season: teams, players, fixtures, stats, snapshots.

    Player identity joins on `code`, never `element` — FPL reassigns element
    ids every August, so joining on them silently merges different players
    across seasons.
    """
    root = archive_dir / "data" / season_label
    if not root.exists():
        raise FileNotFoundError(f"{root} — check the archive path and season label")

    teams_df = _read_csv(root / "teams.csv")
    players_df = _read_csv(root / "players_raw.csv")
    fixtures_df = _read_csv(root / "fixtures.csv")
    gw_df = _read_csv(root / "gws" / "merged_gw.csv")

    counts: dict[str, int] = {}
    start_year = int(season_label.split("-")[0])

    # -- season ----------------------------------------------------------
    season = session.scalar(select(Season).where(Season.label == season_label))
    if season is None:
        season = Season(label=season_label, start_year=start_year, is_current=False)
        session.add(season)
        session.flush()

    # -- gameweeks: derive deadlines from the first kickoff of each event --
    fixtures_df["kickoff_time"] = pd.to_datetime(
        fixtures_df["kickoff_time"], utc=True, errors="coerce"
    )
    events = (
        fixtures_df.dropna(subset=["event", "kickoff_time"])
        .groupby("event")
        .agg(first_kickoff=("kickoff_time", "min"), n_fixtures=("id", "count"))
        .reset_index()
    )
    gw_rows = [
        {
            "season_id": season.id,
            "number": int(e.event),
            # real deadlines aren't in the archive; 90 minutes before the first
            # kickoff is close enough, and only affects snapshot ordering
            "deadline": e.first_kickoff - timedelta(minutes=90),
            "finished": True,
            "data_checked": True,
            "is_double": bool(e.n_fixtures > 10),
            "is_blank": bool(e.n_fixtures < 10),
        }
        for e in events.itertuples()
    ]
    stmt = insert(Gameweek).values(gw_rows)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=["season_id", "number"],
            set_={"deadline": stmt.excluded.deadline},
        )
    )
    session.flush()
    counts["gameweeks"] = len(gw_rows)

    # -- teams -----------------------------------------------------------
    team_rows = [
        {"code": int(t.code), "name": t.name, "short_name": t.short_name}
        for t in teams_df.itertuples()
    ]
    stmt = insert(Team).values(team_rows)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=["code"], set_={"name": stmt.excluded.name}
        )
    )
    session.flush()
    code_to_team = dict(session.execute(select(Team.code, Team.id)).all())

    ts_rows = [
        {
            "season_id": season.id,
            "team_id": code_to_team[int(t.code)],
            "fpl_id": int(t.id),
            "strength_overall_home": int(getattr(t, "strength_overall_home", 0) or 0),
            "strength_overall_away": int(getattr(t, "strength_overall_away", 0) or 0),
            "strength_attack_home": int(getattr(t, "strength_attack_home", 0) or 0),
            "strength_attack_away": int(getattr(t, "strength_attack_away", 0) or 0),
            "strength_defence_home": int(getattr(t, "strength_defence_home", 0) or 0),
            "strength_defence_away": int(getattr(t, "strength_defence_away", 0) or 0),
        }
        for t in teams_df.itertuples()
    ]
    stmt = insert(TeamSeason).values(ts_rows)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=["season_id", "team_id"], set_={"fpl_id": stmt.excluded.fpl_id}
        )
    )
    session.flush()
    counts["teams"] = len(ts_rows)

    fplid_to_ts = dict(
        session.execute(
            select(TeamSeason.fpl_id, TeamSeason.id).where(
                TeamSeason.season_id == season.id
            )
        ).all()
    )

    # -- players ---------------------------------------------------------
    player_rows = [
        {
            "code": int(p.code),
            "first_name": str(p.first_name),
            "second_name": str(p.second_name),
            "web_name": str(p.web_name),
        }
        for p in players_df.itertuples()
    ]
    stmt = insert(Player).values(player_rows)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=["code"], set_={"web_name": stmt.excluded.web_name}
        )
    )
    session.flush()
    code_to_player = dict(session.execute(select(Player.code, Player.id)).all())

    ps_rows = [
        {
            "season_id": season.id,
            "player_id": code_to_player[int(p.code)],
            "element_id": int(p.id),
            "team_season_id": fplid_to_ts[int(p.team)],
            "position": Position(int(p.element_type)),
            "start_price": int(p.now_cost),
        }
        for p in players_df.itertuples()
        if int(p.team) in fplid_to_ts
    ]
    stmt = insert(PlayerSeason).values(ps_rows)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=["season_id", "player_id"],
            set_={"element_id": stmt.excluded.element_id},
        )
    )
    session.flush()
    counts["players"] = len(ps_rows)

    element_to_ps = dict(
        session.execute(
            select(PlayerSeason.element_id, PlayerSeason.id).where(
                PlayerSeason.season_id == season.id
            )
        ).all()
    )
    gw_num_to_id = dict(
        session.execute(
            select(Gameweek.number, Gameweek.id).where(Gameweek.season_id == season.id)
        ).all()
    )

    # -- fixtures --------------------------------------------------------
    fx_rows = [
        {
            "season_id": season.id,
            "fpl_fixture_id": int(f.id),
            "gameweek_id": gw_num_to_id.get(int(f.event)) if pd.notna(f.event) else None,
            "kickoff_time": f.kickoff_time if pd.notna(f.kickoff_time) else None,
            "home_team_season_id": fplid_to_ts[int(f.team_h)],
            "away_team_season_id": fplid_to_ts[int(f.team_a)],
            "home_score": int(f.team_h_score) if pd.notna(f.team_h_score) else None,
            "away_score": int(f.team_a_score) if pd.notna(f.team_a_score) else None,
            "finished": bool(f.finished),
            "minutes_played": int(getattr(f, "minutes", 0) or 0),
            "home_difficulty": int(getattr(f, "team_h_difficulty", 0) or 0),
            "away_difficulty": int(getattr(f, "team_a_difficulty", 0) or 0),
        }
        for f in fixtures_df.itertuples()
        if int(f.team_h) in fplid_to_ts and int(f.team_a) in fplid_to_ts
    ]
    stmt = insert(Fixture).values(fx_rows)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=["season_id", "fpl_fixture_id"],
            set_={"finished": stmt.excluded.finished},
        )
    )
    session.flush()
    counts["fixtures"] = len(fx_rows)

    fplfx_to_fx = dict(
        session.execute(
            select(Fixture.fpl_fixture_id, Fixture.id).where(
                Fixture.season_id == season.id
            )
        ).all()
    )

    # -- player-fixture stats --------------------------------------------
    g = gw_df
    stat_rows = []
    for r in g.itertuples():
        ps_id = element_to_ps.get(int(r.element))
        fx_id = fplfx_to_fx.get(int(r.fixture))
        gw_id = gw_num_to_id.get(int(r.round))
        opp_ts = fplid_to_ts.get(int(r.opponent_team))
        if not (ps_id and fx_id and gw_id and opp_ts):
            continue
        minutes = int(r.minutes)
        stat_rows.append(
            {
                "player_season_id": ps_id,
                "fixture_id": fx_id,
                "gameweek_id": gw_id,
                "was_home": bool(r.was_home),
                "opponent_team_season_id": opp_ts,
                "minutes": minutes,
                # `starts` is absent before 2022-23; 60+ minutes is a decent proxy
                "starts": int(getattr(r, "starts", 1 if minutes >= 60 else 0) or 0),
                "goals_scored": int(r.goals_scored),
                "assists": int(r.assists),
                "clean_sheets": int(r.clean_sheets),
                "goals_conceded": int(r.goals_conceded),
                "own_goals": int(r.own_goals),
                "penalties_saved": int(r.penalties_saved),
                "penalties_missed": int(r.penalties_missed),
                "yellow_cards": int(r.yellow_cards),
                "red_cards": int(r.red_cards),
                "saves": int(r.saves),
                "clearances_blocks_interceptions": int(
                    getattr(r, "clearances_blocks_interceptions", 0) or 0
                ),
                "tackles": int(getattr(r, "tackles", 0) or 0),
                "recoveries": int(getattr(r, "recoveries", 0) or 0),
                "defensive_contribution": int(
                    getattr(r, "defensive_contribution", 0) or 0
                ),
                "bonus": int(r.bonus),
                "bps": int(r.bps),
                "total_points": int(r.total_points),
                "influence": float(r.influence),
                "creativity": float(r.creativity),
                "threat": float(r.threat),
                "ict_index": float(r.ict_index),
                "expected_goals": float(getattr(r, "expected_goals", 0.0) or 0.0),
                "expected_assists": float(getattr(r, "expected_assists", 0.0) or 0.0),
                "expected_goal_involvements": float(
                    getattr(r, "expected_goal_involvements", 0.0) or 0.0
                ),
                "expected_goals_conceded": float(
                    getattr(r, "expected_goals_conceded", 0.0) or 0.0
                ),
                "value": int(r.value),
                "selected": int(r.selected),
                "transfers_in": int(r.transfers_in),
                "transfers_out": int(r.transfers_out),
            }
        )

    seen: set[tuple[int, int]] = set()
    deduped = []
    for row in stat_rows:
        key = (row["player_season_id"], row["fixture_id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    if len(deduped) != len(stat_rows):
        logger.warning(
            "%s: dropped %d duplicate player-fixture rows",
            season_label, len(stat_rows) - len(deduped),
        )
    stat_rows = deduped

    for i in range(0, len(stat_rows), 1500):
        chunk = stat_rows[i: i + 1500]
        stmt = insert(PlayerFixtureStat).values(chunk)
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=["player_season_id", "fixture_id"],
                set_={"total_points": stmt.excluded.total_points},
            )
        )
    session.flush()
    counts["stats"] = len(stat_rows)

    session.commit()

    # -- synthetic snapshots ---------------------------------------------
    counts["snapshots"] = backfill_snapshots_from_stats(session, season_label)

    logger.info("%s: %s", season_label, counts)
    return counts


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical FPL data")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_snap = sub.add_parser("snapshots", help="synthetic snapshots, current season")
    p_snap.add_argument("--season", default=None)
    p_snap.add_argument("--overwrite", action="store_true")

    p_csv = sub.add_parser("csv", help="load past seasons from the vaastav archive")
    p_csv.add_argument("--archive", required=True, type=Path)
    p_csv.add_argument("--season", default=None)
    p_csv.add_argument("--all", action="store_true", help="every season in the archive")

    args = parser.parse_args()

    with SessionLocal() as session:
        if args.cmd == "snapshots":
            label = args.season
            if label is None:
                label = session.scalar(
                    select(Season.label).where(Season.is_current.is_(True))
                )
            backfill_snapshots_from_stats(session, label, overwrite=args.overwrite)
            return

        # args.cmd == "csv"
        if args.all:
            current = session.scalar(
                select(Season.label).where(Season.is_current.is_(True))
            )
            seasons = sorted(
                d.name
                for d in (args.archive / "data").iterdir()
                if d.is_dir() and "-" in d.name
            )
            # expected_goals only exists from 2022-23; earlier seasons still give
            # minutes, points and bps, which is most of the value
            seasons = [
                s for s in seasons
                if int(s.split("-")[0]) >= 2019 and s != current
            ]
        elif args.season:
            seasons = [args.season]
        else:
            raise SystemExit("pass --season or --all")

        logger.info("loading %d season(s): %s", len(seasons), ", ".join(seasons))

        for label in seasons:
            try:
                backfill_season_from_csv(session, args.archive, label)
            except Exception as exc:
                logger.error("%s failed: %s", label, exc)
                session.rollback()


if __name__ == "__main__":
    main()