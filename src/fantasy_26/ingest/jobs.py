"""
Ingest orchestration.

    uv run python -m fantasy_26.ingest.jobs bootstrap   # reference data, run once
    uv run python -m fantasy_26.ingest.jobs snapshot    # daily, ~2am
    uv run python -m fantasy_26.ingest.jobs results     # after each GW completes
    uv run python -m fantasy_26.ingest.jobs squad       # after each deadline
    uv run python -m fantasy_26.ingest.jobs daily       # bootstrap + snapshot
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select

from fantasy_26.config import settings
from fantasy_26.db import SessionLocal
from fantasy_26.ingest.client import FPLClient
from fantasy_26.ingest import loaders as L
from fantasy_26.models.schema import Gameweek, PlayerSeason, Season

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger("ingest")


# --------------------------------------------------------------------------


async def run_bootstrap() -> None:
    """Reference data: season, gameweeks, teams, players, fixtures.
    Idempotent — safe to re-run whenever squads or the fixture list change."""
    async with FPLClient(settings.fpl_base_url) as client:
        boot = await client.bootstrap_static()
        fixtures = await client.fixtures()

    with SessionLocal() as session, L.job_log(session, "bootstrap") as log:
        L.archive(session, boot)
        L.archive(session, fixtures)

        season = L.load_season(session, boot.payload)
        session.flush()

        n = L.load_gameweeks(session, season, boot.payload)
        logger.info("gameweeks: %d", n)

        n = L.load_teams(session, season, boot.payload)
        session.flush()
        logger.info("teams: %d", n)

        n = L.load_players(session, season, boot.payload)
        session.flush()
        logger.info("players: %d", n)

        n = L.load_fixtures(session, season, fixtures.payload)
        logger.info("fixtures: %d", n)

        L.mark_blanks_and_doubles(session, season)

        log.rows_written = n
        session.commit()


async def run_snapshot() -> None:
    """Daily append-only capture of price, ownership, availability and news.

    Run this at least once a day. Twice is better — once around 02:00 UTC to
    catch overnight price changes, once shortly before the deadline to catch
    late team news. Missing days leaves gaps you cannot backfill.
    """
    captured_at = datetime.now(timezone.utc)

    async with FPLClient(settings.fpl_base_url) as client:
        boot = await client.bootstrap_static()

    with SessionLocal() as session, L.job_log(session, "snapshot") as log:
        L.archive(session, boot)
        season = L.load_season(session, boot.payload)

        # Squads change: new signings appear mid-season
        L.load_players(session, season, boot.payload)
        session.flush()

        n = L.load_snapshots(session, season, boot.payload, captured_at)
        logger.info("snapshots: %d", n)

        changes = L.derive_price_changes(session, season)
        logger.info("price changes: %d", changes)

        log.rows_written = n
        session.commit()


async def run_results(gameweek: int | None = None) -> None:
    """Pull per-fixture player stats.

    Wait for `data_checked` before trusting a gameweek — bonus points and
    some stats are provisional until then, usually a day or two after the
    final whistle.
    """
    with SessionLocal() as session:
        season = session.scalar(select(Season).where(Season.is_current.is_(True)))
        if season is None:
            raise SystemExit("no current season — run `bootstrap` first")

        element_ids = list(
            session.scalars(
                select(PlayerSeason.element_id).where(
                    PlayerSeason.season_id == season.id
                )
            ).all()
        )
        season_id = season.id

    logger.info("fetching history for %d players", len(element_ids))

    async with FPLClient(settings.fpl_base_url) as client:
        summaries = await client.element_summaries(element_ids)

    with SessionLocal() as session, L.job_log(session, "results") as log:
        season = session.get(Season, season_id)
        total = 0
        for result in summaries:
            element_id = int(result.params)
            history = result.payload.get("history", [])
            if gameweek is not None:
                history = [h for h in history if h["round"] == gameweek]
            total += L.load_player_history(session, season, element_id, history)

        logger.info("stat rows: %d", total)
        log.rows_written = total
        session.commit()


async def run_squad() -> None:
    """Fetch your squad for the most recent COMPLETED gameweek.

    Pending-GW picks are private, so this deliberately targets the last
    finished GW. If you have already made transfers for the upcoming week,
    patch them in by hand — that is the whole manual step, and it takes
    about ten seconds.
    """
    if not settings.entry_id:
        raise SystemExit("set entry_id in .env — it's in your FPL team page URL")

    with SessionLocal() as session:
        season = session.scalar(select(Season).where(Season.is_current.is_(True)))
        last_finished = session.scalar(
            select(Gameweek.number)
            .where(Gameweek.season_id == season.id, Gameweek.finished.is_(True))
            .order_by(Gameweek.number.desc())
            .limit(1)
        )

    if last_finished is None:
        logger.info("no completed gameweek yet — nothing to pull")
        return

    async with FPLClient(settings.fpl_base_url) as client:
        entry = await client.entry(settings.entry_id)
        history = await client.entry_history(settings.entry_id)
        picks = await client.entry_picks(settings.entry_id, last_finished)
        transfers = await client.entry_transfers(settings.entry_id)

    with SessionLocal() as session, L.job_log(session, "squad") as log:
        for r in (entry, history, picks, transfers):
            L.archive(session, r)
        season = session.merge(season)
        log.rows_written = L.load_squad(
            session,
            season,
            last_finished,
            entry.payload,
            history.payload,
            picks.payload,
            transfers.payload,
        )
        logger.info("squad: %d picks for GW%d", log.rows_written, last_finished)
        session.commit()


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="FPL ingest jobs")
    parser.add_argument(
        "job",
        choices=["bootstrap", "snapshot", "results", "squad", "daily"],
    )
    parser.add_argument("--gameweek", type=int, default=None)
    args = parser.parse_args()

    if args.job == "bootstrap":
        asyncio.run(run_bootstrap())
    elif args.job == "snapshot":
        asyncio.run(run_snapshot())
    elif args.job == "results":
        asyncio.run(run_results(args.gameweek))
    elif args.job == "squad":
        asyncio.run(run_squad())
    elif args.job == "daily":

        async def _daily() -> None:
            await run_bootstrap()
            await run_snapshot()

        asyncio.run(_daily())


if __name__ == "__main__":
    main()
