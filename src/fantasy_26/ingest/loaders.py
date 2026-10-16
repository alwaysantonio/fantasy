"""
Loaders: map raw FPL payloads into the schema.

Two rules govern everything here:

1. Reference data (seasons, teams, players, fixtures, gameweeks) is UPSERTED.
   Re-running is safe and idempotent.

2. Snapshots (`player_snapshots`) are APPENDED, never updated. Each run adds a
   new row with the current mutable state. This is what makes deadline-aware
   feature building possible — see the note in schema.py.

Load order matters because of foreign keys:
    season -> gameweeks + team_seasons -> player_seasons -> fixtures -> stats
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from fantasy_26.ingest.client import FetchResult
from fantasy_26.models.schema import (
    Availability,
    Entry,
    EntryGameweek,
    EntryPick,
    Fixture,
    Gameweek,
    IngestLog,
    Player,
    PlayerFixtureStat,
    PlayerSeason,
    PlayerSnapshot,
    Position,
    PriceChange,
    RawIngest,
    Season,
    Team,
    TeamSeason,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _dt(value: Optional[str]) -> Optional[datetime]:
    """FPL returns ISO-8601 with a trailing Z."""
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _upsert(session: Session, model, rows: list[dict], conflict_cols: list[str]) -> int:
    """Bulk INSERT ... ON CONFLICT DO UPDATE. Postgres-specific and much faster
    than a merge-per-row loop."""
    if not rows:
        return 0
    stmt = insert(model).values(rows)
    update_cols = {
        c.name: stmt.excluded[c.name]
        for c in model.__table__.columns
        if c.name not in conflict_cols and c.name != "id"
    }
    stmt = stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_cols)
    session.execute(stmt)
    return len(rows)


# --------------------------------------------------------------------------
# raw archive
# --------------------------------------------------------------------------


def archive(session: Session, result: FetchResult) -> None:
    """Store the untouched payload. Skips storage if we already hold an
    identical body — bootstrap-static barely changes between deadline days."""
    exists = session.scalar(
        select(RawIngest.id).where(RawIngest.content_hash == result.content_hash).limit(1)
    )
    if exists:
        return
    session.add(
        RawIngest(
            endpoint=result.endpoint,
            params=result.params,
            fetched_at=result.fetched_at,
            status_code=result.status_code,
            payload=result.payload,
            content_hash=result.content_hash,
        )
    )


# --------------------------------------------------------------------------
# reference data
# --------------------------------------------------------------------------


def load_season(session: Session, boot: dict, label: Optional[str] = None) -> Season:
    """Derive the season from the first gameweek deadline unless told otherwise.
    GW1 is always in August, so the deadline year is the start year."""
    if label is None:
        first = min(boot["events"], key=lambda e: e["id"])
        start_year = _dt(first["deadline_time"]).year
        label = f"{start_year}-{str(start_year + 1)[-2:]}"
    else:
        start_year = int(label.split("-")[0])

    season = session.scalar(select(Season).where(Season.label == label))
    if season is None:
        session.execute(
            insert(Season)
            .values(label=label, start_year=start_year, is_current=True)
            .on_conflict_do_nothing(index_elements=["label"])
        )
        session.flush()
        season = session.scalar(select(Season).where(Season.label == label))
    return season


def load_gameweeks(session: Session, season: Season, boot: dict) -> int:
    """Also derives blank/double flags from the fixture list, which the events
    payload doesn't give you directly."""
    rows = []
    for e in boot["events"]:
        rows.append(
            {
                "season_id": season.id,
                "number": e["id"],
                "deadline": _dt(e["deadline_time"]),
                "finished": bool(e.get("finished")),
                "data_checked": bool(e.get("data_checked")),
                "is_blank": False,  # set by mark_blanks_and_doubles()
                "is_double": False,
                "average_entry_score": e.get("average_entry_score"),
                "highest_score": e.get("highest_score"),
                "chip_plays": e.get("chip_plays"),
            }
        )
    return _upsert(session, Gameweek, rows, ["season_id", "number"])


def load_teams(session: Session, season: Season, boot: dict) -> int:
    """Two steps: canonical Team keyed on the stable `code`, then the
    season-specific TeamSeason holding this year's 1..20 id."""
    team_rows = [
        {"code": t["code"], "name": t["name"], "short_name": t["short_name"]}
        for t in boot["teams"]
    ]
    _upsert(session, Team, team_rows, ["code"])
    session.flush()

    code_to_id = dict(session.execute(select(Team.code, Team.id)).all())

    ts_rows = []
    for t in boot["teams"]:
        ts_rows.append(
            {
                "season_id": season.id,
                "team_id": code_to_id[t["code"]],
                "fpl_id": t["id"],
                "strength_overall_home": t.get("strength_overall_home"),
                "strength_overall_away": t.get("strength_overall_away"),
                "strength_attack_home": t.get("strength_attack_home"),
                "strength_attack_away": t.get("strength_attack_away"),
                "strength_defence_home": t.get("strength_defence_home"),
                "strength_defence_away": t.get("strength_defence_away"),
            }
        )
    return _upsert(session, TeamSeason, ts_rows, ["season_id", "team_id"])


def load_players(session: Session, season: Season, boot: dict) -> int:
    """Canonical Player on `code`, then PlayerSeason for this year's element id.

    `code` is stable across seasons; `id` is not. Getting this wrong silently
    joins the wrong players together in multi-season training sets.
    """
    player_rows = []
    for e in boot["elements"]:
        player_rows.append(
            {
                "code": e["code"],
                "opta_code": e.get("opta_code"),
                "first_name": e["first_name"],
                "second_name": e["second_name"],
                "web_name": e["web_name"],
                "date_of_birth": e.get("birth_date"),
            }
        )
    _upsert(session, Player, player_rows, ["code"])
    session.flush()

    code_to_pid = dict(session.execute(select(Player.code, Player.id)).all())
    fplid_to_tsid = dict(
        session.execute(
            select(TeamSeason.fpl_id, TeamSeason.id).where(
                TeamSeason.season_id == season.id
            )
        ).all()
    )

    ps_rows = []
    for e in boot["elements"]:
        ps_rows.append(
            {
                "season_id": season.id,
                "player_id": code_to_pid[e["code"]],
                "element_id": e["id"],
                "team_season_id": fplid_to_tsid[e["team"]],
                "position": Position(e["element_type"]),
                "start_price": _i(e["now_cost"]),  # only correct on first load
                "squad_number": e.get("squad_number"),
            }
        )
    return _upsert(session, PlayerSeason, ps_rows, ["season_id", "player_id"])


def load_fixtures(session: Session, season: Season, payload: list[dict]) -> int:
    gw_map = dict(
        session.execute(
            select(Gameweek.number, Gameweek.id).where(Gameweek.season_id == season.id)
        ).all()
    )
    ts_map = dict(
        session.execute(
            select(TeamSeason.fpl_id, TeamSeason.id).where(
                TeamSeason.season_id == season.id
            )
        ).all()
    )

    rows = []
    for f in payload:
        rows.append(
            {
                "season_id": season.id,
                "fpl_fixture_id": f["id"],
                "gameweek_id": gw_map.get(f["event"]) if f.get("event") else None,
                "kickoff_time": _dt(f.get("kickoff_time")),
                "home_team_season_id": ts_map[f["team_h"]],
                "away_team_season_id": ts_map[f["team_a"]],
                "home_score": f.get("team_h_score"),
                "away_score": f.get("team_a_score"),
                "finished": bool(f.get("finished")),
                "minutes_played": _i(f.get("minutes")),
                "home_difficulty": f.get("team_h_difficulty"),
                "away_difficulty": f.get("team_a_difficulty"),
            }
        )
    return _upsert(session, Fixture, rows, ["season_id", "fpl_fixture_id"])


def mark_blanks_and_doubles(session: Session, season: Season) -> None:
    """A team with two fixtures in a GW = double; zero = blank. Both are
    important features and both break naive per-gameweek assumptions."""
    from sqlalchemy import func

    counts = session.execute(
        select(
            Fixture.gameweek_id,
            func.count(Fixture.id).label("n"),
        )
        .where(Fixture.season_id == season.id, Fixture.gameweek_id.isnot(None))
        .group_by(Fixture.gameweek_id)
    ).all()

    for gw_id, n in counts:
        gw = session.get(Gameweek, gw_id)
        if gw is None:
            continue
        gw.is_double = n > 10  # 20 teams / 2 = 10 fixtures in a normal GW
        gw.is_blank = n < 10


# --------------------------------------------------------------------------
# snapshots (append-only)
# --------------------------------------------------------------------------


def load_snapshots(
    session: Session,
    season: Season,
    boot: dict,
    captured_at: datetime,
) -> int:
    """One row per player capturing everything mutable: price, ownership,
    availability, news, set-piece order, position, club.

    Position and team are captured here because both change mid-season and you
    need to know what was true at each deadline, not what is true today.
    """
    ps_map = dict(
        session.execute(
            select(PlayerSeason.element_id, PlayerSeason.id).where(
                PlayerSeason.season_id == season.id
            )
        ).all()
    )
    ts_map = dict(
        session.execute(
            select(TeamSeason.fpl_id, TeamSeason.id).where(
                TeamSeason.season_id == season.id
            )
        ).all()
    )
    current_gw = session.scalar(
        select(Gameweek.id)
        .where(Gameweek.season_id == season.id, Gameweek.finished.is_(False))
        .order_by(Gameweek.number)
        .limit(1)
    )

    rows = []
    for e in boot["elements"]:
        ps_id = ps_map.get(e["id"])
        if ps_id is None:
            continue  # new player mid-run; picked up next cycle
        rows.append(
            {
                "player_season_id": ps_id,
                "captured_at": captured_at,
                "gameweek_id": current_gw,
                "position": Position(e["element_type"]),
                "team_season_id": ts_map[e["team"]],
                "now_cost": _i(e["now_cost"]),
                "cost_change_event": _i(e.get("cost_change_event")),
                "selected_by_percent": _f(e.get("selected_by_percent")),
                "transfers_in_event": _i(e.get("transfers_in_event")),
                "transfers_out_event": _i(e.get("transfers_out_event")),
                "status": Availability(e.get("status", "a")),
                "chance_of_playing_this_round": e.get("chance_of_playing_this_round"),
                "chance_of_playing_next_round": e.get("chance_of_playing_next_round"),
                "news": e.get("news") or None,
                "news_added": _dt(e.get("news_added")),
                "form": _f(e.get("form")),
                "points_per_game": _f(e.get("points_per_game")),
                "ep_this": _f(e.get("ep_this")),
                "ep_next": _f(e.get("ep_next")),
                "penalties_order": e.get("penalties_order"),
                "corners_and_indirect_freekicks_order": e.get(
                    "corners_and_indirect_freekicks_order"
                ),
                "direct_freekicks_order": e.get("direct_freekicks_order"),
            }
        )

    stmt = insert(PlayerSnapshot).values(rows)
    stmt = stmt.on_conflict_do_nothing(index_elements=["player_season_id", "captured_at"])
    session.execute(stmt)
    return len(rows)


def derive_price_changes(session: Session, season: Season) -> int:
    """Diff consecutive snapshots into an explicit price-change table. The
    price model trains on this directly; the self-join is slow and fiddly."""
    from sqlalchemy import func

    sub = (
        select(
            PlayerSnapshot.player_season_id,
            PlayerSnapshot.captured_at,
            PlayerSnapshot.now_cost,
            PlayerSnapshot.selected_by_percent,
            PlayerSnapshot.transfers_in_event,
            PlayerSnapshot.transfers_out_event,
            func.lag(PlayerSnapshot.now_cost)
            .over(
                partition_by=PlayerSnapshot.player_season_id,
                order_by=PlayerSnapshot.captured_at,
            )
            .label("prev_cost"),
        )
        .join(PlayerSeason, PlayerSeason.id == PlayerSnapshot.player_season_id)
        .where(PlayerSeason.season_id == season.id)
        .subquery()
    )

    changed = session.execute(
        select(sub).where(sub.c.prev_cost.isnot(None), sub.c.now_cost != sub.c.prev_cost)
    ).all()

    rows = [
        {
            "player_season_id": r.player_season_id,
            "changed_on": r.captured_at.date(),
            "old_cost": r.prev_cost,
            "new_cost": r.now_cost,
            "delta": r.now_cost - r.prev_cost,
            "net_transfers_24h": r.transfers_in_event - r.transfers_out_event,
            "ownership_at_change": r.selected_by_percent,
        }
        for r in changed
    ]
    if not rows:
        return 0
    stmt = insert(PriceChange).values(rows)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=["player_season_id", "changed_on"]
    )
    session.execute(stmt)
    return len(rows)


# --------------------------------------------------------------------------
# match results
# --------------------------------------------------------------------------


def load_player_history(
    session: Session,
    season: Season,
    element_id: int,
    history: list[dict],
) -> int:
    """Per-fixture results from element-summary/{id}/.

    Keyed on fixture, so double gameweeks produce two rows naturally.

    NB: verify what `defensive_contribution` contains on your first ingest —
    it may be the raw action count or the 0/2 points awarded. The cbit/cbirt
    properties on the model recompute it from components either way, so prefer
    those for modelling.
    """
    ps_id = session.scalar(
        select(PlayerSeason.id).where(
            PlayerSeason.season_id == season.id,
            PlayerSeason.element_id == element_id,
        )
    )
    if ps_id is None:
        return 0

    fx_map = dict(
        session.execute(
            select(Fixture.fpl_fixture_id, Fixture.id).where(
                Fixture.season_id == season.id
            )
        ).all()
    )
    gw_map = dict(
        session.execute(
            select(Gameweek.number, Gameweek.id).where(Gameweek.season_id == season.id)
        ).all()
    )
    ts_map = dict(
        session.execute(
            select(TeamSeason.fpl_id, TeamSeason.id).where(
                TeamSeason.season_id == season.id
            )
        ).all()
    )

    rows = []
    for h in history:
        fixture_id = fx_map.get(h["fixture"])
        gameweek_id = gw_map.get(h["round"])
        if fixture_id is None or gameweek_id is None:
            continue
        rows.append(
            {
                "player_season_id": ps_id,
                "fixture_id": fixture_id,
                "gameweek_id": gameweek_id,
                "was_home": bool(h["was_home"]),
                "opponent_team_season_id": ts_map[h["opponent_team"]],
                "minutes": _i(h.get("minutes")),
                "starts": _i(h.get("starts")),
                "goals_scored": _i(h.get("goals_scored")),
                "assists": _i(h.get("assists")),
                "clean_sheets": _i(h.get("clean_sheets")),
                "goals_conceded": _i(h.get("goals_conceded")),
                "own_goals": _i(h.get("own_goals")),
                "penalties_saved": _i(h.get("penalties_saved")),
                "penalties_missed": _i(h.get("penalties_missed")),
                "yellow_cards": _i(h.get("yellow_cards")),
                "red_cards": _i(h.get("red_cards")),
                "saves": _i(h.get("saves")),
                "clearances_blocks_interceptions": _i(
                    h.get("clearances_blocks_interceptions")
                ),
                "tackles": _i(h.get("tackles")),
                "recoveries": _i(h.get("recoveries")),
                "defensive_contribution": _i(h.get("defensive_contribution")),
                "bonus": _i(h.get("bonus")),
                "bps": _i(h.get("bps")),
                "total_points": _i(h.get("total_points")),
                "influence": _f(h.get("influence")),
                "creativity": _f(h.get("creativity")),
                "threat": _f(h.get("threat")),
                "ict_index": _f(h.get("ict_index")),
                "expected_goals": _f(h.get("expected_goals")),
                "expected_assists": _f(h.get("expected_assists")),
                "expected_goal_involvements": _f(h.get("expected_goal_involvements")),
                "expected_goals_conceded": _f(h.get("expected_goals_conceded")),
                "value": _i(h.get("value")),
                "selected": _i(h.get("selected")),
                "transfers_in": _i(h.get("transfers_in")),
                "transfers_out": _i(h.get("transfers_out")),
            }
        )

    return _upsert(session, PlayerFixtureStat, rows, ["player_season_id", "fixture_id"])


# --------------------------------------------------------------------------
# job logging
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# your squad
# --------------------------------------------------------------------------


def selling_price(purchase: int, now: int) -> int:
    """FPL keeps half of any rise, rounded down; a fall is passed on in full."""
    return purchase + (now - purchase) // 2 if now > purchase else now


def load_squad(
    session: Session,
    season: Season,
    gameweek_number: int,
    entry: dict,
    history: dict,
    picks: dict,
    transfers: list[dict],
) -> int:
    """Entry, EntryGameweek and 15 EntryPicks for one completed gameweek.

    Purchase price is the cost of the most recent transfer that brought the
    player in. For players held since the entry started, it's their price at
    that gameweek's deadline. The current price is the latest snapshot, so
    selling prices are as of the last `snapshot` run.
    """
    _upsert(
        session,
        Entry,
        [
            {
                "fpl_entry_id": entry["id"],
                "name": entry.get("name"),
                "player_name": f"{entry.get('player_first_name', '')} "
                f"{entry.get('player_last_name', '')}".strip(),
                "is_mine": True,
            }
        ],
        ["fpl_entry_id"],
    )
    entry_pk = session.scalar(select(Entry.id).where(Entry.fpl_entry_id == entry["id"]))
    gw = session.scalar(
        select(Gameweek).where(
            Gameweek.season_id == season.id, Gameweek.number == gameweek_number
        )
    )
    start_gw = session.scalar(
        select(Gameweek).where(
            Gameweek.season_id == season.id,
            Gameweek.number == entry.get("started_event", 1),
        )
    )

    h = next((r for r in history.get("current", []) if r["event"] == gameweek_number), {})
    chip = next(
        (c["name"] for c in history.get("chips", []) if c["event"] == gameweek_number), None
    )
    _upsert(
        session,
        EntryGameweek,
        [
            {
                "entry_id": entry_pk,
                "gameweek_id": gw.id,
                "points": h.get("points"),
                "total_points": h.get("total_points"),
                "overall_rank": h.get("overall_rank"),
                "bank": _i(h.get("bank")),
                "team_value": _i(h.get("value"), 1000),
                "transfers_made": _i(h.get("event_transfers")),
                "transfer_cost": _i(h.get("event_transfers_cost")),
                "chip_played": chip,
            }
        ],
        ["entry_id", "gameweek_id"],
    )
    egw_id = session.scalar(
        select(EntryGameweek.id).where(
            EntryGameweek.entry_id == entry_pk, EntryGameweek.gameweek_id == gw.id
        )
    )

    elements = [p["element"] for p in picks["picks"]]
    ps_by_element = dict(
        session.execute(
            select(PlayerSeason.element_id, PlayerSeason.id).where(
                PlayerSeason.season_id == season.id,
                PlayerSeason.element_id.in_(elements),
            )
        ).all()
    )

    def price_at(player_season_id: int, when: Optional[datetime]) -> Optional[int]:
        q = select(PlayerSnapshot.now_cost).where(
            PlayerSnapshot.player_season_id == player_season_id
        )
        if when is not None:
            q = q.where(PlayerSnapshot.captured_at < when)
        return session.scalar(q.order_by(PlayerSnapshot.captured_at.desc()).limit(1))

    bought = {}
    for t in sorted(transfers, key=lambda t: t["time"]):
        bought[t["element_in"]] = t["element_in_cost"]

    rows = []
    for p in picks["picks"]:
        ps_id = ps_by_element[p["element"]]
        now = price_at(ps_id, None)
        purchase = bought.get(p["element"]) or price_at(ps_id, start_gw.deadline)
        rows.append(
            {
                "entry_gameweek_id": egw_id,
                "player_season_id": ps_id,
                "slot": p["position"],
                "is_captain": p["is_captain"],
                "is_vice_captain": p["is_vice_captain"],
                "multiplier": p["multiplier"],
                "purchase_price": purchase,
                "selling_price": selling_price(purchase, now) if purchase and now else now,
            }
        )
    return _upsert(session, EntryPick, rows, ["entry_gameweek_id", "slot"])


class job_log:
    """Context manager writing an IngestLog row. Silent ingest failure is the
    most common way these projects rot — you want a record of every run."""

    def __init__(self, session: Session, job: str) -> None:
        self.session = session
        self.entry = IngestLog(
            job=job, started_at=datetime.now(timezone.utc), rows_written=0
        )

    def __enter__(self) -> IngestLog:
        self.session.add(self.entry)
        self.session.flush()
        return self.entry

    def __exit__(self, exc_type, exc, _tb) -> bool:
        self.entry.finished_at = datetime.now(timezone.utc)
        self.entry.success = exc_type is None
        if exc is not None:
            self.entry.error = f"{exc_type.__name__}: {exc}"[:4000]
        self.session.commit()
        return False
