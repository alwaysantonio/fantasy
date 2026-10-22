"""
Prune superseded model outputs so the database stays small.

The daily job writes a fresh xP run (≈4,000 rows) and a fresh team strength
run every morning. Only two kinds of run are ever read again:

  * the latest run — what the optimiser uses today
  * the last run before each deadline — what you'd have used for that
    gameweek, which is what the backtest scores

Everything in between is deleted. The model_runs rows themselves are kept,
since they're tiny and optimisation_runs point at them. Team strength runs
covering more than one season (the full walk-forward history from
`make strength`) are always kept.

    uv run python -m fantasy_26.prune            # delete
    uv run python -m fantasy_26.prune --dry-run  # just count
"""

from __future__ import annotations

import argparse
import logging

from sqlalchemy import text

from fantasy_26.db import engine

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("prune")

# runs to drop: not the latest, and not the last one before any deadline
DROP_RUNS_SQL = """
WITH runs AS (
    SELECT id, trained_at FROM model_runs
    WHERE name = :name {extra}
),
keep AS (
    SELECT max(id) AS id FROM runs
    UNION
    SELECT (SELECT r.id FROM runs r WHERE r.trained_at < g.deadline
            ORDER BY r.trained_at DESC LIMIT 1)
    FROM gameweeks g
)
SELECT id FROM runs WHERE id NOT IN (SELECT id FROM keep WHERE id IS NOT NULL)
"""

TARGETS = [
    # (model name, extra filter on runs, tables holding that run's rows)
    ("xp_aggregate", "", ["predictions"]),
    (
        "team_strength_dc",
        "AND json_array_length(config -> 'seasons') = 1",
        ["fixture_projections", "team_strengths"],
    ),
]


def prune(dry_run: bool = False) -> int:
    total = 0
    with engine.begin() as conn:
        for name, extra, tables in TARGETS:
            ids = [r[0] for r in conn.execute(text(DROP_RUNS_SQL.format(extra=extra)), {"name": name})]
            if not ids:
                continue
            for table in tables:
                sql = f"FROM {table} WHERE model_run_id = ANY(:ids)"
                if dry_run:
                    n = conn.execute(text(f"SELECT count(*) {sql}"), {"ids": ids}).scalar()
                else:
                    n = conn.execute(text(f"DELETE {sql}"), {"ids": ids}).rowcount
                total += n
                logger.info("%s: %s %d rows from %d superseded %s runs",
                            table, "would delete" if dry_run else "deleted", n, len(ids), name)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune superseded model outputs")
    parser.add_argument("--dry-run", action="store_true")
    prune(parser.parse_args().dry_run)


if __name__ == "__main__":
    main()
