
# fantasy-26 — project commands
#
#   make            show this help
#   make setup      first-time setup from a fresh clone
#
# Everything assumes `uv` and Docker are installed.

UV      := uv run
COMPOSE := docker compose
PKG     := fantasy_26
SEASON  ?= 2026-27
GW      ?=

.DEFAULT_GOAL := help

# --------------------------------------------------------------------------
# help
# --------------------------------------------------------------------------

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------

.PHONY: install
install:  ## Install dependencies into .venv
	uv sync

.PHONY: setup
setup: install up wait-db migrate ingest-bootstrap  ## Full first-time setup
	@echo ""
	@echo "Setup complete. Next: 'make ingest-snapshot', then backfill history."

.PHONY: reset
reset: nuke up wait-db migrate ingest-bootstrap  ## Wipe everything and rebuild
	@echo "Database rebuilt from scratch."

# --------------------------------------------------------------------------
# docker / postgres
# --------------------------------------------------------------------------

.PHONY: up
up:  ## Start Postgres
	$(COMPOSE) up -d

.PHONY: down
down:  ## Stop Postgres (keeps data)
	$(COMPOSE) down

#.PHONY: nuke
#nuke:  ## Stop Postgres and DELETE the volume — all data lost
#	$(COMPOSE) down -v

.PHONY: wait-db
wait-db:  ## Block until Postgres accepts connections
	@echo "waiting for postgres..."
	@until $(COMPOSE) exec -T db pg_isready -U fpl -d fpl >/dev/null 2>&1; do \
		printf '.'; sleep 1; \
	done
	@echo " ready"

.PHONY: logs
logs:  ## Tail Postgres logs
	$(COMPOSE) logs -f db

.PHONY: psql
psql:  ## Open a psql shell
	$(COMPOSE) exec db psql -U fantasy_26 -d fantasy_26

# --------------------------------------------------------------------------
# migrations
# --------------------------------------------------------------------------

.PHONY: migrate
migrate:  ## Apply all pending migrations
	$(UV) alembic upgrade head

.PHONY: migration
migration:  ## Autogenerate a migration: make migration m="add foo"
	@test -n "$(m)" || (echo "usage: make migration m=\"description\""; exit 1)
	$(UV) alembic revision --autogenerate -m "$(m)"
	@echo ""
	@echo "Review the generated file in migrations/versions/ before applying."

.PHONY: downgrade
downgrade:  ## Roll back one migration
	$(UV) alembic downgrade -1

.PHONY: history
history:  ## Show migration history
	$(UV) alembic history --verbose

.PHONY: current
current:  ## Show the currently applied revision
	$(UV) alembic current

# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------

.PHONY: ingest-bootstrap
ingest-bootstrap:  ## Reference data: season, gameweeks, teams, players, fixtures
	$(UV) python -m $(PKG).ingest.jobs bootstrap

.PHONY: ingest-snapshot
ingest-snapshot:  ## Daily append-only capture of price/ownership/availability
	$(UV) python -m $(PKG).ingest.jobs snapshot

.PHONY: ingest-results
ingest-results:  ## Per-fixture player stats. Optional: make ingest-results GW=7
	$(UV) python -m $(PKG).ingest.jobs results $(if $(GW),--gameweek $(GW),)

.PHONY: ingest-squad
ingest-squad:  ## Pull your squad for the last completed gameweek
	$(UV) python -m $(PKG).ingest.jobs squad

.PHONY: ingest-daily
ingest-daily:  ## bootstrap + snapshot — this is the cron target
	$(UV) python -m $(PKG).ingest.jobs daily

# --------------------------------------------------------------------------
# inspection
# --------------------------------------------------------------------------

.PHONY: status
status:  ## Row counts per table — quick sanity check
	@$(UV) python -c "\
	from fantasy_26.db import engine; import pandas as pd; \
	print(pd.read_sql('''select s.label, s.is_current, \
	 (select count(*) from team_seasons t where t.season_id=s.id) teams, \
	 (select count(*) from player_seasons ps where ps.season_id=s.id) players, \
	 (select count(*) from fixtures f where f.season_id=s.id) fixtures, \
	 (select count(*) from gameweeks g where g.season_id=s.id and g.finished) done_gws, \
	 (select count(*) from player_fixture_stats pfs join player_seasons p2 \
	   on p2.id=pfs.player_season_id where p2.season_id=s.id) stats, \
	 (select count(*) from player_snapshots sn join player_seasons p3 \
	   on p3.id=sn.player_season_id where p3.season_id=s.id) snapshots \
	 from seasons s order by s.start_year''', engine).to_string(index=False))"

.PHONY: ingest-log
ingest-log:  ## Last 15 ingest job runs
	@$(UV) python -c "\
	from fantasy_26.db import engine; import pandas as pd; \
	print(pd.read_sql('''select job, started_at, success, rows_written, \
	 left(coalesce(error,''),60) error from ingest_logs \
	 order by started_at desc limit 15''', engine).to_string(index=False))"

.PHONY: features
features:  ## Build the training frame and run leakage checks
	@$(UV) python -c "\
	from fantasy_26.db import engine; \
	from fantasy_26.features.builder import build_training_frame, assert_no_leakage; \
	df = build_training_frame(engine, ['$(SEASON)']); \
	assert_no_leakage(df); \
	print(df.shape); \
	print(df[['web_name','gameweek','minutes_mean_5','xg90','y_points']].tail(10))"

# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

.PHONY: minutes-eval
minutes-eval:  ## Minutes model: train with a holdout season and print metrics
	$(UV) python -m $(PKG).predict.minutes train

.PHONY: minutes-train
minutes-train:  ## Minutes model: production fit, calibrated on the last full season
	$(UV) python -m $(PKG).predict.minutes train --calib 2025-26 --no-valid

.PHONY: strength-eval
strength-eval:  ## Team strength: tune on 2024-25, report on 2025-26
	$(UV) python -m $(PKG).predict.team_strength evaluate

.PHONY: strength
strength:  ## Team strength: walk-forward projections for every fixture to date
	$(UV) python -m $(PKG).predict.team_strength run

.PHONY: attacking-eval
attacking-eval:  ## Attacking rates: fit on 2023-25, report on 2025-26
	$(UV) python -m $(PKG).predict.attacking evaluate

.PHONY: attacking-train
attacking-train:  ## Attacking rates: production fit
	$(UV) python -m $(PKG).predict.attacking train

.PHONY: xp-eval
xp-eval:  ## Expected points: score this season's finished GWs (unseen by every model)
	$(UV) python -m $(PKG).predict.aggregate evaluate --season $(SEASON)

.PHONY: xp
xp:  ## Expected points: next gameweek + 5 ahead, written to predictions
	$(UV) python -m $(PKG).predict.aggregate run

.PHONY: xp-actuals
xp-actuals:  ## Fill actual_points on predictions for finished gameweeks
	$(UV) python -m $(PKG).predict.aggregate actuals

.PHONY: optimise
optimise:  ## Squad optimiser. Optional: HITS=1 (allow hits), TRANSFERS=2 (force exactly 2)
	$(UV) python -m $(PKG).optimise.squad \
		$(if $(HITS),--max-hits $(HITS),) $(if $(TRANSFERS),--transfers $(TRANSFERS),)

.PHONY: wildcard
wildcard:  ## Squad optimiser with a wildcard this gameweek
	$(UV) python -m $(PKG).optimise.squad --wildcard

.PHONY: weekly
weekly: ingest-squad xp optimise  ## Weekly decision: reload squad, refresh xP, solve (takes HITS=, TRANSFERS=)

.PHONY: strength-current
strength-current:  ## Team strength: refresh projections for the current season only (daily job)
	$(UV) python -m $(PKG).predict.team_strength run --seasons $(SEASON)

# --------------------------------------------------------------------------
# dev
# --------------------------------------------------------------------------

.PHONY: fmt
fmt:  ## Format and autofix
	$(UV) ruff format src migrations
	$(UV) ruff check --fix src

.PHONY: lint
lint:  ## Lint without fixing
	$(UV) ruff check src

.PHONY: test
test:  ## Run tests
	$(UV) pytest -q

.PHONY: clean
clean:  ## Remove caches
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

LAUNCH_DIR := $(HOME)/Library/LaunchAgents
JOBS       := fpl-ingest-daily fpl-ingest-results

.PHONY: prune
prune:  ## Delete superseded daily model runs (keeps latest + last-before-each-deadline)
	$(UV) python -m $(PKG).prune

.PHONY: neon-copy
neon-copy:  ## One-off: copy the local Docker DB into an empty Neon DB (NEON_DATABASE_URL in .env)
	./scripts/copy_to_neon.sh

.PHONY: wait-network
wait-network:  ## Block until the FPL API is reachable (up to 10 min) — the Mac may have just woken
	@for i in $$(seq 1 60); do \
		curl -s -o /dev/null --max-time 5 https://fantasy.premierleague.com/api/ && exit 0; \
		sleep 10; \
	done; \
	echo "FPL API unreachable after 10 minutes"; exit 1

.PHONY: schedule
schedule:  ## Generate plists from JSON and load them into launchd
	@mkdir -p logs $(LAUNCH_DIR)
	@for j in $(JOBS); do \
		plutil -convert xml1 scripts/launchd/$$j.json \
			-o $(LAUNCH_DIR)/com.antonio.$$j.plist; \
		launchctl bootout gui/$$(id -u)/com.antonio.$$j 2>/dev/null || true; \
		launchctl bootstrap gui/$$(id -u) $(LAUNCH_DIR)/com.antonio.$$j.plist; \
		echo "loaded com.antonio.$$j"; \
	done

.PHONY: unschedule
unschedule:  ## Remove the scheduled jobs
	@for j in $(JOBS); do \
		launchctl bootout gui/$$(id -u)/com.antonio.$$j 2>/dev/null || true; \
		rm -f $(LAUNCH_DIR)/com.antonio.$$j.plist; \
		echo "removed com.antonio.$$j"; \
	done

.PHONY: schedule-status
schedule-status:  ## Show scheduled job state
	@launchctl list | grep -E 'PID|com\.antonio\.fpl' || echo "no jobs loaded"

.PHONY: schedule-test
schedule-test:  ## Fire the daily job immediately
	launchctl kickstart -p gui/$$(id -u)/com.antonio.fpl-ingest-daily

.PHONY: backfill-snapshots
backfill-snapshots:  ## Synthetic snapshots for gameweeks before live ingest started
	$(UV) python -m $(PKG).ingest.backfill snapshots

.PHONY: backfill-archive
backfill-archive:  ## Load past seasons: make backfill-archive ARCHIVE=~/data/fpl-archive
	$(UV) python -m $(PKG).ingest.backfill csv --archive $(ARCHIVE) --all