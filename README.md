# fantasy-26

Fantasy Premier League squad optimiser. Predicts expected points per player per
gameweek, then solves for the best starting XI, captain and transfers.

**Picking your team each week? See [WEEKLY.md](WEEKLY.md).**

Two separate problems, deliberately kept apart:

1. **Prediction** — expected points per player (statistics / ML)
2. **Decision** — which XI, which captain, which transfer (integer programming)

Most of the value is in the second half. A decent predictor whose output you
eyeball in a sorted list throws away most of the gain.

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker (for Postgres)
- `make`

---

## First-time setup

```bash
git clone <your-repo> fantasy-26
cd fantasy-26
cp .env.example .env     # then edit: set entry_id
make setup
```

`make setup` runs: `uv sync` → start Postgres → wait for it → apply migrations →
ingest reference data.

Verify:

```bash
make status
```

Expect ~20 teams, ~700 players, 380 fixtures for the current season.

---

## Daily operation

| Command | When | What it does |
|---|---|---|
| `make ingest-daily` | every day, ~02:15 UTC | bootstrap + snapshot |
| `make ingest-results` | after each GW is `data_checked` | per-fixture player stats |
| `make ingest-squad` | after each deadline | your squad for the last completed GW |

**Snapshot gaps are permanent.** No endpoint tells you what a player's price or
injury status was last Tuesday. If the job doesn't run, that day is gone. Put it
on a scheduler before writing any modelling code:

```cron
15 2 * * * cd ~/personal/fantasy-26 && /usr/local/bin/make ingest-daily >> /tmp/fpl.log 2>&1
```

Check it's been running:

```bash
make ingest-log
```

---

## All commands

```bash
make           # list every target
```

### Docker / Postgres

| Target | Effect |
|---|---|
| `make up` | start Postgres |
| `make down` | stop it, keep data |
| `make nuke` | stop it, **delete the volume** |
| `make psql` | open a psql shell |
| `make logs` | tail Postgres logs |

### Migrations

| Target | Effect |
|---|---|
| `make migrate` | apply pending migrations |
| `make migration m="add foo"` | autogenerate a new one |
| `make downgrade` | roll back one |
| `make current` / `make history` | inspect state |

Always open the generated file in `migrations/versions/` before applying it.
Autogenerate misreads enums and indexes often enough to be worth the ten seconds.

### Rebuilding from scratch

```bash
make reset     # nuke + up + migrate + bootstrap
```

Cheap at this stage, and usually faster than hand-editing a migration that
fails because a new non-nullable column has no default.

---

## Project layout

```
fantasy-26/
├── Makefile
├── alembic.ini
├── docker-compose.yaml
├── pyproject.toml
├── migrations/              # project tooling, NOT importable code
│   └── versions/
└── src/fantasy_26/
    ├── config.py            # pydantic-settings
    ├── db.py                # engine, SessionLocal
    ├── models/schema.py     # SQLAlchemy models
    ├── ingest/
    │   ├── client.py        # async FPL API client
    │   ├── loaders.py       # upsert logic
    │   └── jobs.py          # CLI entry points
    ├── features/builder.py  # deadline-aware rolling features
    ├── predict/             # minutes, team strength, attack, defcon, bonus
    ├── optimise/            # the ILP solver
    └── api/                 # FastAPI app
```

`migrations/` sits at the root, not under `src/`. `src/` holds only what gets
installed into site-packages; Alembic loads version files by path and never
imports them.

---

## Data model: the one rule

**Anything mutable is appended, never updated.**

`player_snapshots` gets a new row per player per day capturing price, ownership,
availability, news and position. Features for gameweek N filter to
`captured_at < gameweek.deadline`.

Update those in place instead and you'll train on injury news that only appeared
after the deadline, get an excellent validation score, and lose rank every week.
That failure mode is much worse than a crash because nothing visibly breaks.

`features/builder.py` enforces this in two places: `_lagged_roll` shifts by one
row before rolling, and `attach_snapshots` uses `merge_asof(direction="backward",
allow_exact_matches=False)`. `assert_no_leakage()` is the cheap guard — run it
after every feature change.

---

## Model stack

| # | Model | Method | Feeds |
|---|---|---|---|
| 1 | Team strength | Dixon-Coles / Poisson MLE | clean sheets, fixture multiplier |
| 2 | **Minutes** | LightGBM multiclass (0 / 1-59 / 60+) | gates everything downstream |
| 3 | Attacking rates | shrunk xG90 / xA90 × fixture | goals, assists |
| 4 | DefCon | negative binomial over threshold | +2 pts |
| 5 | Saves | Poisson, step at every 3 | GK only |
| 6 | Bonus | BPS prediction → rank within fixture | 0-3 pts |
| 7 | Negatives | shrunk historical rates | cards, OGs |
| 8 | Price change | logistic regression | transfer timing |

Build order if going incrementally: minutes first (biggest single gain), then
team strength, then attacking rates. Bonus, DefCon and saves add a few percent
each and can wait.

Expected points is the sum of components. **Store the breakdown, not just the
total** — when xP is wrong, `predicted 4.2, actual 12` tells you nothing, but the
component split tells you which sub-model missed.

Realistic ceiling: correlation ~0.4–0.5 between predicted and actual weekly
points. Anyone claiming much more is overfitting.

---

## Selection is not a model

Given xP, choosing the XI is integer linear programming (PuLP), not ML:

- 11 players, 1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD
- captain doubles
- transfers: budget, max 3 per club, −4 per extra transfer
- solve over a 4–6 GW horizon with ~0.85 weekly decay

The multi-week horizon is where rank is actually gained. Single-week greedy
transfers are usually a mistake.

---

## Historical backfill

One season of data trains nothing. Clone the archive:

```bash
git clone --depth 1 https://github.com/vaastav/Fantasy-Premier-League.git ~/data/fpl-archive
```

Load `data/{season}/gws/merged_gw.csv` — roughly one row per player per gameweek,
mapping closely onto `player_fixture_stats`.

Two wrinkles:

- **No snapshots exist** for past seasons, so `attach_snapshots` drops every
  historical row. Synthesise minimal snapshots from the CSV's `value` and
  `element_type` columns; injury news and ownership simply aren't recoverable.
- **Join on `code`, not `element`.** FPL reassigns element ids every August.
  `players_raw.csv` carries both and is the bridge.

Coverage limits: `expected_goals` from 2022-23 onward, defensive contribution
stats only from 2025-26. The DefCon model is stuck with roughly one season of
history regardless of how far back you load.

---

## Configuration

`.env`:

```ini
DATABASE_URL=postgresql+psycopg://fpl:fpl@localhost:5432/fpl
FPL_BASE_URL=https://fantasy.premierleague.com/api
ENTRY_ID=1234567
```

`ENTRY_ID` is in your FPL team page URL.

---

## A note on authentication

There isn't any, and none is needed. These are all public:

- `bootstrap-static/` — players, teams, gameweeks
- `fixtures/` — all fixtures
- `element-summary/{id}/` — per-player history
- `entry/{id}/history/` — your season
- `entry/{id}/event/{gw}/picks/` — your squad, **once that GW's deadline passes**

Only the *pending* gameweek's picks are private. So the workflow pulls your last
completed squad and you patch in any transfers by hand — about ten seconds, and
it removes any need to automate a Cloudflare-protected login.