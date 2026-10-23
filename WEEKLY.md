# Weekly use

How to get your team decisions each gameweek. Everything else (data,
predictions, your squad) updates on its own every morning.

## Every gameweek

1. **On deadline day, before you change anything in the FPL app**, run:

   ```
   make weekly
   ```

   This reloads your squad, refreshes expected points and runs the optimiser.

2. **Read the line under the team.** It will be one of:

   | Output | What to do |
   |---|---|
   | `Transfers (1 of 1 free): X → Y` | Make that transfer |
   | `No transfer this week: roll it (you'll have 2 free next week)` | Change nothing; the transfer is banked |
   | `Transfers (2 of 2 free): ...` | Use both banked transfers |

   Hits (−4) are never suggested by default. Two options change that:

   | Command | What it does |
   |---|---|
   | `make weekly HITS=1` | Allows up to 1 hit per week; it takes it only if the extra transfer gains more than 4 points |
   | `make weekly TRANSFERS=2` | Makes exactly 2 transfers this week, taking whatever hits that needs |

   `TRANSFERS` is for when you've already decided how many players to change
   and want to know who. Compare the `net of N hit(s)` total against a plain
   `make weekly`. If it's lower, the hit isn't worth it.

3. **Set your XI, captain, vice-captain and bench order** exactly as listed
   (`XI (C)`, `XI (V)`, `bench 1` to `bench 4`).

4. **Check the news.** The model only knows what FPL's injury flags say. If a
   press conference rules someone out before FPL flags him, the model won't
   know.

## Wildcard week

Run this instead of `make weekly`, then play the Wildcard chip in the app and
build the squad it lists:

```
make xp && make wildcard
```

## Things to know

- **Run it before making transfers in the app.** Your picks only become public
  once the deadline passes, so the model can't see transfers you've already
  made for the coming week.
- **Run it after 09:00 on deadline day.** The 08:00 job picks up the latest
  injury flags and prices, and the 09:00 job refreshes predictions. Running
  `make weekly` yourself refreshes predictions anyway, but the 08:00 snapshot
  is what carries the news.
- **The plan for later weeks is not a commitment.** It shows why this week's
  move makes sense, and it is re-solved every week with new predictions.
- **Free transfers are worked out from your FPL history** (banked up to five),
  so you don't need to tell it how many you have.

## On GitHub Actions

Once the project is pushed and the secrets are set (see below), you don't need
your Mac at all:

- **Every morning (06:17 UTC, with a backup at 09:47 UTC):** the **daily** workflow runs everything below,
  then writes the optimiser's advice for today to the run's summary page.
  Open the Actions tab and click the latest *daily* run to read it.
- **On deadline day:** Actions tab, then **weekly**, then **Run workflow**.
  Pick wildcard, hits (0-2) and forced transfers (auto or 1-5); these are the
  same as `--wildcard`, `HITS=` and `TRANSFERS=`. The answer appears on that
  run's summary page.

If a run fails, GitHub emails you. Re-run it from the same page.

### One-off setup

1. Create a Neon project on **Postgres 17** and copy its **direct**
   (non-pooler) connection string.
2. Add it to `.env` as `NEON_DATABASE_URL=...`, then run `make neon-copy`.
   This copies your existing data in and refuses to run if the Neon database
   isn't empty.
3. In `.env`, set `DATABASE_URL=` to the same Neon URL. Local `make` commands
   now use Neon too.
4. Commit and push, including `models/` (the workflows load the trained models
   from the repo) and `.github/`.
5. Add the repository secrets:
   ```
   gh secret set DATABASE_URL      # paste the Neon URL when prompted
   gh secret set ENTRY_ID --body 5280561
   ```
6. Actions tab → **daily** → **Run workflow**. Check it goes green.
7. Only then switch off the Mac jobs: `make unschedule`. Doing this earlier
   risks a missed snapshot day; leaving them on writes duplicate snapshots.

## What runs automatically (Mac, before moving to Actions)

| Time | Job |
|---|---|
| 08:00 | Player prices, injury flags and ownership snapshot; reload your squad |
| 09:00 | Match results; team strength; expected points for the next 6 gameweeks; fill in actual points |

Check the jobs are loaded with `make schedule-status`. Each job should show
status `0`. A missed snapshot day can't be recovered, so if the Mac was off
for a while, run `make ingest-daily` by hand.

Logs are in `logs/fpl-ingest-daily.log` and `logs/fpl-ingest-results.log`.
