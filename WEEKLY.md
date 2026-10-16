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

   Hits (−4) are never suggested by default. To let it consider one:
   `uv run python -m fantasy_26.optimise.squad --max-hits 1`

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

## What runs automatically

| Time | Job |
|---|---|
| 08:00 | Player prices, injury flags and ownership snapshot; reload your squad |
| 09:00 | Match results; team strength; expected points for the next 6 gameweeks; fill in actual points |

Check the jobs are loaded with `make schedule-status`. Each job should show
status `0`. A missed snapshot day can't be recovered, so if the Mac was off
for a while, run `make ingest-daily` by hand.

Logs are in `logs/fpl-ingest-daily.log` and `logs/fpl-ingest-results.log`.
