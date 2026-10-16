"""
Team strength — Dixon-Coles with time decay.

Every team gets an attack and a defence coefficient on the log scale:

    log λ_home = μ + home + attack[home] + defence[away]
    log λ_away = μ        + attack[away] + defence[home]

`defence` is a log multiplier on goals CONCEDED, so higher means leakier.
Both are centred on zero, so 0 is an average Premier League side.

The Dixon-Coles τ term corrects the four low scorelines (0-0, 1-0, 0-1, 1-1)
that two independent Poissons get wrong. That matters here more than in
betting models, because 0 goals conceded IS the clean sheet.

Downstream uses:

    p_home_clean_sheet / p_away_clean_sheet   defender + keeper points
    lambda_home / lambda_away                 attacking fixture multiplier
                                              (λ_team / league mean λ)

TARGET. Goals are noisy and xG is a better measure of what a team does, but
xG only exists from 2022-23. The fit uses a blend, `xg_weight * xG +
(1 - xg_weight) * goals`, as a Poisson quasi-likelihood, and falls back to
goals where xG is missing. τ is always evaluated on actual goals, because it
describes how real scorelines cluster.

WALK-FORWARD. A fixture's projection comes from a fit that only saw matches
kicked off before that gameweek's deadline. Every gameweek gets its own fit,
which gives the backtest honest point-in-time projections for free. For
gameweeks still to come, the cutoff is "now".

PROMOTED TEAMS have no recent top-flight data, and shrinking them to an
average side would flatter them badly for the first month. They are shrunk
toward a below-average prior instead, which the data overrides within a few
weeks.

    # tune decay + xG weight on one season, report honestly on the next
    uv run python -m fantasy_26.predict.team_strength evaluate

    # write projections for every fixture from 2022-23 to now
    uv run python -m fantasy_26.predict.team_strength run
"""

from __future__ import annotations

import argparse
import itertools
import logging
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sqlalchemy import insert

from fantasy_26.db import SessionLocal, engine
from fantasy_26.models.schema import FixtureProjection, ModelRun, TeamStrength
from fantasy_26.predict.minutes import calibration_table, expected_calibration_error

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("team_strength")

MODEL_NAME = "team_strength_dc"
VERSION = "0.1"


@dataclass(frozen=True)
class DCConfig:
    # Defaults chosen by `evaluate` (tuned on 2024-25, checked on 2025-26).
    xi: float = 0.004  # decay per day; half-life = ln2 / xi ≈ 175 days
    xg_weight: float = 1.0  # 0 = goals only, 1 = xG only (goals where xG missing)
    # Ridge strength in pseudo-matches. A Gaussian prior with sd ≈ 1/sqrt(2k),
    # so k=2 says "team effects are usually within about ±0.5". 2 vs 5 vs 10
    # barely moved the score; it mostly matters for promoted teams.
    prior_matches: float = 2.0
    window_days: int = 1460  # beyond ~4 years the decay weight is negligible
    promoted_attack: float = -0.2
    promoted_defence: float = 0.2
    max_goals: int = 10


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

FIXTURES_SQL = """
WITH xg AS (
    SELECT fixture_id,
           SUM(expected_goals) FILTER (WHERE was_home)     AS home_xg,
           SUM(expected_goals) FILTER (WHERE NOT was_home) AS away_xg
    FROM player_fixture_stats
    GROUP BY fixture_id
)
SELECT f.id                     AS fixture_id,
       s.label                  AS season_label,
       s.start_year,
       f.gameweek_id,
       gw.deadline,
       f.kickoff_time,
       f.finished,
       f.home_score,
       f.away_score,
       f.home_team_season_id,
       f.away_team_season_id,
       hts.team_id              AS home_team,
       ats.team_id              AS away_team,
       -- seasons before FPL published xG store 0, not NULL; a real match
       -- never has zero xG on both sides, so treat that as missing
       CASE WHEN xg.home_xg + xg.away_xg > 0 THEN xg.home_xg END AS home_xg,
       CASE WHEN xg.home_xg + xg.away_xg > 0 THEN xg.away_xg END AS away_xg
FROM fixtures f
JOIN seasons s          ON s.id = f.season_id
JOIN team_seasons hts   ON hts.id = f.home_team_season_id
JOIN team_seasons ats   ON ats.id = f.away_team_season_id
LEFT JOIN gameweeks gw  ON gw.id = f.gameweek_id
LEFT JOIN xg            ON xg.fixture_id = f.id
WHERE f.kickoff_time IS NOT NULL
ORDER BY f.kickoff_time
"""

TEAM_SEASONS_SQL = """
SELECT ts.id AS team_season_id, ts.team_id, s.start_year
FROM team_seasons ts JOIN seasons s ON s.id = ts.season_id
"""


def load_fixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    fx = pd.read_sql(FIXTURES_SQL, engine)
    fx["kickoff_time"] = pd.to_datetime(fx["kickoff_time"], utc=True)
    fx["deadline"] = pd.to_datetime(fx["deadline"], utc=True)
    ts = pd.read_sql(TEAM_SEASONS_SQL, engine)
    return fx, ts


def promoted_teams(team_seasons: pd.DataFrame, start_year: int) -> set[int]:
    """Teams in this season that weren't in the previous one."""
    now = set(team_seasons.loc[team_seasons["start_year"] == start_year, "team_id"])
    before = set(team_seasons.loc[team_seasons["start_year"] == start_year - 1, "team_id"])
    if not before:  # first season on record: nothing to compare against
        return set()
    return now - before


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


@dataclass
class DCFit:
    teams: np.ndarray  # team_id for each index
    attack: np.ndarray
    defence: np.ndarray
    mu: float
    home: float
    rho: float

    def index(self, team_ids: np.ndarray) -> np.ndarray:
        lookup = {t: i for i, t in enumerate(self.teams)}
        return np.array([lookup[t] for t in team_ids])

    def lambdas(self, home_ids, away_ids, team_effects: bool = True):
        h, a = self.index(home_ids), self.index(away_ids)
        k = 1.0 if team_effects else 0.0
        lam = np.exp(self.mu + self.home + k * (self.attack[h] + self.defence[a]))
        nu = np.exp(self.mu + k * (self.attack[a] + self.defence[h]))
        return lam, nu


def _tau_terms(hg, ag, lam, nu, rho):
    """log τ and its partial derivatives for each match.

    τ only differs from 1 on the four low scorelines, so everything else
    contributes zeros.
    """
    tau = np.ones_like(lam)
    d_lam = np.zeros_like(lam)
    d_nu = np.zeros_like(lam)
    d_rho = np.zeros_like(lam)

    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    tau[m00] = 1 - lam[m00] * nu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + nu[m10] * rho
    tau[m11] = 1 - rho
    tau = np.maximum(tau, 1e-10)

    d_lam[m00] = -nu[m00] * rho / tau[m00]
    d_nu[m00] = -lam[m00] * rho / tau[m00]
    d_rho[m00] = -lam[m00] * nu[m00] / tau[m00]
    d_lam[m01] = rho / tau[m01]
    d_rho[m01] = lam[m01] / tau[m01]
    d_nu[m10] = rho / tau[m10]
    d_rho[m10] = nu[m10] / tau[m10]
    d_rho[m11] = -1 / tau[m11]

    return np.log(tau), d_lam, d_nu, d_rho


def fit_dixon_coles(
    train: pd.DataFrame,
    cutoff: pd.Timestamp,
    cfg: DCConfig,
    extra_teams: np.ndarray,
    promoted: set[int],
) -> DCFit:
    """Weighted maximum likelihood with an analytic gradient.

    Team effects are centred inside the objective, so the raw parameters have
    a flat direction. The ridge prior removes it and also gives teams with
    little data (or none) a sensible value.
    """
    teams = np.unique(np.concatenate([train["home_team"], train["away_team"], extra_teams]))
    n = len(teams)
    lookup = {t: i for i, t in enumerate(teams)}
    h = train["home_team"].map(lookup).to_numpy()
    a = train["away_team"].map(lookup).to_numpy()

    hg = train["home_score"].to_numpy(float)
    ag = train["away_score"].to_numpy(float)
    hx = train["home_xg"].fillna(train["home_score"]).to_numpy(float)
    ax = train["away_xg"].fillna(train["away_score"]).to_numpy(float)
    tx = cfg.xg_weight * hx + (1 - cfg.xg_weight) * hg
    ty = cfg.xg_weight * ax + (1 - cfg.xg_weight) * ag

    days_ago = (cutoff - train["kickoff_time"]).dt.total_seconds().to_numpy() / 86400
    w = np.exp(-cfg.xi * days_ago)
    W = w.sum()

    prior_a = np.array([cfg.promoted_attack if t in promoted else 0.0 for t in teams])
    prior_d = np.array([cfg.promoted_defence if t in promoted else 0.0 for t in teams])
    k = cfg.prior_matches

    def objective(theta):
        mu, home, rho = theta[:3]
        att = theta[3 : 3 + n] - theta[3 : 3 + n].mean()
        dfc = theta[3 + n :] - theta[3 + n :].mean()

        eh = mu + home + att[h] + dfc[a]
        ea = mu + att[a] + dfc[h]
        lam, nu = np.exp(eh), np.exp(ea)
        log_tau, dl_lam, dl_nu, dl_rho = _tau_terms(hg, ag, lam, nu, rho)

        ll = w * (tx * eh - lam + ty * ea - nu + log_tau)
        penalty = k * (((att - prior_a) ** 2).sum() + ((dfc - prior_d) ** 2).sum())
        f = (-ll.sum() + penalty) / W

        gh = w * (tx - lam + lam * dl_lam)
        ga = w * (ty - nu + nu * dl_nu)
        g_att = -(np.bincount(h, gh, n) + np.bincount(a, ga, n)) + 2 * k * (att - prior_a)
        g_dfc = -(np.bincount(a, gh, n) + np.bincount(h, ga, n)) + 2 * k * (dfc - prior_d)
        grad = np.concatenate(
            [
                [-(gh + ga).sum(), -gh.sum(), -(w * dl_rho).sum()],
                g_att - g_att.mean(),
                g_dfc - g_dfc.mean(),
            ]
        )
        return f, grad / W

    theta0 = np.zeros(3 + 2 * n)
    theta0[0], theta0[1] = np.log(1.3), 0.2
    bounds = [(None, None), (None, None), (-0.25, 0.25)] + [(None, None)] * (2 * n)
    res = minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        logger.warning("fit at %s did not converge: %s", cutoff, res.message)

    att = res.x[3 : 3 + n] - res.x[3 : 3 + n].mean()
    dfc = res.x[3 + n :] - res.x[3 + n :].mean()
    return DCFit(teams, att, dfc, float(res.x[0]), float(res.x[1]), float(res.x[2]))


def score_matrix(lam: np.ndarray, nu: np.ndarray, rho: float, max_goals: int) -> np.ndarray:
    """P(home=x, away=y) for every fixture, shape (n, G+1, G+1)."""
    g = np.arange(max_goals + 1)
    px = poisson.pmf(g[None, :], lam[:, None])
    py = poisson.pmf(g[None, :], nu[:, None])
    m = px[:, :, None] * py[:, None, :]
    m[:, 0, 0] *= 1 - lam * nu * rho
    m[:, 0, 1] *= 1 + lam * rho
    m[:, 1, 0] *= 1 + nu * rho
    m[:, 1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum(axis=(1, 2), keepdims=True)  # truncation at max_goals


def project(fit: DCFit, fixtures: pd.DataFrame, cfg: DCConfig, team_effects=True) -> pd.DataFrame:
    lam, nu = fit.lambdas(
        fixtures["home_team"].to_numpy(), fixtures["away_team"].to_numpy(), team_effects
    )
    m = score_matrix(lam, nu, fit.rho, cfg.max_goals)
    out = pd.DataFrame(index=fixtures.index)
    out["lambda_home"] = lam
    out["lambda_away"] = nu
    out["p_home_clean_sheet"] = m[:, :, 0].sum(axis=1)
    out["p_away_clean_sheet"] = m[:, 0, :].sum(axis=1)
    out["p_home_win"] = np.tril(m, -1).sum(axis=(1, 2))
    out["p_draw"] = np.trace(m, axis1=1, axis2=2)
    out["p_away_win"] = np.triu(m, 1).sum(axis=(1, 2))

    # scoreline probability, kept for evaluation only
    done = fixtures["finished"].to_numpy() & fixtures["home_score"].notna().to_numpy()
    p_score = np.full(len(fixtures), np.nan)
    if done.any():
        x = fixtures["home_score"].to_numpy()[done].astype(int).clip(0, cfg.max_goals)
        y = fixtures["away_score"].to_numpy()[done].astype(int).clip(0, cfg.max_goals)
        p_score[done] = m[np.flatnonzero(done), x, y]
    out["p_scoreline"] = p_score
    return out


# --------------------------------------------------------------------------
# walk-forward
# --------------------------------------------------------------------------


def walk_forward(
    fx: pd.DataFrame,
    team_seasons: pd.DataFrame,
    seasons: list[str],
    cfg: DCConfig,
    now: Optional[pd.Timestamp] = None,
    with_baseline: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project every fixture in `seasons` using only matches before its cutoff.

    Returns (projections, strengths). With `with_baseline`, projections also
    carry `base_*` columns from the same fit with team effects switched off —
    the "every team is average" model that this has to beat.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    targets = fx[fx["season_label"].isin(seasons)].copy()
    cutoff = targets["deadline"].fillna(targets["kickoff_time"])
    targets["cutoff"] = cutoff.where(cutoff < now, now)

    history = fx[fx["finished"] & fx["home_score"].notna()]
    projections, strengths = [], []

    groups = list(targets.groupby(["cutoff", "start_year"]))
    logger.info("walk-forward: %d fixtures, %d fits", len(targets), len(groups))
    for (cut, start_year), group in groups:
        train = history[
            (history["kickoff_time"] < cut)
            & (history["kickoff_time"] >= cut - timedelta(days=cfg.window_days))
        ]
        promoted = promoted_teams(team_seasons, start_year)
        season_teams = team_seasons[team_seasons["start_year"] == start_year]
        fit = fit_dixon_coles(
            train, cut, cfg, season_teams["team_id"].to_numpy(), promoted
        )

        p = project(fit, group, cfg)
        if with_baseline:
            base = project(fit, group, cfg, team_effects=False)
            p = p.join(base.add_prefix("base_"))
        projections.append(group.join(p))

        idx = fit.index(season_teams["team_id"].to_numpy())
        strengths.append(
            pd.DataFrame(
                {
                    "team_season_id": season_teams["team_season_id"].to_numpy(),
                    "as_of": cut.date(),
                    "attack": fit.attack[idx],
                    "defence": fit.defence[idx],
                    "home_advantage": fit.home,
                    "rho": fit.rho,
                }
            )
        )

    return pd.concat(projections), pd.concat(strengths)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def metrics_for(proj: pd.DataFrame, prefix: str = "") -> dict:
    """Proper scoring rules only, plus ECE on the clean-sheet probabilities.

    `scoreline_ll` (−log P(exact score)) is the headline: it scores the whole
    joint distribution, so it rewards getting both λs and ρ right at once.
    """
    done = proj[proj["finished"] & proj["home_score"].notna()]
    if done.empty:
        return {}
    hs, as_ = done["home_score"].to_numpy(), done["away_score"].to_numpy()
    eps = 1e-12

    outcome = np.select([hs > as_, hs == as_], [0, 1], default=2)
    p1x2 = done[[f"{prefix}p_home_win", f"{prefix}p_draw", f"{prefix}p_away_win"]].to_numpy()
    y_cs = np.concatenate([(as_ == 0), (hs == 0)]).astype(int)
    p_cs = np.concatenate(
        [done[f"{prefix}p_home_clean_sheet"], done[f"{prefix}p_away_clean_sheet"]]
    ).clip(eps, 1 - eps)

    return {
        "n": int(len(done)),
        "scoreline_ll": float(-np.log(done[f"{prefix}p_scoreline"].clip(eps)).mean()),
        "result_ll": float(-np.log(p1x2[np.arange(len(done)), outcome].clip(eps)).mean()),
        "cs_ll": float(-(y_cs * np.log(p_cs) + (1 - y_cs) * np.log(1 - p_cs)).mean()),
        "cs_ece": expected_calibration_error(y_cs, p_cs),
    }


def _log_metrics(label: str, m: dict) -> None:
    logger.info(
        "%-28s n=%4d  scoreline=%.4f  result=%.4f  cs_ll=%.4f  cs_ECE=%.4f",
        label,
        m["n"],
        m["scoreline_ll"],
        m["result_ll"],
        m["cs_ll"],
        m["cs_ece"],
    )


def evaluate(tune_season: str, valid_season: str) -> tuple[DCConfig, dict]:
    """Pick decay and xG weight on one season, then report on the next.

    Selecting on the season you report would make the headline number
    optimistic, the same way early-stopping on the validation set would.
    """
    fx, ts = load_fixtures()
    base_cfg = DCConfig()

    grid = list(
        itertools.product(
            [0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.012],
            [0.0, 0.5, 0.75, 1.0],
            [2.0, 5.0, 10.0],
        )
    )
    results = []
    for xi, xgw, k in grid:
        cfg = replace(base_cfg, xi=xi, xg_weight=xgw, prior_matches=k)
        proj, _ = walk_forward(fx, ts, [tune_season], cfg)
        m = metrics_for(proj)
        results.append((cfg, m))
        _log_metrics(f"tune xi={xi} xg_w={xgw} k={k}", m)

    best_cfg, _ = min(results, key=lambda r: r[1]["scoreline_ll"])
    logger.info(
        "selected xi=%s xg_weight=%s prior_matches=%s",
        best_cfg.xi,
        best_cfg.xg_weight,
        best_cfg.prior_matches,
    )

    proj, _ = walk_forward(fx, ts, [valid_season], best_cfg, with_baseline=True)
    goals_only, _ = walk_forward(fx, ts, [valid_season], replace(best_cfg, xg_weight=0.0))
    metrics = {
        "model": metrics_for(proj),
        "goals_only": metrics_for(goals_only),
        "average_team": metrics_for(proj, prefix="base_"),
        "tune_season": tune_season,
        "valid_season": valid_season,
        "grid": [
            {"xi": c.xi, "xg_weight": c.xg_weight, "prior_matches": c.prior_matches, **m}
            for c, m in results
        ],
    }
    _log_metrics(f"VALID {valid_season} model", metrics["model"])
    _log_metrics(f"VALID {valid_season} goals only", metrics["goals_only"])
    _log_metrics(f"VALID {valid_season} average team", metrics["average_team"])

    print("\nClean sheet calibration (validation season):")
    done = proj[proj["finished"]]
    y = np.concatenate([(done["away_score"] == 0), (done["home_score"] == 0)]).astype(int)
    p = np.concatenate([done["p_home_clean_sheet"], done["p_away_clean_sheet"]])
    print(calibration_table(y, p).to_string())
    return best_cfg, metrics


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def run(seasons: list[str], cfg: DCConfig) -> int:
    """Walk forward over `seasons` and store every projection and strength."""
    fx, ts = load_fixtures()
    now = pd.Timestamp.now(tz="UTC")
    proj, strengths = walk_forward(fx, ts, seasons, cfg, now=now)
    metrics = metrics_for(proj)
    if metrics:
        _log_metrics("walk-forward (finished)", metrics)

    strengths = strengths.drop_duplicates(["team_season_id", "as_of"], keep="last")

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        sha = None

    cols = [
        "lambda_home",
        "lambda_away",
        "p_home_clean_sheet",
        "p_away_clean_sheet",
        "p_home_win",
        "p_draw",
        "p_away_win",
    ]
    with SessionLocal() as session:
        mr = ModelRun(
            name=MODEL_NAME,
            version=VERSION,
            trained_at=datetime.now(timezone.utc),
            train_cutoff=now.to_pydatetime(),
            config={**asdict(cfg), "seasons": seasons},
            metrics=metrics,
            git_sha=sha,
            note="walk-forward Dixon-Coles; each fixture fitted at its own deadline",
        )
        session.add(mr)
        session.flush()

        rows = proj[["fixture_id", *cols]].assign(model_run_id=mr.id)
        session.execute(insert(FixtureProjection), rows.to_dict("records"))
        session.execute(
            insert(TeamStrength),
            strengths.assign(model_run_id=mr.id).to_dict("records"),
        )
        session.commit()
        run_id = mr.id

    logger.info(
        "model run %d: %d fixture projections, %d strength rows",
        run_id,
        len(proj),
        len(strengths),
    )
    return run_id


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Team strength (Dixon-Coles)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("evaluate")
    e.add_argument("--tune", default="2024-25")
    e.add_argument("--valid", default="2025-26")

    r = sub.add_parser("run")
    r.add_argument(
        "--seasons", nargs="+", default=["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
    )
    r.add_argument("--xi", type=float, default=DCConfig.xi)
    r.add_argument("--xg-weight", type=float, default=DCConfig.xg_weight)
    r.add_argument("--prior-matches", type=float, default=DCConfig.prior_matches)

    args = parser.parse_args()
    if args.cmd == "evaluate":
        evaluate(args.tune, args.valid)
    elif args.cmd == "run":
        cfg = DCConfig(
            xi=args.xi, xg_weight=args.xg_weight, prior_matches=args.prior_matches
        )
        run(args.seasons, cfg)


if __name__ == "__main__":
    main()
