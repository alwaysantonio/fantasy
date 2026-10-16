"""
HTTP client for the public Fantasy Premier League API.

All endpoints used here are public — no authentication, no cookies. The only
private endpoint is the *pending* gameweek's picks, which we deliberately do
not touch (see `entry_picks` docstring).

Be polite: element-summary is ~700 calls for a full refresh. The semaphore
caps concurrency and there's a small delay between requests. There is no
published rate limit, but hammering it will get you blocked by Cloudflare.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "fantasy-26/0.1 (personal FPL tool)"


@dataclass(slots=True)
class FetchResult:
    """A single API response plus the metadata needed to archive it."""

    endpoint: str
    params: Optional[str]
    fetched_at: datetime
    status_code: int
    payload: Any
    content_hash: str

    @classmethod
    def build(
        cls,
        endpoint: str,
        params: Optional[str],
        status_code: int,
        payload: Any,
    ) -> "FetchResult":
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return cls(
            endpoint=endpoint,
            params=params,
            fetched_at=datetime.now(timezone.utc),
            status_code=status_code,
            payload=payload,
            content_hash=hashlib.sha256(blob.encode()).hexdigest(),
        )


class FPLClient:
    """Async client. Use as a context manager:

        async with FPLClient() as client:
            boot = await client.bootstrap_static()
    """

    def __init__(
        self,
        base_url: str = "https://fantasy.premierleague.com/api",
        *,
        timeout: float = 30.0,
        max_concurrency: int = 4,
        max_retries: int = 4,
        polite_delay: float = 0.15,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.polite_delay = polite_delay
        self._sem = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            follow_redirects=True,
        )

    async def __aenter__(self) -> "FPLClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self._client.aclose()

    # -- core ------------------------------------------------------------

    async def _get(self, path: str, params: Optional[str] = None) -> FetchResult:
        """GET with exponential backoff on 429/5xx/network errors."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        delay = 1.0

        async with self._sem:
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = await self._client.get(url)

                    if resp.status_code == 429:
                        wait = float(resp.headers.get("Retry-After", delay))
                        logger.warning("429 on %s, sleeping %.1fs", path, wait)
                        await asyncio.sleep(wait)
                        delay *= 2
                        continue

                    if resp.status_code >= 500:
                        logger.warning(
                            "%s on %s, attempt %d", resp.status_code, path, attempt
                        )
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue

                    resp.raise_for_status()
                    await asyncio.sleep(self.polite_delay)
                    return FetchResult.build(path, params, resp.status_code, resp.json())

                except httpx.HTTPStatusError:
                    raise  # 4xx other than 429 — genuine error, don't retry
                except (httpx.RequestError, json.JSONDecodeError) as exc:
                    logger.warning("%s on %s, attempt %d", type(exc).__name__, path, attempt)
                    if attempt == self.max_retries:
                        raise
                    await asyncio.sleep(delay)
                    delay *= 2

        raise RuntimeError(f"exhausted retries for {path}")

    # -- endpoints -------------------------------------------------------

    async def bootstrap_static(self) -> FetchResult:
        """Everything global: events (gameweeks), teams, elements (players),
        element_types, phases. ~2MB. This is your primary daily pull."""
        return await self._get("bootstrap-static/")

    async def fixtures(self, event: Optional[int] = None) -> FetchResult:
        """All fixtures for the season. Pass `event` to filter to one GW.

        Fixtures with a null `event` are postponed/unscheduled — keep them,
        they get a gameweek assigned later.
        """
        path = "fixtures/" if event is None else f"fixtures/?event={event}"
        return await self._get(path, params=str(event) if event else None)

    async def element_summary(self, element_id: int) -> FetchResult:
        """Per-player detail: `history` (this season, per fixture),
        `history_past` (season totals for prior seasons), `fixtures` (upcoming).

        NOTE: `history` only covers the current season. Past per-gameweek detail
        is not available here — that's why you need the DB, and why the
        vaastav/Fantasy-Premier-League repo is worth backfilling from.
        """
        return await self._get(f"element-summary/{element_id}/", params=str(element_id))

    async def element_summaries(self, element_ids: list[int]) -> list[FetchResult]:
        """Fetch many players concurrently, bounded by the semaphore."""
        tasks = [self.element_summary(eid) for eid in element_ids]
        results: list[FetchResult] = []
        for coro in asyncio.as_completed(tasks):
            try:
                results.append(await coro)
            except Exception as exc:  # one bad player shouldn't kill the run
                logger.error("element_summary failed: %s", exc)
        return results

    async def event_live(self, event: int) -> FetchResult:
        """Live stats for every player in a gameweek. Cheaper than 700
        element-summary calls when you only need one GW's results."""
        return await self._get(f"event/{event}/live/", params=str(event))

    # -- your team (all public) -------------------------------------------

    async def entry(self, entry_id: int) -> FetchResult:
        """Manager metadata: name, region, current GW, overall rank."""
        return await self._get(f"entry/{entry_id}/", params=str(entry_id))

    async def entry_history(self, entry_id: int) -> FetchResult:
        """Per-GW points, rank, bank, team value, transfers, chips."""
        return await self._get(f"entry/{entry_id}/history/", params=str(entry_id))

    async def entry_transfers(self, entry_id: int) -> FetchResult:
        """Every transfer this season with `element_in_cost`. That's the only
        public source of purchase prices, which selling prices depend on."""
        return await self._get(f"entry/{entry_id}/transfers/", params=str(entry_id))

    async def entry_picks(self, entry_id: int, event: int) -> FetchResult:
        """Squad for a gameweek.

        Only works once the deadline for `event` has passed — before that the
        endpoint 404s, because pending picks are private. This is the whole
        reason you don't need to automate login: pull the last completed GW's
        squad and patch in any transfers you've since made by hand.
        """
        return await self._get(
            f"entry/{entry_id}/event/{event}/picks/", params=f"{entry_id}:{event}"
        )
