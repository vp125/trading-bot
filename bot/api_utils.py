"""Retry / error-classification helpers for flaky network and API conditions."""
from __future__ import annotations

import functools
import logging
import random
import time
from typing import Callable, TypeVar

import requests
from alpaca_trade_api.rest import APIError

import config

log = logging.getLogger(__name__)
T = TypeVar("T")

AUTH_STATUS = (401, 403)


def status_of(exc: Exception):
    return getattr(exc, "status_code", None) if isinstance(exc, APIError) else None


def is_transient(exc: Exception) -> bool:
    """True for disconnects, timeouts, rate limits and 5xx -- worth retrying."""
    if isinstance(exc, (requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                        requests.exceptions.ChunkedEncodingError,
                        ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, APIError):
        code = status_of(exc)
        return code == 429 or (code is not None and code >= 500)
    return False


def retry_api(fn: Callable[..., T]) -> Callable[..., T]:
    """Retry transient failures with exponential backoff + jitter.

    Non-transient errors (bad request, 403, 404, 422 ...) are raised
    immediately so callers can react to them (e.g. 404 = no such position).
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        delay = config.API_BACKOFF_BASE_S
        for attempt in range(1, config.API_MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if not is_transient(exc) or attempt == config.API_MAX_RETRIES:
                    raise
                sleep_for = min(delay, config.API_BACKOFF_MAX_S) * (0.8 + 0.4 * random.random())
                log.warning("%s failed (%s: %s); retry %d/%d in %.1fs",
                            fn.__name__, type(exc).__name__, exc, attempt,
                            config.API_MAX_RETRIES, sleep_for)
                time.sleep(sleep_for)
                delay *= 2
        raise RuntimeError("unreachable")  # pragma: no cover
    return wrapper
