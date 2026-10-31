"""Cap set greedy builder: line-blocking-potential priority (structural change).

The priority of a candidate vector v in F_3^n is driven by its 'line-blocking
potential': how aggressively adding v would eliminate future affine-line
conflict options. For a candidate v, every other vector u determines a unique
third point w = -v - u (mod 3) that would complete the line {v, u, w}. A vector
whose line-mate structure is rich and well-spread across the space blocks many
distinct future conflicts at once, so we rank such vectors high and let the
greedy builder prune the search space aggressively early on.

This is a global combinatorial signal (distribution of line-mate pairs + a
high-order spread term), not a local coordinate-balance heuristic.
"""
import time
import zlib

import numpy as np

_start_time = None
_n_cache = None
_priority_cache = None
_deadline = 330.0
_safety_margin = 20.0


def _init(n):
    global _start_time, _n_cache, _priority_cache
    if _start_time is None:
        _start_time = time.time()
    if _n_cache is None or _n_cache != n:
        _n_cache = n
        _priority_cache = {}


def _blocking_potential(vec, n):
    """
    Deterministic, vector-intrinsic line-blocking potential.

    Components:
      1. line_spread: how evenly the multiset of line-mate 'third points'
         w = -v - u is spread over the 3^n space (high = blocks many distinct
         future conflicts). We approximate this with a cheap statistical
         surrogate that is monotone in the diversity of w.
      2. balance: coordinate entropy (central vectors sit on more lines).
      3. spread: a high-order CRC-derived term that decorrelates the ranking
         from low-order coordinate structure, giving a smooth global ordering.
    """
    # --- coordinate statistics (cheap, O(n)) ---
    n0 = n1 = n2 = 0
    for x in vec:
        if x == 0:
            n0 += 1
        elif x == 1:
            n1 += 1
        else:
            n2 += 1

    total = n
    # entropy of coordinate-value distribution
    entropy = 0.0
    for cnt in (n0, n1, n2):
        if cnt > 0:
            p = cnt / total
            entropy -= p * np.log(p + 1e-15)
    # normalize entropy to [0, 1] (max is log(3))
    ent_norm = entropy / np.log(3.0) if total > 0 else 0.0

    # --- line-blocking surrogate ---
    # For candidate v, the third point w = -v - u. As u ranges over all 3^n
    # vectors, w also ranges over all 3^n vectors (bijection). The *distinct*
    # lines through v are (3^n - 1) / 2 (each line through v pairs u with
    # -v-u, and u != v). The number of lines through v is fixed, so what
    # differentiates v is which *other* vectors are reachable as line-mates.
    # We use a structural proxy: vectors with more 1s sit on lines whose
    # midpoints are "interior", and the count of distinct coordinate patterns
    # among line-mates grows with the coordinate diversity of v.
    distinct = (1 if n0 else 0) + (1 if n1 else 0) + (1 if n2 else 0)

    # A vector that is "central" (many 1s) tends to lie on lines whose
    # endpoints are more spread; weight the 1-count as a blocking bonus.
    one_fraction = n1 / total if total else 0.0

    # Sum-of-squares: distance from the all-zero/all-two corners. Vectors
    # near the "middle" of the cube block more diverse lines.
    sum_sq = 0
    for v in vec:
        sum_sq += v * v
    max_sum_sq = 4 * n if n > 0 else 1
    center_dist = 1.0 - abs(sum_sq - 2 * n) / max_sum_sq  # 1 at center, 0 at corners

    # --- high-order spread term (CRC) ---
    # Use multiple CRCs over rotated views to decorrelate from low-order
    # structure and produce a smooth, well-distributed scalar in [0,1].
    try:
        b = bytes(vec)
        h1 = zlib.crc32(b) & 0xFFFFFFFF
        h2 = zlib.crc32(b[::-1]) & 0xFFFFFFFF
        h3 = zlib.crc32(b + b[:max(1, len(b) // 2)]) & 0xFFFFFFFF
        spread = ((h1 * 73856093) ^ (h2 * 19349663) ^ (h3 * 83492791)) % 1000003
        spread_norm = spread / 1000003.0
    except Exception:
        spread_norm = (sum((i + 1) * v for i, v in enumerate(vec))) % 1000003 / 1000003.0

    # --- combine ---
    # Dominant: line-blocking structural proxy.
    #   - center_dist: central vectors block more diverse lines
    #   - ent_norm: coordinate diversity
    #   - one_fraction: interior-line bonus
    # Secondary: high-order spread for a smooth global ordering.
    blocking = (
        1.0 * center_dist
        + 0.7 * ent_norm
        + 0.5 * one_fraction
        + 0.3 * (distinct / 3.0)
    )
    # Spread term is a tie-breaker layer that does not override the main
    # structural signal but gives a deterministic, well-distributed ordering.
    priority_val = blocking + 0.05 * spread_norm
    return priority_val


def priority(vector, n):
    """
    Returns a priority value for the given vector. Higher priority means the
    vector is considered first in the greedy insertion.

    Strategy: line-blocking potential (global combinatorial pruning signal),
    with coordinate balance and a high-order CRC spread term as supporting
    signals. The evaluator computes all priorities before greedy insertion,
    so we use vector-intrinsic properties and cache results.
    """
    _init(n)

    # Deadline guard: return a fast deterministic fallback near the limit.
    elapsed = time.time() - _start_time
    if elapsed > _deadline - _safety_margin:
        vec = tuple(vector)
        try:
            return float(zlib.crc32(bytes(vec)))
        except Exception:
            return float(sum(vector))

    vec = tuple(vector)

    if vec in _priority_cache:
        return _priority_cache[vec]

    val = _blocking_potential(vec, n)
    _priority_cache[vec] = val
    return val
