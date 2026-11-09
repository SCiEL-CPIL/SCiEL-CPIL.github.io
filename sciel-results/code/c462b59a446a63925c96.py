"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

_N = 5040
_TARGET = 4979
_DEADLINE_SECONDS = 75.0
_START_TIME = time.time()
_DEADLINE = _START_TIME + _DEADLINE_SECONDS

# Precompute distinct positive divisors of 5040, sorted descending.
_DIVS = tuple(d for d in range(_N, 0, -1) if _N % d == 0)
_DIVS_SET = frozenset(_DIVS)


def _time_ok():
    return time.time() < _DEADLINE


@lru_cache(maxsize=None)
def _can_make(idx, remaining):
    """
    True if some subset of _DIVS[idx:] sums exactly to remaining.
    """
    if remaining == 0:
        return True
    if idx >= len(_DIVS) or remaining < 0:
        return False
    # If remaining is larger than the sum of all remaining divisors, impossible.
    # (Not necessary for speed here, but harmless.)
    d = _DIVS[idx]
    # Branch with divisor first (helps find larger terms earlier in reconstruction)
    if d <= remaining and _can_make(idx + 1, remaining - d):
        return True
    if _can_make(idx + 1, remaining):
        return True
    return False


def construct():
    """
    Return at most 256 distinct positive divisors of 5040 whose sum is 4979.
    Minimize the number of selected divisors.
    """
    global _DEADLINE, _START_TIME
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + _DEADLINE_SECONDS

    # Quick sanity: target must be representable; if not, return empty (invalid) to avoid exception.
    if not _can_make(0, _TARGET):
        return []

    # Reconstruct a minimum-cardinality solution.
    # Strategy: for k from 1 upward, try to find k distinct divisors summing to target.
    # Use a recursive search with pruning and time guard.
    n_divs = len(_DIVS)

    # Precompute suffix sums for pruning: max possible sum from idx onward.
    suffix_sum = [0] * (n_divs + 1)
    for i in range(n_divs - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + _DIVS[i]

    # Also precompute min possible sum from idx onward with exactly t items? Too heavy.
    # Instead, use simple bounds and memoization for (idx, remaining, count_left).

    memo = {}

    def dfs(idx, remaining, count_left, chosen):
        """
        Try to pick exactly count_left divisors from _DIVS[idx:] summing to remaining.
        Returns True on success, appends to chosen.
        """
        if not _time_ok():
            return False

        if count_left == 0:
            return remaining == 0

        if idx >= n_divs:
            return False

        # Pruning: not enough items left
        if n_divs - idx < count_left:
            return False

        # Pruning: max possible sum with count_left largest available
        # Take the count_left largest divisors from idx onward
        max_possible = sum(_DIVS[idx:idx + count_left])
        if remaining > max_possible:
            return False

        # Pruning: min possible sum with count_left smallest available
        # Take the count_left smallest divisors from idx onward (they are at the end)
        min_possible = sum(_DIVS[n_divs - count_left:n_divs])
        if remaining < min_possible:
            return False

        key = (idx, remaining, count_left)
        if key in memo:
            return memo[key]

        # Try including _DIVS[idx] if it fits
        d = _DIVS[idx]
        if d <= remaining:
            chosen.append(d)
            if dfs(idx + 1, remaining - d, count_left - 1, chosen):
                memo[key] = True
                return True
            chosen.pop()

        # Try excluding _DIVS[idx]
        if dfs(idx + 1, remaining, count_left, chosen):
            memo[key] = True
            return True

        memo[key] = False
        return False

    # Try increasing k from 1 to 256 (but realistically much smaller)
    for k in range(1, 257):
        if not _time_ok():
            break
        chosen = []
        if dfs(0, _TARGET, k, chosen):
            # Verify
            if len(chosen) == k and sum(chosen) == _TARGET and all(d in _DIVS_SET for d in chosen) and len(set(chosen)) == len(chosen):
                return chosen
            else:
                # Should not happen, but return empty if broken
                return []

    # Fallback: return empty (invalid) if no solution found within time
    return []

# EVOLVE-BLOCK-END
