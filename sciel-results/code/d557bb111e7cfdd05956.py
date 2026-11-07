"""Rewritten program: find witnesses in every consecutive-prime gap."""
# EVOLVE-BLOCK-START
import time
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061],
              [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091],
              [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111],
              [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151],
              [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177],
              [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223],
              [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259],
              [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve_small(n):
    """Return list of primes up to n using a simple sieve."""
    if n < 2:
        return []
    flag = bytearray(b'\x01') * (n + 1)
    flag[0] = 0
    flag[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if flag[i]:
            for j in range(i * i, n + 1, i):
                flag[j] = 0
    return [k for k in range(2, n + 1) if flag[k]]


def _is_rough(m, forbidden):
    """Check whether m has no prime factor in forbidden."""
    for p in forbidden:
        if m % p == 0:
            return False
    return True


def _all_rough_in_interval(lo, hi, forbidden):
    """Yield every integer m with lo < m < hi that is rough wrt forbidden."""
    for m in range(lo + 1, hi):
        if _is_rough(m, forbidden):
            yield m


def construct():
    t_start = time.perf_counter()
    limit = t_start + 52.0

    # Determine the maximum gap among the pinned pairs.
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    small_primes = _sieve_small(max_gap)

    # Group gaps by their size so we can cache forbidden-prime lists.
    forbidden_by_gap = {}
    for g in {hi - lo for lo, hi in P['gaps']}:
        forbidden_by_gap[g] = [p for p in small_primes if p < g]

    result = [None] * len(P['gaps'])

    # First pass: find at least one witness per gap where one exists.
    for idx, (lo, hi) in enumerate(P['gaps']):
        if time.perf_counter() > limit:
            break
        g = hi - lo
        forb = forbidden_by_gap[g]
        # Scan from both ends inward to increase chance of finding a rough number quickly.
        candidates = list(_all_rough_in_interval(lo, hi, forb))
        if candidates:
            # Prefer a middle candidate as a "balanced" witness; any works.
            result[idx] = int(candidates[len(candidates) // 2])

    # Second pass: verify and, if time permits, search for alternative witnesses
    # (not strictly needed for scoring, but keeps the function robust if gaps change).
    # The critical requirement is that each non-None entry is a valid witness.
    # Re-validate each chosen witness to be safe against any off-by-one.
    for idx, (lo, hi) in enumerate(P['gaps']):
        if time.perf_counter() > limit:
            break
        val = result[idx]
        if val is not None:
            g = hi - lo
            forb = forbidden_by_gap[g]
            if not (lo < val < hi) or not _is_rough(val, forb):
                # Fallback: re-scan
                found = None
                for m in range(lo + 1, hi):
                    if _is_rough(m, forb):
                        found = m
                        break
                result[idx] = int(found) if found is not None else None

    # Final cleanup: ensure exact Python int types.
    out = []
    for v in result:
        if v is None:
            out.append(None)
        else:
            out.append(int(v))
    return out
# EVOLVE-BLOCK-END
