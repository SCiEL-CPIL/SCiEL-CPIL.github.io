"""Pack distinct multiples of a fixed integer set - minimize the enclosing span."""
import time

# EVOLVE-BLOCK-START
_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_MAXV = 1000000
_DEADLINE = time.time() + 85  # safety margin vs 90 s wall-clock
_N = len(_DIVISORS)
_BEST = list(_DIVISORS)  # always-valid fallback: the divisors themselves (distinct, <=1e6)
_BEST_SPAN = max(_BEST) - min(_BEST)


def _try_window(L, U):
    """Try to assign distinct multiples of each divisor within [L, U].

    Returns (span, reps) on success or None.
    """
    n = U - L + 1
    if n < _N:
        return None
        
    cands = []
    for d in _DIVISORS:
        k = (L + d - 1) // d
        c = [m - L for m in range(k * d, U + 1, d)] # store indices
        if not c:
            return None
        cands.append(c)
        
    # order divisors by number of candidates (ascending)
    order = sorted(range(_N), key=lambda i: len(cands[i]))
    
    used_mask = 0
    reps = [0] * _N
    
    def bt(pos):
        nonlocal used_mask
        if pos == _N:
            return True
        i = order[pos]
        for idx in cands[i]:
            if not (used_mask & (1 << idx)):
                used_mask |= (1 << idx)
                reps[i] = L + idx
                if bt(pos + 1):
                    return True
                used_mask ^= (1 << idx)
        return False

    if bt(0):
        return (U - L), reps
    return None


def construct():
    global _BEST, _BEST_SPAN
    start = time.time()

    # 1) Search short windows first.
    # We search spans from 1 up to current best span - 1.
    # We restrict L to a reasonable range to avoid timeout.
    # If the optimal solution is at a large L, we might miss it, but the periodicity
    # and small divisors suggest small L is likely to contain a solution.
    # We will expand the L range if time permits.
    
    l_limit = 5000
    
    for span in range(1, _BEST_SPAN):
        if time.time() > _DEADLINE:
            break
            
        # If span is small, the window is small.
        # Check L in 1..l_limit
        found = False
        for L in range(1, l_limit + 1):
            if time.time() > _DEADLINE:
                break
            res = _try_window(L, L + span)
            if res is not None:
                sp, reps = res
                if all(1 <= x <= _MAXV for x in reps) and len(set(reps)) == _N:
                    if sp < _BEST_SPAN:
                        _BEST = list(reps)
                        _BEST_SPAN = sp
                found = True
                break # Found a solution for this span, no need to check other L for this span
        
        # If we found a solution for this span, we can break?
        # No, we want the MINIMUM span. Since we iterate span from 1 upwards,
        # the first span for which we find ANY valid L is the global minimum.
        if found:
            break

    # 2) If we didn't find anything better in the restricted range,
    #    do a broader search with a time limit, or use the greedy fallback.
    #    The current _BEST_SPAN is still the initial one if nothing was found.
    
    if _BEST_SPAN == max(_DIVISORS) - min(_DIVISORS):
        # Try a broader L range for small spans, but with strict time limit
        for span in range(1, _BEST_SPAN):
            if time.time() > _DEADLINE - 10: # leave 10s for fallback
                break
            for L in range(1, 50000, 10): # step 10 to speed up
                if time.time() > _DEADLINE - 10:
                    break
                res = _try_window(L, L + span)
                if res is not None:
                    sp, reps = res
                    if all(1 <= x <= _MAXV for x in reps) and len(set(reps)) == _N:
                        if sp < _BEST_SPAN:
                            _BEST = list(reps)
                            _BEST_SPAN = sp
                    break

    # 3) Greedy fallback if still not improved
    def _smallest_multiple_at_least(d, L):
        return ((L + d - 1) // d) * d

    best_greedy_span = _BEST_SPAN
    best_greedy_reps = _BEST
    
    # Try a few specific L values for greedy
    for L in range(1, 1000):
        if time.time() > _DEADLINE - 5:
            break
        reps = []
        ok = True
        for d in _DIVISORS:
            m = _smallest_multiple_at_least(d, L)
            if m > _MAXV:
                ok = False
                break
            reps.append(m)
        if not ok:
            break
        if len(set(reps)) == _N:
            sp = max(reps) - min(reps)
            if sp < best_greedy_span:
                best_greedy_span = sp
                best_greedy_reps = list(reps)

    if best_greedy_span < _BEST_SPAN:
        _BEST = best_greedy_reps
        _BEST_SPAN = best_greedy_span

    # Final validation + type coercion
    out = [int(x) for x in _BEST]
    assert len(out) == _N
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in out)
    assert len(set(out)) == _N
    assert all(1 <= x <= _MAXV for x in out)
    assert all(out[i] % _DIVISORS[i] == 0 for i in range(_N))
    return out
# EVOLVE-BLOCK-END
