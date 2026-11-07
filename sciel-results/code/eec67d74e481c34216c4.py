"""Pack distinct multiples of a fixed integer set - minimize the enclosing span."""
import time

# EVOLVE-BLOCK-START
_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_MAXV = 1000000
_DEADLINE = time.time() + 75  # safety margin vs 90 s wall-clock
_N = len(_DIVISORS)
_BEST = list(_DIVISORS)  # always-valid fallback: the divisors themselves (distinct, <=1e6)


def _try_window(L, U):
    """Try to assign distinct multiples of each divisor within [L, U].

    Returns (span, reps) on success or None.  Divisors are ordered by
    fewest candidate multiples first (most constrained first).
    """
    n = U - L + 1
    # candidate multiples for each divisor, within [L, U]
    cands = []
    for d in _DIVISORS:
        k = (L + d - 1) // d
        c = list(range(k * d, U + 1, d))
        cands.append(c)
    # order divisors by number of candidates (ascending)
    order = sorted(range(_N), key=lambda i: len(cands[i]))
    if any(len(cands[i]) == 0 for i in order):
        return None

    used = [False] * n
    reps = [0] * _N
    found = [None]

    def bt(pos):
        if pos == _N:
            found[0] = reps[:]
            return True
        i = order[pos]
        for m in cands[i]:
            idx = m - L
            if not used[idx]:
                used[idx] = True
                reps[i] = m
                if bt(pos + 1):
                    return True
                used[idx] = False
        return False

    if bt(0):
        r = found[0]
        return (U - L), r
    return None


def construct():
    global _BEST
    start = time.time()

    # 1) Search short windows first: most likely to hold the optimum.
    for span in range(1, 101):
        if time.time() > _DEADLINE:
            break
        for L in range(1, 100001):
            if time.time() > _DEADLINE:
                break
            res = _try_window(L, L + span)
            if res is not None:
                sp, reps = res
                # distinctness guaranteed by backtracking; verify bounds
                if all(1 <= x <= _MAXV for x in reps) and len(set(reps)) == _N:
                    if sp < max(_BEST) - min(_BEST):
                        _BEST = list(reps)
                break  # smallest span for this L found; move on

    # 2) If short-window search found nothing better, do a greedy lower-bound sweep.
    #    For a given L, the greedy (smallest multiple >= L) gives a span; we record
    #    the best valid greedy result as a fallback improvement.
    def _smallest_multiple_at_least(d, L):
        return ((L + d - 1) // d) * d

    best_greedy = None
    for L in range(1, 100001, 1):
        if time.time() > _DEADLINE:
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
            if best_greedy is None or sp < best_greedy[0]:
                best_greedy = (sp, list(reps))

    if best_greedy is not None and best_greedy[0] < max(_BEST) - min(_BEST):
        _BEST = best_greedy[1]

    # Final validation + type coercion to plain Python ints in a list.
    out = [int(x) for x in _BEST]
    assert len(out) == _N
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in out)
    assert len(set(out)) == _N
    assert all(1 <= x <= _MAXV for x in out)
    assert all(out[i] % _DIVISORS[i] == 0 for i in range(_N))
    return out
# EVOLVE-BLOCK-END
