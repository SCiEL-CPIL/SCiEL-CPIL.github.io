"""Rewritten program: minimize the LARGEST denominator in an Egyptian expansion of 5/31."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

_TARGET = Fraction(5, 31)
_START = time.time()
_LIMIT = 55.0  # wall-clock safety margin (evaluator kills at ~60s)

def _greedy_max(d):
    """Lower bound on the largest denominator needed to realize Fraction d.

    If d = a/b (a,b>0), then any Egyptian representation of d using denominators
    at least m requires 1/m <= d  =>  m <= 1/d = b/a.  Conversely a greedy step
    can always finish with denominator ceil(b/a).  So the largest denominator is
    at least floor(b/a) (and in practice the greedy completion achieves ~ceil(b/a)).
    """
    if d <= 0:
        return 0
    return (d.denominator + d.numerator - 1) // d.numerator  # ceil(b/a)

def _greedy_complete(r, D, used):
    """Complete r with denominators > D, distinct, all <= bound; return list or None.

    Returns (denoms, max_denom) or None.
    """
    out = []
    last = D
    while r > 0:
        # smallest allowed denominator >= ceil(1/r), and > last, and > any used
        a, b = r.numerator, r.denominator
        d = (b + a - 1) // a  # ceil(1/r)
        if d <= last:
            d = last + 1
        if d in used:
            d += 1
            while d in used:
                d += 1
        out.append(d)
        used.add(d)
        last = d
        r -= Fraction(1, d)
    return out

def construct():
    global _START
    best = None
    best_max = 10 ** 12

    # Precompute a strong initial upper bound via multi-start greedy.
    for start in range(2, 12):
        if start in (1,):
            continue
        r = _TARGET - Fraction(1, start)
        if r <= 0:
            continue
        used = {start}
        comp = _greedy_complete(r, start, used)
        if comp is not None:
            vals = [start] + comp
            mx = max(vals)
            if mx < best_max:
                best_max = mx
                best = vals

    # Branch and bound over increasing first denominators.
    # We keep denominators distinct and increasing.
    def search(remaining, last, depth, used, bound):
        nonlocal best, best_max
        if remaining <= 0:
            return
        if time.time() - _START > _LIMIT:
            return
        # Lower bound on the largest denominator from this branch:
        lb = _greedy_max(remaining)
        if lb >= bound:
            return  # cannot improve
        # Choose next denominator d > last, d >= ceil(1/remaining)
        a, b = remaining.numerator, remaining.denominator
        d = (b + a - 1) // a  # ceil(1/remaining)
        if d <= last:
            d = last + 1
        # We will try a handful of candidates around d (small d keeps max small)
        tried = 0
        while d < bound and tried < 6:
            if d not in used:
                nr = remaining - Fraction(1, d)
                if nr > 0:
                    # If adding this term, the remaining must be representable with
                    # denominators > d; the largest denominator is at least
                    # max(d, greedy_max(nr)).
                    sub_lb = _greedy_max(nr)
                    if sub_lb > d:
                        cand_bound = sub_lb
                    else:
                        cand_bound = d
                    if cand_bound < bound:
                        # Try to complete greedily first (fast) for an upper bound
                        nused = used | {d}
                        comp = _greedy_complete(nr, d, nused)
                        if comp is not None:
                            mx = max([d] + comp)
                            if mx < best_max:
                                best_max = mx
                                best = [d] + comp
                                bound = best_max
                        # Recurse one more level for refinement
                        search(nr, d, depth + 1, nused, bound)
                elif nr == 0:
                    mx = d
                    if mx < best_max:
                        best_max = mx
                        best = [d]
                        bound = best_max
            d += 1
            tried += 1

    # Initial bound from greedy multi-start
    bound = best_max if best else 10 ** 12

    # Search over first denominators from 2 upward, with a small window
    # to keep the largest denominator low.
    for start in range(2, 30):
        if time.time() - _START > _LIMIT:
            break
        if start >= bound:
            break
        r = _TARGET - Fraction(1, start)
        if r <= 0:
            continue
        used = {start}
        # Fast greedy completion to set bound
        comp = _greedy_complete(r, start, used)
        if comp is not None:
            mx = max([start] + comp)
            if mx < bound:
                bound = mx
                best_max = mx
                best = [start] + comp
        # Deeper search from this start
        search(r, start, 1, used, bound)

    if best is None:
        # Fallback: pure greedy from 2
        used = set()
        r = _TARGET
        vals = []
        last = 1
        while r > 0:
            a, b = r.numerator, r.denominator
            d = (b + a - 1) // a
            if d <= last:
                d = last + 1
            if d in used:
                d += 1
                while d in used:
                    d += 1
            vals.append(d)
            used.add(d)
            last = d
            r -= Fraction(1, d)
        best = vals

    return best
# EVOLVE-BLOCK-END
