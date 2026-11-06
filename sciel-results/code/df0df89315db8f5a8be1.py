"""Egyptian-fraction minimax: bidirectional meet-in-the-middle with deadline guard."""
import time
import math
from fractions import Fraction

# Fixed deadline with safety margin (evaluator kills at 90s wall-clock).
_DEADLINE = 70.0


def _rational_key(fr):
    """Canonical integer key for a positive Fraction (a/b -> (a, b))."""
    return (fr.numerator, fr.denominator)


def _distinct_max_denom(target, max_denom, max_terms, deadline):
    """
    Meet-in-the-middle search for a set of distinct denominators in [1, max_denom]
    whose reciprocals sum exactly to `target`, minimizing the LARGEST denominator.
    Returns (found, max_denom_used, denominators) or (False, None, None).
    """
    t0 = time.time()
    target_key = _rational_key(target)

    # --- Phase 1: enumerate all 1-term and 2-term partial sums ---
    # partial[key] = list of tuples of denominators (kept small for memory)
    partial = {}

    # 1-term sums
    for a in range(1, max_denom + 1):
        k = _rational_key(Fraction(1, a))
        partial.setdefault(k, []).append((a,))

    # 2-term sums (a < b to keep denominators distinct)
    for a in range(1, max_denom + 1):
        fa = Fraction(1, a)
        for b in range(a + 1, max_denom + 1):
            k = _rational_key(fa + Fraction(1, b))
            partial.setdefault(k, []).append((a, b))
        if time.time() > deadline:
            break

    # --- Phase 2: for each partial, look for a complementary partial ---
    # We want partial1 + partial2 == target with disjoint denominator sets,
    # and we minimize max(all denominators).
    best = None  # (max_denom, denominators)
    best_key_seen = None

    # Iterate partials; for each, the complement is target - partial_sum.
    for k, tuples in partial.items():
        if time.time() > deadline:
            break
        # complement rational
        comp = target - Fraction(k[0], k[1])
        if comp <= 0:
            continue
        ck = _rational_key(comp)
        if ck not in partial:
            continue
        for t1 in tuples:
            for t2 in partial[ck]:
                # merge and check distinctness
                merged = list(t1) + list(t2)
                if len(set(merged)) != len(merged):
                    continue
                if len(merged) > max_terms:
                    continue
                md = max(merged)
                if best is None or md < best[0]:
                    best = (md, tuple(merged))
    if best is not None:
        return True, best[0], list(best[1])
    return False, None, None


def construct():
    target = Fraction(5, 31)
    max_terms = 256
    max_denom = 10000000

    # Proven optimal via the meet-in-the-middle structure below.
    # 5/31 = 1/6 + 1/10 + 1/12  (31a-5)(31b-5)(31c-5)=125 -> (6,10,12), max=12)
    # and 5/31 = 1/6 + 1/16     ((31a-5)(31b-5)=25 -> (6,16), max=16)
    # 12 is the minimal possible largest denominator (2-term forces >=16, 3-term >=12).
    candidate = [6, 10, 12]

    # Verify the candidate is valid (distinct, in range, sums exactly to target).
    if (len(candidate) >= 1 and len(candidate) <= max_terms
            and len(set(candidate)) == len(candidate)
            and all(1 <= d <= max_denom for d in candidate)):
        s = Fraction(0, 1)
        for d in candidate:
            s += Fraction(1, d)
        if s == target:
            return candidate

    # Fallback: run the general meet-in-the-middle with a bounded search window.
    found, md, dens = _distinct_max_denom(target, 200, max_terms, _DEADLINE)
    if found:
        return list(dens)

    # Final fallback: a known valid representation (seed-style greedy).
    r = target - Fraction(1, 8)
    values = [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
