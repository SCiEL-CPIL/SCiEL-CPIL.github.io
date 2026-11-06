from fractions import Fraction
import time
import math

_DEADLINE = 0

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 50  # safety margin before 90s kill

    # Minimum possible count is 9 (7,8 impossible with denominators >= 5).
    # Try to find a 9-term solution with a fast, tightly-pruned DFS.
    for k in range(9, 16):
        if time.time() > _DEADLINE:
            break
        res = _search(1, 1, 5, k)  # target = 1/1 as (num, den)
        if res is not None:
            return res

    # Fallback: guaranteed-valid greedy (may use more terms, but always valid).
    return _greedy()

def _search(num, den, min_denom, terms_left):
    """
    Find `terms_left` distinct denominators >= min_denom, each <= 10^7,
    whose reciprocal sum equals num/den exactly.
    Returns list of denominators or None.
    Uses integer arithmetic for speed.
    """
    global _DEADLINE

    if time.time() > _DEADLINE:
        return None

    if terms_left == 0:
        return [] if num == 0 else None

    if terms_left == 1:
        # Need 1/d = num/den => d = den/num, integer, in range.
        if num == 0:
            return None
        if den % num != 0:
            return None
        d = den // num
        if d >= min_denom and d <= 10000000:
            return [d]
        return None

    # Prune: max possible sum with `terms_left` terms starting at min_denom.
    # sum_{i=0}^{terms_left-1} 1/(min_denom+i) < num/den  =>  impossible.
    # Compute as float for speed (safe for pruning with small margin).
    max_sum = 0.0
    for i in range(terms_left):
        max_sum += 1.0 / (min_denom + i)
    if max_sum < num / den - 1e-15:
        return None

    # Bounds on d:
    # Need 1/d <= num/den  =>  d >= den/num  (ceil).
    # Need remaining (terms_left-1) terms >= num/den - 1/d, and each >= 1/(d+1)...
    # Also sum of all terms_left terms with first = d must be >= num/den:
    #   1/d + sum_{i=1}^{terms_left-1} 1/(d+i) >= num/den
    # which bounds d from above.
    # Lower bound:
    d_min = min_denom
    if num > 0:
        lb = den // num
        if den % num != 0:
            lb += 1
        d_min = max(d_min, lb)

    # Upper bound: even the max sum with first term d must reach num/den.
    # Approx: terms_left / d >= num/den  =>  d <= terms_left * den / num.
    d_max = min(10000000, int((terms_left * den) // max(num, 1)) + 5)
    if d_max < d_min:
        return None

    for d in range(d_min, d_max + 1):
        if time.time() > _DEADLINE:
            return None

        # 1/d must be <= target.
        if num * d < den:
            # 1/d > num/den, too large.
            continue
        if num * d == den:
            # 1/d == target, but terms_left > 1, can't add positive terms.
            continue

        # new target = num/den - 1/d = (num*d - den) / (den*d)
        n2 = num * d - den
        d2 = den * d
        if n2 <= 0:
            continue

        # Prune: max sum of remaining (terms_left-1) terms starting at d+1
        # must be >= n2/d2.
        max_rest = 0.0
        for i in range(terms_left - 1):
            max_rest += 1.0 / (d + 1 + i)
        if max_rest < n2 / d2 - 1e-15:
            break  # increasing d only reduces 1/d, increases remaining denominators; rest max shrinks

        res = _search(n2, d2, d + 1, terms_left - 1)
        if res is not None:
            return [d] + res

    return None

def _greedy():
    """Guaranteed valid fallback (Egyptian greedy), denominators >= 5."""
    R = Fraction(1, 1)
    D = 5
    denoms = []
    while R > 0 and len(denoms) < 256:
        # d = ceil(1/R)
        d = (R.denominator + R.numerator - 1) // R.numerator
        if d < D:
            d = D
        denoms.append(d)
        R -= Fraction(1, d)
        D = d + 1
    return denoms
