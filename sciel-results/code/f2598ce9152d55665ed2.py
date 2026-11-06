"""Combined program: DFS for minimal terms + DP fallback."""
import time
import math
from fractions import Fraction

_DEADLINE = 85.0
_T0 = time.time()

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Precompute sums of 1 to 3 unit fractions with denominators in [5, 100]
# Map (num, den) to list of denominator lists
_TAIL_TABLE = {}

def _build_tail_table(max_d=100):
    """
    Precompute all sums of 1 to 3 distinct unit fractions with denominators in [5, max_d].
    """
    def gen(den_min, remaining_terms, current_sum_num, current_sum_den, current_dens):
        if not _deadline_ok():
            return
        if remaining_terms == 0:
            if current_sum_num == 0:
                return
            g = _gcd(current_sum_num, current_sum_den)
            n, d = current_sum_num // g, current_sum_den // g
            key = (n, d)
            if key not in _TAIL_TABLE:
                _TAIL_TABLE[key] = []
            _TAIL_TABLE[key].append(list(current_dens))
            return
        
        for d in range(den_min, max_d + 1):
            if not _deadline_ok():
                return
            new_num = current_sum_num * d + current_sum_den
            new_den = current_sum_den * d
            
            current_dens.append(d)
            gen(d + 1, remaining_terms - 1, new_num, new_den, current_dens)
            current_dens.pop()

    for t in range(1, 4):
        gen(5, t, 0, 1, [])

_build_tail_table(100)

def _search(frac_num, frac_den, max_terms, min_den, max_den, out):
    """
    Recursively search for an Egyptian fraction representation.
    """
    if not _deadline_ok():
        return False
    
    g = _gcd(frac_num, frac_den)
    frac_num //= g
    frac_den //= g

    if max_terms == 0:
        return frac_num == 0

    if max_terms == 1:
        if frac_den % frac_num == 0:
            d = frac_den // frac_num
            if min_den <= d <= max_den:
                out.append(d)
                return True
        return False

    # Check tail table for max_terms <= 3
    if max_terms <= 3:
        key = (frac_num, frac_den)
        if key in _TAIL_TABLE:
            for cands in _TAIL_TABLE[key]:
                if len(cands) == max_terms:
                    if min(cands) > min_den and max(cands) <= max_den:
                        out.extend(cands)
                        return True
        # If table fails, we can still try standard DFS for 2 or 3 terms?
        # Standard DFS for 2 terms is fast.
        # Standard DFS for 3 terms is moderate.
        # If table fails, it means no solution with small denominators.
        # We should try standard DFS to be safe?
        # But standard DFS for 3 terms might be slow if min_den is large.
        # Let's try standard DFS if table fails.
        pass

    if max_terms == 2:
        d1_min = max(min_den, frac_den // frac_num + 1)
        for d1 in range(d1_min, max_den + 1):
            if not _deadline_ok():
                return False
            rem_num = frac_num * d1 - frac_den
            if rem_num <= 0:
                continue
            rem_den = frac_den * d1
            if rem_den % rem_num != 0:
                continue
            d2 = rem_den // rem_num
            if d2 > d1 and d2 <= max_den:
                out.append(d1)
                out.append(d2)
                return True
        return False

    # Determine range for next denominator d
    d_min = max(min_den, frac_den // frac_num + 1)
    
    for d in range(d_min, max_den + 1):
        if not _deadline_ok():
            return False
        
        if frac_num * d >= frac_den:
            continue
            
        rem_num = frac_num * d - frac_den
        rem_den = frac_den * d
        
        if rem_num * (d + 1) > rem_den * (max_terms - 1):
            break
            
        out.append(d)
        if _search(rem_num, rem_den, max_terms - 1, d + 1, max_den, out):
            return True
        out.pop()
        
    return False

def _validate(dens):
    if not (1 <= len(dens) <= 256):
        return False
    if len(set(dens)) != len(dens):
        return False
    for d in dens:
        if d < 5 or d > 10000000:
            return False
    from fractions import Fraction
    s = Fraction(0, 1)
    for d in dens:
        s += Fraction(1, d)
    return s == 1

def _dp_fallback():
    target = 2520
    best_count = {0: 0}
    best_path = {0: []}
    divs = [d for d in range(5, target + 1) if target % d == 0]
    for d in divs:
        w = target // d
        for s in list(best_count.keys()):
            new_s = s + w
            if new_s <= target:
                new_count = best_count[s] + 1
                if new_s not in best_count or new_count < best_count[new_s]:
                    best_count[new_s] = new_count
                    best_path[new_s] = best_path[s] + [d]
    if target in best_count:
        return best_path[target]
    return None

def construct():
    for k in range(5, 12):
        if not _deadline_ok():
            break
        out = []
        if _search(1, 1, k, 5, 10000000, out):
            if _validate(out):
                return out
    if _deadline_ok():
        sol = _dp_fallback()
        if sol and _validate(sol):
            return sol
    raise RuntimeError("no representation found")
