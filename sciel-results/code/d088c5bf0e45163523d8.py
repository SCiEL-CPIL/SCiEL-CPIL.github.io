"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _greedy_complete(remainder):
    """Complete a rational remainder using the greedy Egyptian fraction algorithm.
    Returns list of denominators or None if it fails."""
    if remainder == 0:
        return []
    if remainder < 0:
        return None
    values = []
    r = remainder
    while r != 0:
        # d = ceil(1/r)
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d <= 0:
            return None
        values.append(d)
        r -= Fraction(1, d)
        # Safety: if r becomes negative or we have too many terms
        if r < 0:
            return None
        if len(values) > 200:
            return None
    return values

def _try_completion(first_denoms, target):
    """Given a list of first denominators, check validity and complete greedily.
    Returns (max_denominator, full_list) or (None, None) if invalid."""
    used = set(first_denoms)
    # Check all distinct and positive
    if len(used) != len(first_denoms):
        return None, None
    if any(d <= 0 for d in first_denoms):
        return None, None
    remainder = target
    for d in first_denoms:
        remainder -= Fraction(1, d)
    if remainder < 0:
        return None, None
    if remainder == 0:
        return max(first_denoms), list(first_denoms)
    completion = _greedy_complete(remainder)
    if completion is None:
        return None, None
    # Check for duplicates
    all_denoms = first_denoms + completion
    if len(set(all_denoms)) != len(all_denoms):
        return None, None
    return max(all_denoms), all_denoms

def construct():
    target = Fraction(5, 31)
    deadline = time.time() + 55  # Safety margin before 60s kill
    
    best_max = None
    best_list = None
    
    def update_candidate(denoms):
        nonlocal best_max, best_list
        if denoms is None or len(denoms) == 0 or len(denoms) > 256:
            return
        mx = max(denoms)
        if best_max is None or mx < best_max:
            best_max = mx
            best_list = denoms
    
    # Try single denominators from 2 up to a reasonable bound
    # 1/d < 5/31 means d > 31/5 = 6.2, so d >= 7
    # Also 1/d should not be too small to keep max reasonable
    # Let's try d from 7 to about 200 for single starts
    for d in range(7, 300):
        if time.time() > deadline:
            break
        rem = target - Fraction(1, d)
        if rem < 0:
            continue
        if rem == 0:
            update_candidate([d])
            continue
        comp = _greedy_complete(rem)
        if comp is not None:
            all_denoms = [d] + comp
            if len(set(all_denoms)) == len(all_denoms) and len(all_denoms) <= 256:
                update_candidate(all_denoms)
    
    # Try pairs of denominators
    # First denom from 7 to ~100, second from first+1 to ~200
    # This is more expensive, so be selective
    for d1 in range(7, 80):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, d1)
        if rem1 < 0:
            continue
        # Second denominator d2 > d1, and 1/d2 < rem1
        # d2 > 1/rem1
        if rem1 > 0:
            d2_min = d1 + 1
            # Upper bound: we want 1/d2 < rem1, so d2 > 1/rem1
            # But we also don't want d2 too large
            d2_max = min(500, int(1 / rem1.numerator * rem1.denominator) + 10 if rem1.numerator > 0 else 500)
            d2_max = max(d2_max, d2_min)
            for d2 in range(d2_min, d2_max + 1):
                if time.time() > deadline:
                    break
                rem2 = rem1 - Fraction(1, d2)
                if rem2 < 0:
                    continue
                if rem2 == 0:
                    update_candidate([d1, d2])
                    continue
                comp = _greedy_complete(rem2)
                if comp is not None:
                    all_denoms = [d1, d2] + comp
                    if len(set(all_denoms)) == len(all_denoms) and len(all_denoms) <= 256:
                        update_candidate(all_denoms)
    
    if best_list is not None:
        return best_list
    
    # Fallback: use the original greedy approach
    r = target
    values = []
    while r != 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
