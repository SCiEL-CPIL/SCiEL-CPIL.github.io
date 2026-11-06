import time
from fractions import Fraction

def construct():
    deadline = time.time() + 80
    best = None
    
    for min_count in range(8, 16):
        if time.time() > deadline:
            break
        result = _search(Fraction(1, 1), min_count, 5, [], deadline)
        if result is not None:
            return result
        best = result
    
    if best is not None:
        return best
    raise RuntimeError("no solution found")

def _search(remaining, terms_left, min_denom, current, deadline):
    if time.time() > deadline:
        return None
    
    if terms_left == 0:
        return current if remaining == 0 else None
    
    if remaining <= 0:
        return None
    
    # Prune: max possible sum
    max_sum = Fraction(0, 1)
    for i in range(terms_left):
        max_sum += Fraction(1, min_denom + i)
    if remaining > max_sum:
        return None
    
    # d range
    d_min_needed = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
    d_max_allowed = (terms_left * remaining.numerator) // remaining.denominator
    
    d_start = max(min_denom, d_min_needed)
    d_end = d_max_allowed
    
    if d_start > d_end:
        return None
    
    for d in range(d_start, d_end + 1):
        if time.time() > deadline:
            return None
        new_rem = remaining - Fraction(1, d)
        if new_rem < 0:
            continue
        if terms_left == 1:
            if new_rem == 0:
                return current + [d]
        else:
            result = _search(new_rem, terms_left - 1, d + 1, current + [d], deadline)
            if result is not None:
                return result
    
    return None
