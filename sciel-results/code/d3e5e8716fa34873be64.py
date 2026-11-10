"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time

def construct():
    target = Fraction(1, 1)
    min_den = 5
    max_den = 10000000
    start_time = time.time()
    
    for count in range(1, 257):
        if time.time() - start_time > 55:
            break
        result = _search(target, min_den, count, max_den)
        if result is not None:
            return result
    raise RuntimeError("no representation found")

def _search(remaining, min_d, terms_left, max_d):
    if terms_left == 0:
        return [] if remaining == 0 else None
    
    if remaining <= 0:
        return None
    
    # Max sum bound
    if remaining > Fraction(terms_left, min_d):
        return None
    
    # Min sum bound (for small terms_left)
    if terms_left <= 30:
        min_sum = Fraction(0, 1)
        for i in range(terms_left):
            min_sum += Fraction(1, min_d + i)
        if remaining < min_sum:
            return None
    
    if terms_left == 1:
        if remaining.numerator != 1:
            return None
        d = remaining.denominator
        if d < min_d or d > max_d:
            return None
        return [d]
    
    # Determine d range
    # d >= ceil(1/remaining)
    d_min = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
    
    # Generous upper bound
    # d should not be much larger than 1/remaining
    d_max = min(max_d, int(1.0 / float(remaining)) * (terms_left + 5) + 1000)
    
    for d in range(d_min, d_max + 1):
        if d > max_d:
            break
        rem = remaining - Fraction(1, d)
        if rem < 0:
            break
        if rem == 0:
            if terms_left == 1:
                return [d]
            continue
        
        # Prune: rem <= (terms_left-1)/(d+1)
        if rem > Fraction(terms_left - 1, d + 1):
            break
        
        # Prune: rem >= min possible sum of (terms_left-1) terms > d
        if terms_left - 1 <= 20:
            min_rem = Fraction(0, 1)
            for i in range(terms_left - 1):
                min_rem += Fraction(1, d + 1 + i)
            if rem < min_rem:
                continue
        
        sub = _search(rem, d + 1, terms_left - 1, max_d)
        if sub is not None:
            return [d] + sub
    
    return None

# EVOLVE-BLOCK-END
