import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # Target: Minimize count.
    # We established k >= 10.
    # Try to find a solution with k=10.
    
    # Helper to check if a list of denominators sums to 1
    def is_valid(denoms):
        if len(denoms) < 1 or len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        if any(d < 5 or d > 10000000 for d in denoms):
            return False
        s = Fraction(0, 1)
        for d in denoms:
            s += Fraction(1, d)
        return s == 1

    # Strategy 1: Greedy with small search for tail
    # Fix first few terms to be small to maximize sum, then solve for the rest.
    
    # Try k=10
    # Base: 5, 6, 7, 8, 9, 10, 11, 12
    base8 = [5, 6, 7, 8, 9, 10, 11, 12]
    sum8 = sum(Fraction(1, d) for d in base8)
    rem8 = 1 - sum8
    
    # Need 1/d9 + 1/d10 = rem8
    # d9 > 12
    # 1/d9 < rem8 => d9 > 1/rem8
    # 1/d9 > rem8/2 => d9 < 2/rem8
    d9_min = max(13, int(1 / float(rem8)) + 1)
    d9_max = int(2 / float(rem8))
    
    found = None
    for d9 in range(d9_min, d9_max + 1):
        if time.time() > deadline:
            break
        rem9 = rem8 - Fraction(1, d9)
        if rem9 <= 0:
            continue
        d10 = 1 / float(rem9)
        # Check if d10 is close to integer
        d10_int = round(d10)
        if d10_int > d9 and d10_int <= 10000000:
            # Verify exactly
            if is_valid(base8 + [d9, d10_int]):
                found = base8 + [d9, d10_int]
                break
    
    if found:
        return found

    # If k=10 with fixed 8 terms failed, try k=10 with fixed 7 terms
    base7 = [5, 6, 7, 8, 9, 10, 11]
    sum7 = sum(Fraction(1, d) for d in base7)
    rem7 = 1 - sum7
    
    # Need 1/d8 + 1/d9 + 1/d10 = rem7
    # d8 > 11
    # 1/d8 < rem7 => d8 > 1/rem7
    # 1/d8 > rem7/3 => d8 < 3/rem7
    d8_min = max(12, int(1 / float(rem7)) + 1)
    d8_max = int(3 / float(rem7))
    
    for d8 in range(d8_min, d8_max + 1):
        if time.time() > deadline:
            break
        rem8_b = rem7 - Fraction(1, d8)
        if rem8_b <= 0:
            continue
        # Need 1/d9 + 1/d10 = rem8_b
        # d9 > d8
        d9_min_b = max(d8 + 1, int(1 / float(rem8_b)) + 1)
        d9_max_b = int(2 / float(rem8_b))
        
        for d9 in range(d9_min_b, d9_max_b + 1):
            if time.time() > deadline:
                break
            rem9_b = rem8_b - Fraction(1, d9)
            if rem9_b <= 0:
                continue
            d10 = 1 / float(rem9_b)
            d10_int = round(d10)
            if d10_int > d9 and d10_int <= 10000000:
                if is_valid(base7 + [d8, d9, d10_int]):
                    found = base7 + [d8, d9, d10_int]
                    break
        if found:
            break
            
    if found:
        return found

    # Fallback: Pure Greedy
    # This will find a solution, likely with more terms, but valid.
    R = Fraction(1, 1)
    d_prev = 4
    denoms = []
    while R > 0:
        if time.time() > deadline:
            break
        # d >= 1/R
        d_min = max(d_prev + 1, int(1 / float(R)) + 1)
        d = d_min
        denoms.append(d)
        R -= Fraction(1, d)
        d_prev = d
        
    if is_valid(denoms):
        return denoms

    # Last resort: return a known valid solution if found in search history or simple construction
    # If greedy failed due to time or error, we might have a partial list.
    # But greedy always terminates.
    
    raise RuntimeError("No valid solution found")
