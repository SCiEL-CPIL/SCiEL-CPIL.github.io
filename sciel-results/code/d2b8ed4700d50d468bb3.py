import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
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

    # Strategy: Try k=10 with various base configurations
    # For each base of size 7, search for 3 more terms
    # For each base of size 8, search for 2 more terms
    
    # Try multiple base configurations for k=10
    base_configs_7 = [
        [5, 6, 7, 8, 9, 10, 11],
        [5, 6, 7, 8, 9, 10, 12],
        [5, 6, 7, 8, 9, 11, 12],
        [5, 6, 7, 8, 10, 11, 12],
        [5, 6, 7, 9, 10, 11, 12],
        [5, 6, 8, 9, 10, 11, 12],
        [5, 7, 8, 9, 10, 11, 12],
        [6, 7, 8, 9, 10, 11, 12],
    ]
    
    base_configs_8 = [
        [5, 6, 7, 8, 9, 10, 11, 12],
        [5, 6, 7, 8, 9, 10, 11, 13],
        [5, 6, 7, 8, 9, 10, 12, 13],
        [5, 6, 7, 8, 9, 11, 12, 13],
        [5, 6, 7, 8, 10, 11, 12, 13],
        [5, 6, 7, 9, 10, 11, 12, 13],
        [5, 6, 8, 9, 10, 11, 12, 13],
        [5, 7, 8, 9, 10, 11, 12, 13],
        [6, 7, 8, 9, 10, 11, 12, 13],
    ]
    
    found = None
    
    # Try k=10 with 8-term bases first (faster: only 2 free variables)
    for base8 in base_configs_8:
        if time.time() > deadline:
            break
        sum8 = sum(Fraction(1, d) for d in base8)
        rem8 = 1 - sum8
        if rem8 <= 0:
            continue
        
        # Need 1/d9 + 1/d10 = rem8
        # d9 > max(base8)
        d9_min = max(max(base8) + 1, int(1 / float(rem8)) + 1)
        d9_max = int(2 / float(rem8))
        
        for d9 in range(d9_min, d9_max + 1):
            if time.time() > deadline:
                break
            rem9 = rem8 - Fraction(1, d9)
            if rem9 <= 0:
                continue
            d10 = 1 / float(rem9)
            d10_int = round(d10)
            if d10_int > d9 and d10_int <= 10000000:
                candidate = base8 + [d9, d10_int]
                if is_valid(candidate):
                    found = candidate
                    break
        if found:
            break
    
    if found:
        return found
    
    # Try k=10 with 7-term bases (3 free variables - slower)
    for base7 in base_configs_7:
        if time.time() > deadline:
            break
        sum7 = sum(Fraction(1, d) for d in base7)
        rem7 = 1 - sum7
        if rem7 <= 0:
            continue
        
        # Need 1/d8 + 1/d9 + 1/d10 = rem7
        # d8 > max(base7)
        d8_min = max(max(base7) + 1, int(1 / float(rem7)) + 1)
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
                    candidate = base7 + [d8, d9, d10_int]
                    if is_valid(candidate):
                        found = candidate
                        break
            if found:
                break
        if found:
            break
    
    if found:
        return found
    
    # Fallback: Pure Greedy
    R = Fraction(1, 1)
    d_prev = 4
    denoms = []
    while R > 0:
        if time.time() > deadline:
            break
        d_min = max(d_prev + 1, int(1 / float(R)) + 1)
        d = d_min
        denoms.append(d)
        R -= Fraction(1, d)
        d_prev = d
        
    if is_valid(denoms):
        return denoms
    
    raise RuntimeError("No valid solution found")
