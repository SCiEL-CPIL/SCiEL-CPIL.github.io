"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill
    
    n = len(DIVISORS)
    
    # Compute LCM of all divisors - the pattern of multiples repeats every LCM
    lcm_val = 1
    for d in DIVISORS:
        lcm_val = _lcm(lcm_val, d)
    
    # The optimal window will have its minimum at some L.
    # For a given L, the natural assignment is: for each divisor d_i,
    # x_i = smallest multiple of d_i that is >= L.
    # If collisions occur, bump one up.
    # 
    # Key insight: the "greedy" assignment (smallest multiple >= L for each)
    # only changes when L crosses a multiple of some divisor.
    # Within an interval between consecutive multiples of any divisor,
    # the greedy assignment is constant.
    #
    # So we only need to check L values that are multiples of at least one divisor,
    # or L values just after such multiples.
    #
    # Even better: since the pattern repeats every LCM, we can search one period
    # and then the best solution repeats. But LCM might be large.
    # Let's check: LCM of [6,10,14,15,21,22,26,33,35,39]
    # 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # LCM = 2*3*5*7*11*13 = 30030
    # That's manageable! We can scan all L from 1 to LCM in the worst case,
    # but we only need to check L values where the greedy assignment changes.
    
    # Generate all "critical" L values: multiples of each divisor up to LCM
    # These are the points where the greedy assignment changes.
    critical = set()
    for d in DIVISORS:
        for m in range(d, lcm_val + 1, d):
            critical.add(m)
    # Also include 1 as a starting point
    critical.add(1)
    critical = sorted(critical)
    
    best_range = MAX_VAL
    best_set = None
    
    # For each critical L, compute the greedy assignment with collision resolution
    for L in critical:
        if time.time() > deadline:
            break
        
        # Compute smallest multiple of each divisor >= L
        vals = []
        for i, d in enumerate(DIVISORS):
            # ceil(L/d) * d
            q = (L + d - 1) // d
            vals.append(q * d)
        
        # Resolve collisions: if two values are equal, bump the later one up
        # Use a greedy approach: process in order, if collision, bump up
        seen = set()
        resolved = []
        valid = True
        for i in range(n):
            v = vals[i]
            while v in seen:
                v += DIVISORS[i]
                if v > MAX_VAL:
                    valid = False
                    break
            if not valid:
                break
            seen.add(v)
            resolved.append(v)
        
        if not valid:
            continue
        
        r = max(resolved) - min(resolved)
        
        if r < best_range:
            best_range = r
            best_set = list(resolved)
            
            # Minimum possible range for 10 distinct positive integers is 9
            if best_range <= 9:
                break
    
    # Also try searching with L values that are NOT critical points but are
    # just before a critical point (i.e., L = critical - 1 for each critical)
    # This can sometimes give better results because the "floor" behavior differs.
    if best_range > 9 and time.time() < deadline:
        critical_minus = set()
        for c in critical:
            if c > 1:
                critical_minus.add(c - 1)
        critical_minus = sorted(critical_minus)
        
        for L in critical_minus:
            if time.time() > deadline:
                break
            
            vals = []
            for i, d in enumerate(DIVISORS):
                q = (L + d - 1) // d
                vals.append(q * d)
            
            seen = set()
            resolved = []
            valid = True
            for i in range(n):
                v = vals[i]
                while v in seen:
                    v += DIVISORS[i]
                    if v > MAX_VAL:
                        valid = False
                        break
                if not valid:
                    break
                seen.add(v)
                resolved.append(v)
            
            if not valid:
                continue
            
            r = max(resolved) - min(resolved)
            
            if r < best_range:
                best_range = r
                best_set = list(resolved)
                
                if best_range <= 9:
                    break
    
    if best_set is None:
        # Fallback: use the divisors themselves
        best_set = [int(d) for d in DIVISORS]
    
    # Ensure all values are Python int (not numpy int)
    return [int(v) for v in best_set]
# EVOLVE-BLOCK-END
