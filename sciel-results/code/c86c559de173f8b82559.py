"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_N = len(_DIVISORS)
_MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // math.gcd(a, b)

_LCM = 1
for d in _DIVISORS:
    _LCM = _lcm(_LCM, d)

def _can_assign(L, R, divisors):
    """Check if we can assign distinct multiples in [L, L+R-1] for each divisor."""
    n = len(divisors)
    candidates = []
    for i in range(n):
        d = divisors[i]
        k_start = (L + d - 1) // d
        k_end = (L + R - 1) // d
        if k_start > k_end:
            return None
        cand = list(range(k_start * d, k_end * d + 1, d))
        if not cand:
            return None
        candidates.append(cand)
    
    order = sorted(range(n), key=lambda i: len(candidates[i]))
    assigned = [None] * n
    used = set()
    
    def backtrack(idx):
        if idx == n:
            return True
        i = order[idx]
        for val in candidates[i]:
            if val not in used:
                used.add(val)
                assigned[i] = val
                if backtrack(idx + 1):
                    return True
                used.remove(val)
                assigned[i] = None
        return False
    
    if backtrack(0):
        return assigned
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85
    divisors = _DIVISORS
    n = len(divisors)
    
    # Precompute for each R the set of feasible L values (mod LCM)
    # where every divisor has at least one multiple in [L, L+R-1]
    
    for R in range(1, 200):
        if time.time() > deadline:
            break
        
        # Build feasible L mask
        # For each divisor, compute which L in [0, LCM-1] have a multiple in [L, L+R-1]
        # Use a list of booleans
        all_feasible = [True] * _LCM
        
        for d in divisors:
            # For this divisor, mark L values where NO multiple is in [L, L+R-1]
            # A multiple m = k*d is in [L, L+R-1] iff m-R+1 <= L <= m
            # So good L values are union of [m-R+1, m] for all multiples m
            # Mark bad L values (complement)
            
            # Instead, mark good L values
            good = [False] * _LCM
            # Multiples of d that could be in a window starting in [0, LCM-1]
            # m must satisfy: there exists L in [0, LCM-1] with L <= m <= L+R-1
            # i.e., m-R+1 <= L <= m, and 0 <= L <= LCM-1
            # So m-R+1 <= LCM-1 and m >= 0
            # m ranges from max(0, 0) to LCM-1+R-1
            for k in range(0, _LCM // d + R // d + 2):
                m = k * d
                if m < 0:
                    continue
                lo = max(0, m - R + 1)
                hi = min(_LCM - 1, m)
                if lo <= hi:
                    for L in range(lo, hi + 1):
                        good[L] = True
            
            for L in range(_LCM):
                if all_feasible[L] and not good[L]:
                    all_feasible[L] = False
        
        # Check feasible L values with backtracking
        for L in range(1, _LCM):
            if time.time() > deadline:
                break
            if not all_feasible[L]:
                continue
            
            assigned = _can_assign(L, R, divisors)
            if assigned is not None:
                if max(assigned) <= _MAX_VAL:
                    return [int(assigned[i]) for i in range(n)]
    
    # Fallback: return the divisors themselves
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
