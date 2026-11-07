import time
import math
from itertools import permutations

def construct():
    """
    Finds 10 distinct positive integers <= 1000000, each divisible by its corresponding divisor in [6,10,14,15,21,22,26,33,35,39],
    such that max(reps) - min(reps) is minimized.
    """
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    deadline = time.time() + 80  # Safety margin under 90s limit
    
    # Compute LCM of all divisors
    lcm = 1
    for d in divisors:
        lcm = lcm * d // math.gcd(lcm, d)
    # lcm = 3270270 (let me verify: 6*10*14*15*21*22*26*33*35*39 / gcds)
    # Actually let's just compute it
    
    best_span = float('inf')
    best_result = None
    
    # Strategy: The relative positions of multiples repeat every LCM.
    # For a given "phase" (offset within one LCM period), the span is fixed.
    # So we only need to search L in [1, LCM] and the best span will repeat.
    # But LCM might be large. Let's check.
    
    # Alternative: search L from 1 to some bound, but be smarter about it.
    # Key insight: if we find a solution with span S at position L, then L + k*LCM also works with same span.
    # So the minimum span is achieved in the first period [1, LCM].
    
    # Let's compute LCM properly
    from functools import reduce
    lcm_val = reduce(lambda a, b: a * b // math.gcd(a, b), divisors)
    
    # Search L in [1, min(lcm_val, max_val)]
    max_L_search = min(lcm_val, max_val)
    
    # Generate candidate Ls: multiples of any divisor
    candidate_Ls = set()
    for d in divisors:
        candidate_Ls.update(range(d, max_L_search + 1, d))
    
    sorted_Ls = sorted(candidate_Ls)
    
    # Pre-sort indices by divisor for two orders
    indices_desc = sorted(range(n), key=lambda i: divisors[i], reverse=True)
    indices_asc = sorted(range(n), key=lambda i: divisors[i])
    
    for L in sorted_Ls:
        if time.time() > deadline:
            break
        
        # Determine which divisors can be the one corresponding to L
        for k, d_k in enumerate(divisors):
            if L % d_k != 0:
                continue
            
            # Try both orders: descending and ascending
            for order in [indices_desc, indices_asc]:
                used = {L}
                reps = [0] * n
                reps[k] = L
                valid = True
                max_rep = L
                
                for i in order:
                    if i == k:
                        continue
                    d = divisors[i]
                    # Find smallest multiple of d >= L
                    m = L
                    rem = m % d
                    if rem != 0:
                        m += d - rem
                    
                    # Resolve collision
                    while m in used:
                        m += d
                    
                    if m > max_val:
                        valid = False
                        break
                    
                    used.add(m)
                    reps[i] = m
                    if m > max_rep:
                        max_rep = m
                
                if valid:
                    span = max_rep - L
                    if span < best_span:
                        best_span = span
                        best_result = reps[:]
                        if best_span <= 9:
                            return [int(x) for x in best_result]
    
    if best_result is None:
        best_result = divisors[:]
    
    return [int(x) for x in best_result]
