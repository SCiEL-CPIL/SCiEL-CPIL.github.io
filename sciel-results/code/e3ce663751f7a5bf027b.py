"""Bootstrap portfolio: MODEL-AND-OPTIMIZE for Egyptian fraction of 5/31."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 55.0  # Safety margin for 60s limit

def _greedy_complete(remaining: Fraction, used: set, start_d: int):
    """
    Complete the Egyptian fraction greedily from the current state.
    Returns a list of denominators added, or None if it fails (exceeds limits).
    """
    result = []
    r = remaining
    last_d = start_d
    while r > 0:
        if r.numerator == 0:
            break
        # Standard greedy step: ceil(1/r)
        d = (r.denominator + r.numerator - 1) // r.numerator
        
        # Ensure distinctness and increasing order
        while d in used or d <= last_d:
            d += 1
            
        if d > MAX_DENOM:
            return None
            
        if len(result) + len(used) >= MAX_TERMS:
            return None

        result.append(d)
        used.add(d)
        r -= Fraction(1, d)
        last_d = d
        
    return result

def _solve():
    """
    Search for a representation of 5/31 with minimal largest denominator.
    Strategy: 
    1. Enumerate small first denominators.
    2. For each, enumerate small second denominators.
    3. Complete the rest greedily.
    4. Track the solution with the minimum max(denominators).
    """
    start_time = time.time()
    best_solution = None
    best_max_d = MAX_DENOM + 1
    
    # The sum of reciprocals must be 5/31 ~ 0.16129
    # 1/d1 < 5/31 => d1 > 31/5 = 6.2 => d1 >= 7
    # If d1 is too large, the remaining sum is small, leading to large denominators.
    # We focus on small d1 to keep the "mass" of the sum distributed to smaller denominators.
    
    # Heuristic: The optimal max denominator is likely not huge.
    # Let's try d1 in range [7, 20] and d2 in range [d1+1, d1+20]
    
    for d1 in range(7, 25):
        if time.time() - start_time > DEADLINE:
            break
        r1 = TARGET - Fraction(1, d1)
        if r1 <= 0:
            continue
        
        used = {d1}
        
        # For the second term, we want it to be as large as possible (small denominator)
        # but distinct.
        # Greedy for r1 would be ceil(1/r1).
        # But we can explore a small window around the greedy choice or smaller denominators.
        # Actually, to minimize the MAX denominator, we want the remaining terms to be small.
        # This usually means taking the largest possible denominators (smallest 1/d) early?
        # No, standard greedy takes largest 1/d (smallest d). 
        # If we take a smaller d (larger 1/d), the remainder is smaller.
        # Smaller remainder often leads to larger denominators in the tail.
        # So we want to take LARGER d (smaller 1/d) early? 
        # Wait. 
        # Let's look at the current program: d1=8. 
        # 5/31 - 1/8 = (40-31)/248 = 9/248.
        # Greedy for 9/248: ceil(248/9) = 28.
        # 9/248 - 1/28 = (9*28 - 248) / (248*28) = (252-248)/6944 = 4/6944 = 1/1736.
        # So {8, 28, 1736}. Max is 1736.
        
        # What if we pick d1=7?
        # 5/31 - 1/7 = (35-31)/217 = 4/217.
        # Greedy: ceil(217/4) = 55.
        # 4/217 - 1/55 = (220-217)/(217*55) = 3/11935.
        # Greedy: ceil(11935/3) = 3979.
        # 3/11935 - 1/3979 = (3*3979 - 11935) / (11935*3979) = (11937-11935)/... = 2/...
        # This path seems to lead to larger numbers too.
        
        # What if we pick d1=9?
        # 5/31 - 1/9 = (45-31)/279 = 14/279.
        # Greedy: ceil(279/14) = 20.
        # 14/279 - 1/20 = (280-279)/(279*20) = 1/5580.
        # So {9, 20, 5580}. Max 5580. Worse.
        
        # What if we pick d1=8, but change d2?
        # r1 = 9/248.
        # Greedy d2=28.
        # If we pick d2=29?
        # 9/248 - 1/29 = (9*29 - 248) / (248*29) = (261-248)/7192 = 13/7192.
        # Greedy d3: ceil(7192/13) = 554.
        # 13/7192 - 1/554 = (13*554 - 7192) / (7192*554) = (7202-7192)/... = 10/...
        # This is getting complex.
        
        # Let's just run the search for d1 in [7..20] and d2 in [d1+1 .. d1+50].
        # For each pair, complete greedily.
        
        for d2 in range(d1 + 1, d1 + 60):
            if time.time() - start_time > DEADLINE:
                break
            if d2 in used:
                continue
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            
            used2 = used | {d2}
            
            # Complete greedily
            completion = _greedy_complete(r2, used2, d2)
            
            if completion is not None:
                full_solution = [d1, d2] + completion
                max_d = max(full_solution)
                
                if max_d < best_max_d:
                    best_max_d = max_d
                    best_solution = full_solution
                    
                    # Early exit if we find something very good?
                    # Theoretical lower bound for max d is not trivial, but 1000 is good.
                    if max_d < 100:
                        return best_solution

    # If the 2-step search didn't find a great one, try 1-step greedy for a wider range of d1
    # Or just return the best found. If none found (shouldn't happen), fall back to pure greedy.
    if best_solution is None:
        # Pure greedy fallback
        r = TARGET
        used = set()
        last_d = 0
        vals = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            while d in used or d <= last_d:
                d += 1
            if d > MAX_DENOM:
                break
            vals.append(d)
            used.add(d)
            r -= Fraction(1, d)
            last_d = d
        best_solution = vals
        best_max_d = max(vals) if vals else MAX_DENOM

    return best_solution

def construct():
    """
    Entry point. Returns a list of distinct denominators summing to 5/31.
    Minimizes the largest denominator.
    """
    return _solve()
# EVOLVE-BLOCK-END
