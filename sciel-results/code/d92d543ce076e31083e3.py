import time
import bisect
from math import gcd
from functools import reduce

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    max_val = P['maximum']
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Compute LCM of all divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = reduce(lcm, divisors)
    # period = 30030
    
    # Precompute for each divisor, the residues modulo period
    # For divisor d, multiples are at positions k*d mod period
    # We'll precompute which positions in [0, period-1] are multiples of each divisor
    
    # For each divisor d, the set of residues r in [0, period-1] such that r ≡ 0 (mod d)
    # But actually, we need to think about it differently.
    
    # For a window [L, L+W], a multiple of d exists iff there exists k such that
    # L <= k*d <= L+W, i.e., ceil(L/d) <= k <= floor((L+W)/d)
    
    # Due to periodicity with period = LCM, if [L, L+W] is feasible, then
    # [L + t*period, L + t*period + W] is also feasible for any t where the window stays in [1, max_val].
    
    # So for checking feasibility of width W, we only need to check L in [1, min(period + W, max_val - W)]
    
    # Precompute multiples as before
    multiples = {}
    for d in divisors:
        multiples[d] = list(range(d, max_val + 1, d))
    
    # Search for minimum W
    for W in range(0, 1000):
        if time.time() > deadline:
            break
        
        # Max L to check due to periodicity
        max_L = min(period + W, max_val - W)
        if max_L < 1:
            continue
        
        # Check L values. Due to periodicity, we only need to check up to period + W.
        # But we can be smarter: only check L where at least one divisor has a multiple at the boundary.
        # For efficiency, let's check all L from 1 to max_L, but break early.
        
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                break
            
            sol = _solve_window_fast(L, W, divisors, multiples)
            if sol is not None:
                return sol
        
        # If no solution found for this W, continue
    
    # Fallback: return the divisors themselves (they are distinct and divisible by themselves)
    # But we need to verify they're <= max_val, which they are.
    result = list(divisors)
    return result

def _solve_window_fast(L, W, divisors, multiples):
    """Check if window [L, L+W] has a valid assignment using bipartite matching."""
    n = len(divisors)
    
    # Build adjacency: for each divisor i, list of available values (multiples in [L, L+W])
    available = []
    for i, d in enumerate(divisors):
        m_list = multiples[d]
        left_idx = bisect.bisect_left(m_list, L)
        right_idx = bisect.bisect_right(m_list, L + W)
        if left_idx >= right_idx:
            return None
        avail = m_list[left_idx:right_idx]
        available.append(avail)
    
    # Use Hopcroft-Karp or simple DFS matching
    # Since n is small (10), simple DFS should work
    
    # Map values to indices for quick lookup
    # We'll use a set of used values
    
    assignment = [-1] * n  # assignment[i] = value assigned to divisor i
    used_values = set()
    
    # Order by domain size (MRV)
    order = sorted(range(n), key=lambda i: len(available[i]))
    
    def backtrack(idx):
        if idx == n:
            return True
        
        var_idx = order[idx]
        
        for val in available[var_idx]:
            if val not in used_values:
                used_values.add(val)
                assignment[var_idx] = val
                
                if backtrack(idx + 1):
                    return True
                
                used_values.remove(val)
                assignment[var_idx] = -1
        
        return False
    
    if backtrack(0):
        return assignment
    return None
