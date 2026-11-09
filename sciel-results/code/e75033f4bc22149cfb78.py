import time
import sys
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
DEADLINE = 85.0  # Seconds, with safety margin

def _lcm(a, b):
    return a * b // gcd(a, b)

def _all_lcm():
    return reduce(_lcm, DIVISORS)

def _has_perfect_matching(divisors, L, W):
    """Check if there's a perfect matching from divisors to distinct values in [L, L+W]."""
    n = len(divisors)
    values = list(range(L, L + W + 1))
    
    adj = [[] for _ in range(n)]
    for i, d in enumerate(divisors):
        for j, v in enumerate(values):
            if v % d == 0:
                adj[i].append(j)
    
    for i in range(n):
        if not adj[i]:
            return None
    
    match_to = [-1] * len(values)
    match_from = [-1] * n
    
    def bpm(u, seen):
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_to[v] == -1 or bpm(match_to[v], seen):
                match_to[v] = u
                match_from[u] = v
                return True
        return False
    
    for u in range(n):
        seen = [False] * len(values)
        if not bpm(u, seen):
            return None
    
    result = [None] * n
    for u in range(n):
        result[u] = values[match_from[u]]
    return result

def construct():
    start_time = time.time()
    
    # For each width W from 9 upward, search for L with a perfect matching
    for W in range(9, 50):
        if time.time() - start_time > DEADLINE - 3:
            break
        
        max_L = min(200000, MAX_VAL - W)
        for L in range(1, max_L + 1):
            if time.time() - start_time > DEADLINE - 3:
                break
            
            # Quick feasibility check
            feasible = True
            for d in DIVISORS:
                start_k = (L + d - 1) // d
                end_k = (L + W) // d
                if start_k > end_k:
                    feasible = False
                    break
            if not feasible:
                continue
            
            res = _has_perfect_matching(DIVISORS, L, W)
            if res is not None:
                return [int(x) for x in res]
    
    # Fallback: return divisors themselves (valid but not optimal)
    return [int(d) for d in DIVISORS]
# EVOLVE-BLOCK-END
