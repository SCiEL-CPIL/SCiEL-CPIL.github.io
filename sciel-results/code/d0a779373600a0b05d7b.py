import math
import time
import heapq
from functools import lru_cache

T12 = math.factorial(12)          # 479001600
PRIMES = [2, 3, 5, 7, 11]
EXP_TOTAL = (10, 5, 2, 1, 1)      # 2^10 3^5 5^2 7 11

# ---- exponent-vector helpers ---------------------------------------------
def _exp_vector(n):
    v = [0, 0, 0, 0, 0]
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            n //= p
            v[i] += 1
    return tuple(v)

# Precompute every divisor of T12 that is > 12, with its exponent vector.
DIVS = []          # sorted divisors > 12
DIVS_EV = []       # parallel exponent vectors
def _init_divisors():
    limit = math.isqrt(T12)
    seen = set()
    for i in range(1, limit + 1):
        if T12 % i == 0:
            for d in (i, T12 // i):
                if d > 12 and d not in seen:
                    seen.add(d)
                    DIVS.append(d)
    DIVS.sort()
    for d in DIVS:
        DIVS_EV.append(_exp_vector(d))

_init_divisors()
N_DIVS = len(DIVS)

def _valid(factors):
    if not (1 <= len(factors) <= 12):
        return False
    if len(set(factors)) != len(factors):
        return False
    if any(f <= 12 for f in factors):
        return False
    p = 1
    for f in factors:
        p *= f
    return p == T12

# ---- explicit incumbent (from prime-power grouping) -----------------------
# 144=2^4*3^2, 64=2^6, 77=7*11, 25=5^2, 27=3^3  -> 2^10 3^5 5^2 7 11
CANDIDATE = [144, 64, 77, 25, 27]

# ---- Minimax path search using Dijkstra on exponent-vector DAG ------------
# State: (e2, e3, e5, e7, e11) - remaining exponents
# We want to find a path from EXP_TOTAL to (0,0,0,0,0) minimizing the max factor used.
# This is a minimax shortest path problem.

def _minimax_search(deadline):
    """Find decomposition minimizing the largest factor using Dijkstra-like search."""
    START = EXP_TOTAL
    GOAL = (0, 0, 0, 0, 0)
    
    # dist[state] = minimum possible maximum factor to reach this state from START
    dist = {START: 0}
    # prev[state] = (prev_state, factor_used)
    prev = {START: None}
    
    # Priority queue: (max_factor_so_far, state)
    pq = [(0, START)]
    
    while pq:
        if time.time() > deadline:
            break
        
        cur_max, state = heapq.heappop(pq)
        
        # Skip if we've found a better way to this state
        if cur_max > dist.get(state, float('inf')):
            continue
        
        # Check if we reached the goal
        if state == GOAL:
            # Reconstruct path
            factors = []
            s = state
            while prev[s] is not None:
                ps, f = prev[s]
                factors.append(f)
                s = ps
            factors.reverse()
            return factors
        
        # Explore neighbors
        for i in range(N_DIVS):
            d = DIVS[i]
            ev = DIVS_EV[i]
            # Check if we can subtract this exponent vector
            new_state = tuple(state[k] - ev[k] for k in range(5))
            if any(ns < 0 for ns in new_state):
                continue
            
            # New max factor
            new_max = max(cur_max, d)
            
            if new_max < dist.get(new_state, float('inf')):
                dist[new_state] = new_max
                prev[new_state] = (state, d)
                heapq.heappush(pq, (new_max, new_state))
    
    return None

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin under 90s

    # Try minimax search first
    result = _minimax_search(deadline)
    if result is not None and _valid(result):
        return result
    
    # Fallback to explicit candidate
    if _valid(CANDIDATE):
        return list(CANDIDATE)
    
    # Last resort
    return [T12]
