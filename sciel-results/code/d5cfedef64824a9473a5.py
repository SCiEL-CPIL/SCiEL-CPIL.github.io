import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // gcd(a, b)

_LCM_ALL = reduce(_lcm, DIVISORS)  # 30030

def _matching_exists(L, W):
    """Check if there's a perfect matching: assign each divisor a distinct multiple in [L, L+W].
    Returns assignment list or None."""
    adj = []
    for i in range(N):
        d = DIVISORS[i]
        first = ((L + d - 1) // d) * d
        candidates = []
        v = first
        while v <= L + W:
            candidates.append(v)
            v += d
        adj.append(candidates)
    
    for i in range(N):
        if not adj[i]:
            return None
    
    # Kuhn's algorithm for bipartite matching
    match_right = {}  # value -> divisor index
    
    def try_kuhn(i, visited):
        for v in adj[i]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_right or try_kuhn(match_right[v], visited):
                match_right[v] = i
                return True
        return False
    
    for i in range(N):
        visited = set()
        if not try_kuhn(i, visited):
            return None
    
    assignment = [0] * N
    for v, i in match_right.items():
        assignment[i] = v
    
    return assignment

def _check_width(W, deadline):
    """Check if there exists a window of width W with a valid assignment."""
    # The pattern repeats with period _LCM_ALL = 30030
    # So we only need to check S in [1, 30030]
    for S in range(1, _LCM_ALL + 1):
        if time.time() > deadline:
            return None
        if S + W > MAX_VAL:
            break
        assignment = _matching_exists(S, W)
        if assignment is not None:
            # Verify all constraints
            vals = assignment
            if len(set(vals)) == N:
                if all(1 <= v <= MAX_VAL for v in vals):
                    for i, val in enumerate(vals):
                        if val % DIVISORS[i] != 0:
                            break
                    else:
                        return vals
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Search for minimal W starting from theoretical minimum
    # 10 distinct integers require range >= 9
    for W in range(9, 500):
        if time.time() > deadline:
            break
        result = _check_width(W, deadline)
        if result is not None:
            return [int(v) for v in result]
    
    # Fallback: return divisors themselves (valid but not optimal)
    return [int(d) for d in DIVISORS]
# EVOLVE-BLOCK-END
