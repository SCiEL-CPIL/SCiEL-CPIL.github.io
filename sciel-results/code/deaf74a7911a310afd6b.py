"""Rewritten program: DIRECT approach to pack distinct multiples of fixed divisors."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # Calculate LCM of all divisors to exploit periodicity
    def lcm(a, b):
        return a * b // gcd(a, b)
    lcm_all = reduce(lcm, divisors)
    
    # Hopcroft-Karp style bipartite matching with greedy initialization
    def find_assignment(candidates):
        # candidates: list of lists of values
        # We want to assign each divisor i to a distinct value from candidates[i]
        
        # Sort divisors by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        
        # match_val[v] = u  (value v is matched to divisor u)
        # match_div[u] = v  (divisor u is matched to value v)
        match_val = {}
        match_div = {}
        
        # Greedy initialization: try to match each divisor to an unmatched value
        for u in order:
            for v in candidates[u]:
                if v not in match_val:
                    match_val[v] = u
                    match_div[u] = v
                    break
        
        # If greedy succeeded, we're done
        if len(match_div) == n:
            assignment = [0] * n
            for u, v in match_div.items():
                assignment[u] = v
            return assignment
        
        # Use Kuhn's algorithm to fix remaining matches
        def try_kuhn(u, visited):
            for v in candidates[u]:
                if v in visited:
                    continue
                visited.add(v)
                if v not in match_val or try_kuhn(match_val[v], visited):
                    match_val[v] = u
                    match_div[u] = v
                    return True
            return False
        
        # Try to find matches for unmatched divisors
        for u in order:
            if u not in match_div:
                visited = set()
                if not try_kuhn(u, visited):
                    return None
        
        assignment = [0] * n
        for u, v in match_div.items():
            assignment[u] = v
        return assignment

    # Iterate span W from minimum possible (9) upwards
    for W in range(9, 2000):
        if time.time() > deadline:
            break
        
        # For each window position, check if we can pack all divisors
        # Use periodicity: only need to check L in [1, lcm_all]
        for L in range(1, lcm_all + 1):
            if time.time() > deadline:
                break
            
            # Check if window [L, L+W] is within bounds
            if L + W > MAX_V:
                continue

            candidates = []
            valid_window = True
            
            for d in divisors:
                # Find multiples of d in [L, L+W]
                first_k = (L + d - 1) // d
                last_k = (L + W) // d
                
                if first_k > last_k:
                    valid_window = False
                    break
                
                # Generate the list of candidates
                cands = [k * d for k in range(first_k, last_k + 1)]
                candidates.append(cands)
            
            if not valid_window:
                continue
            
            # Try to assign distinct values
            result = find_assignment(candidates)
            
            if result is not None:
                # Verify the solution
                if len(set(result)) != n:
                    continue
                for i, d in enumerate(divisors):
                    if result[i] % d != 0:
                        continue
                if any(v < 1 or v > MAX_V for v in result):
                    continue
                
                return [int(v) for v in result]
        
        # If we finish all L for a given W without success, increment W
    
    # Fallback: use the divisors themselves (may not be optimal but valid)
    fallback = [int(d) for d in divisors]
    return fallback

# EVOLVE-BLOCK-END
