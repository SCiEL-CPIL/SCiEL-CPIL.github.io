"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Compute LCM of all divisors
    lcm_val = reduce(_lcm, divisors)
    # lcm_val = 30030
    
    def get_candidates(d, s, R):
        """Get all multiples of d in [s, s+R]"""
        k_min = (s + d - 1) // d  # ceil(s/d)
        k_max = (s + R) // d       # floor((s+R)/d)
        if k_min > k_max:
            return []
        return [k * d for k in range(k_min, k_max + 1)]
    
    def bipartite_match(cands):
        """
        Check if there's a perfect matching using Hall's theorem via max-flow/bipartite matching.
        cands[i] is the set of values available to divisor i.
        Returns an assignment list or None.
        """
        # Build adjacency: for each value, which divisors can take it
        # Use Hopcroft-Karp or simple augmenting path
        # Since n is small (10), simple DFS-based matching works well
        
        # Map values to indices
        all_vals = set()
        for c in cands:
            all_vals.update(c)
        
        val_to_idx = {v: i for i, v in enumerate(sorted(all_vals))}
        num_vals = len(all_vals)
        
        # adj[i] = list of value indices that divisor i can take
        adj = []
        for c in cands:
            adj.append([val_to_idx[v] for v in c])
        
        # Bipartite matching: left side = divisors (0..n-1), right side = values (0..num_vals-1)
        match_right = [-1] * num_vals  # which divisor is matched to this value
        
        def try_augment(u, visited):
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if match_right[v] == -1 or try_augment(match_right[v], visited):
                    match_right[v] = u
                    return True
            return False
        
        # Try to match each divisor
        for u in range(n):
            visited = [False] * num_vals
            if not try_augment(u, visited):
                return None
        
        # Extract assignment
        assignment = [None] * n
        for v in range(num_vals):
            if match_right[v] != -1:
                assignment[match_right[v]] = sorted(all_vals)[v]
        
        return assignment
    
    def feasible(s, R):
        """Check if we can assign distinct multiples to each divisor in window [s, s+R]"""
        cands = []
        for d in divisors:
            c = get_candidates(d, s, R)
            if not c:
                return None
            cands.append(c)
        
        result = bipartite_match(cands)
        if result is not None:
            # Verify all values are positive and within bounds
            for val in result:
                if val is None or val <= 0 or val > 1000000:
                    return None
            return result
        return None
    
    # Binary search on R
    # Lower bound: 1, Upper bound: lcm_val (guaranteed feasible at R=lcm_val)
    lo = 1
    hi = lcm_val
    
    best_result = None
    best_R = None
    
    # First, verify hi is feasible
    found_hi = False
    for s in range(lcm_val):
        if time.time() > deadline:
            break
        result = feasible(s, hi)
        if result is not None:
            best_result = result
            best_R = hi
            found_hi = True
            break
    
    if not found_hi and best_result is None:
        # Fallback
        best_result = divisors[:]
        return [int(x) for x in best_result]
    
    # Binary search for minimum R
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        found_mid = False
        for s in range(lcm_val):
            if time.time() > deadline:
                break
            result = feasible(s, mid)
            if result is not None:
                best_result = result
                best_R = mid
                hi = mid
                found_mid = True
                break
        if not found_mid:
            lo = mid + 1
    
    # Ensure we return Python ints in a list
    if best_result is None:
        best_result = divisors[:]
    
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
