"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Strategy: Use [51,100] as base, solve min hitting set for bad triples via ILP
    # Then try local search to add elements < 51 if possible
    
    values = list(range(51, 101))
    n_vars = len(values)
    val_to_idx = {v: i for i, v in enumerate(values)}
    
    # Generate all bad triples (x+y+z=200, 51<=x<y<z<=100)
    triples = []
    for z in range(100, 67, -1):
        target_xy = 200 - z
        min_x = max(51, 200 - 2*z + 1)
        max_x = (target_xy - 1) // 2
        for x in range(min_x, max_x + 1):
            y = target_xy - x
            if x < y < z:
                triples.append((x, y, z))
                
    if not triples:
        return list(range(51, 101))
        
    # Solve Minimum Hitting Set via ILP
    A = np.zeros((len(triples), n_vars), dtype=int)
    for i, t in enumerate(triples):
        for v in t:
            A[i, val_to_idx[v]] = 1
            
    c = np.ones(n_vars, dtype=int)
    bounds = [(0, 1)] * n_vars
    integrality = np.ones(n_vars, dtype=int)
    
    H = None
    try:
        constraints = LinearConstraint(A, 1, None)
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
        if res.status == 0:
            H_candidate = set()
            for i in range(n_vars):
                if res.x[i] > 0.5:
                    H_candidate.add(values[i])
            valid = True
            for t in triples:
                if not any(v in H_candidate for v in t):
                    valid = False
                    break
            if valid:
                H = H_candidate
    except Exception:
        H = None

    if H is None:
        # Fallback: greedy hitting set
        active_triples = set(triples)
        H = set()
        while active_triples and time.time() < deadline - 5:
            counts = {}
            for t in active_triples:
                for v in t:
                    counts[v] = counts.get(v, 0) + 1
            if not counts:
                break
            best_elem = max(counts, key=counts.get)
            H.add(best_elem)
            active_triples = {t for t in active_triples if best_elem not in t}
    
    best_set = set(v for v in values if v not in H)
    best_list = sorted(best_set)
    
    # Try to add elements < 51 via local search
    # For each candidate x in 1..50, check if adding it keeps the set valid
    # A set is valid if no subset sums to 200
    # Since best_set ⊆ [51,100], subsets summing to 200 are only triples (size 3)
    # When we add x < 51, we need to check:
    # - No pair in best_set sums to 200-x (i.e., 200-x is in [102, 199], possible for x in [1,98])
    # - No triple in best_set sums to 200-x (i.e., 200-x in [153, 199], possible for x in [1,47])
    # - No larger subset (but 4 elements from [51,100] sum to >= 206 > 200, so no)
    # - x itself doesn't cause issues (x < 200 always)
    
    def is_valid_with_extra(S, x):
        """Check if S ∪ {x} has no subset summing to 200."""
        target = 200 - x
        if target <= 0:
            return False
        # Check if any subset of S sums to target
        # Since S ⊆ [51,100], max subset size for sum <= 200 is 3 (4*51=204>200)
        # So check pairs and triples in S that sum to target
        S_list = sorted(S)
        n = len(S_list)
        # Check pairs
        for i in range(n):
            for j in range(i+1, n):
                if S_list[i] + S_list[j] == target:
                    return False
        # Check triples
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if S_list[i] + S_list[j] + S_list[k] == target:
                        return False
        return True
    
    # Try adding elements from 50 down to 1
    added = set()
    for x in range(50, 0, -1):
        if time.time() > deadline - 2:
            break
        if is_valid_with_extra(best_set, x):
            best_set.add(x)
            added.add(x)
    
    final_set = sorted(best_set)
    
    # Final verification
    # Check no subset of final_set sums to 200
    # Use DP for verification (safe)
    reachable = {0}
    for v in final_set:
        new_reach = {s + v for s in reachable if s + v <= 200}
        if 200 in new_reach:
            # Should not happen, but fallback to [51,100] ILP result
            return best_list
        reachable |= new_reach
    
    return final_set

# EVOLVE-BLOCK-END
