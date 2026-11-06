"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Candidate set: [51, 100]
    # Any subset of size >= 4 sums to > 200.
    # No pairs sum to 200.
    # Only triples sum to 200.
    
    values = list(range(51, 101))
    n_vars = len(values)
    val_to_idx = {v: i for i, v in enumerate(values)}
    
    # Generate all bad triples
    triples = []
    for z in range(100, 67, -1):
        target_xy = 200 - z
        # x < y < z
        # x + y = target_xy
        # y = target_xy - x
        # x < target_xy - x => 2x < target_xy => x < target_xy / 2
        # y < z => target_xy - x < z => x > target_xy - z = 200 - 2z
        # x >= 51
        
        min_x = max(51, 200 - 2*z + 1)
        max_x = (target_xy - 1) // 2
        
        for x in range(min_x, max_x + 1):
            y = target_xy - x
            if x < y < z:
                triples.append((x, y, z))
                
    if not triples:
        return list(range(51, 101))
        
    # Solve Minimum Hitting Set via ILP
    # Minimize sum(v_i) subject to sum(v_j for j in triple) >= 1
    
    A = np.zeros((len(triples), n_vars), dtype=int)
    for i, t in enumerate(triples):
        for v in t:
            A[i, val_to_idx[v]] = 1
            
    c = np.ones(n_vars, dtype=int)
    bounds = [(0, 1)] * n_vars
    integrality = np.ones(n_vars, dtype=int)
    
    H = None
    try:
        constraints = LinearConstraint(A, 1, None) # Ax >= 1
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
        
        if res.status == 0:
            H_candidate = set()
            for i in range(n_vars):
                if res.x[i] > 0.5:
                    H_candidate.add(values[i])
            
            # Verify H_candidate covers all triples
            valid = True
            for t in triples:
                if not any(v in H_candidate for v in t):
                    valid = False
                    break
            
            if valid:
                H = H_candidate
    except Exception:
        H = None

    best_set = None
    if H is not None:
        best_set = [v for v in values if v not in H]
    
    # Fallback / Comparison: Greedy Algorithm (Seed Logic)
    # This handles cases where ILP fails or if mixing small numbers is better
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        # Time check
        if time.time() > deadline:
            break
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
            
    greedy_set = selected
    
    # Choose the larger set
    if best_set is None:
        return greedy_set
        
    if len(best_set) > len(greedy_set):
        return best_set
    else:
        return greedy_set

# EVOLVE-BLOCK-END
