import time
import numpy as np
from scipy.optimize import milp, LinearConstraint

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # The domain [51, 100] is the most promising for maximizing cardinality
    # because the minimum sum of 4 distinct elements is 51+52+53+54 = 210 > 200.
    # Thus, only subsets of size <= 3 can sum to 200.
    # This reduces the problem to finding a Maximum Independent Set in a 3-uniform hypergraph.
    
    values = list(range(51, 101))
    n_vars = len(values)
    val_to_idx = {v: i for i, v in enumerate(values)}
    
    # Generate all bad triples (x, y, z) such that x < y < z and x+y+z == 200
    triples = []
    for z in range(100, 67, -1):
        target_xy = 200 - z
        # x + y = target_xy
        # 51 <= x < y < z
        # y = target_xy - x
        # x < target_xy - x  => 2x < target_xy => x < target_xy / 2
        # target_xy - x < z => x > target_xy - z = 200 - 2z
        # x >= 51
        
        min_x = max(51, 200 - 2*z + 1)
        max_x = (target_xy - 1) // 2
        
        for x in range(min_x, max_x + 1):
            y = target_xy - x
            if x < y < z:
                triples.append((x, y, z))
                
    if not triples:
        # If there are no bad triples, we can take all 50
        return values
        
    # Solve Minimum Hitting Set via ILP
    # Minimize sum(v_i) subject to sum(v_j for j in triple) >= 1 for all triples
    
    # Number of constraints = number of triples
    n_constraints = len(triples)
    
    # Create the constraint matrix A
    # A is sparse-ish, but dense enough for 50 vars and ~1000 triples to use dense numpy
    A = np.zeros((n_constraints, n_vars), dtype=np.float64)
    
    for i, t in enumerate(triples):
        for v in t:
            idx = val_to_idx[v]
            A[i, idx] = 1.0
            
    c = np.ones(n_vars, dtype=np.float64)
    bounds = [(0, 1)] * n_vars
    integrality = np.ones(n_vars, dtype=np.int32)
    
    H = None
    try:
        # Constraints: A @ x >= 1
        constraints = LinearConstraint(A, 1.0, np.inf)
        
        # Use HiGHS solver which is fast for small problems
        res = milp(
            c=c, 
            constraints=constraints, 
            bounds=bounds, 
            integrality=integrality,
            options={"presolve": True, "mip_rel_gap": 1e-6}
        )
        
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
        else:
            # Fallback if solver fails
            H = None
            
    except Exception:
        H = None

    best_set = None
    if H is not None:
        best_set = [v for v in values if v not in H]
        
    # Fallback / Comparison: Greedy Algorithm
    # This handles cases where ILP fails or if we want to compare.
    # The greedy method in the seed was decent, but ILP is exact for this sub-problem.
    # We will also run a simple greedy hitting set to have a backup.
    remaining_triples = [set(t) for t in triples]
    greedy_removed = set()
    while remaining_triples:
        # Find element in most remaining triples
        counts = {}
        for t in remaining_triples:
            for v in t:
                if v not in greedy_removed:
                    counts[v] = counts.get(v, 0) + 1
        if not counts:
            break
        max_count = max(counts.values())
        # Pick one with max count (tie-break by value for determinism)
        candidates = [v for v, c in counts.items() if c == max_count]
        v_to_remove = min(candidates)
        greedy_removed.add(v_to_remove)
        # Remove all triples containing v_to_remove
        remaining_triples = [t for t in remaining_triples if v_to_remove not in t]
        
    greedy_set = [v for v in values if v not in greedy_removed]
    
    # Choose the larger set (ILP should be >= Greedy)
    if best_set is None:
        return greedy_set
        
    if len(best_set) >= len(greedy_set):
        return best_set
    else:
        return greedy_set

# EVOLVE-BLOCK-END
