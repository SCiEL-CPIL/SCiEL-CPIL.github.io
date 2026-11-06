import time
import random
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    MAX_TIME = 115.0 # Safety margin
    
    N = 511
    # Precompute values
    values = np.array([0.0] * (N + 1))
    for i in range(1, N + 1):
        values[i] = 1.0 / (512.0 - i)
        
    # Precompute conflicts for constraint generation
    # We need pairs (a, b) with a < b and gcd(a, b) > 1
    # To speed up constraint generation, we can just iterate all pairs or use a smarter way.
    # N=511, N^2/2 ~ 130k pairs. This is small enough to generate all constraints directly.
    
    # We will build the constraint matrix A_ub and b_ub for:
    # sum(c * x) <= b
    # Here constraints are x_a + x_b <= 1 for conflicting pairs.
    
    # Since the number of constraints might be large, we can use sparse matrices.
    # But for 511 variables and ~100k constraints, dense might be okay or we use scipy.sparse.
    
    # Let's count conflicts first to estimate size
    conflict_pairs = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if gcd(a, b) > 1:
                conflict_pairs.append((a, b))
                
    num_constraints = len(conflict_pairs)
    
    # If no conflicts, just pick all. But there are conflicts.
    
    # Objective: maximize sum(values[a] * x_a)
    # milp minimizes c @ x. So we minimize -values @ x.
    c = -values[1:N+1]
    
    # Variables x_1, ..., x_511 are binary.
    integrality = np.ones(N, dtype=int)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Constraints: x_a + x_b <= 1
    # A_ub is (num_constraints, N)
    # b_ub is (num_constraints,)
    
    # Using sparse matrix for efficiency
    from scipy.sparse import coo_matrix
    
    row_indices = []
    col_indices = []
    data = []
    
    for idx, (a, b) in enumerate(conflict_pairs):
        # x_a is at index a-1, x_b is at index b-1
        row_indices.append(idx)
        col_indices.append(a - 1)
        data.append(1.0)
        
        row_indices.append(idx)
        col_indices.append(b - 1)
        data.append(1.0)
        
    A_ub = coo_matrix((data, (row_indices, col_indices)), shape=(num_constraints, N))
    b_ub = np.ones(num_constraints)
    
    # Linear constraints
    cons = LinearConstraint(A_ub, ub=b_ub)
    
    # Solve
    try:
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds, options={"time_limit": MAX_TIME - (time.time() - start_time) - 2.0})
        
        if res.status == 0:
            # Extract solution
            x = res.x
            # x is float, should be close to 0 or 1
            selected = [i + 1 for i in range(N) if x[i] > 0.5]
            
            # Verify validity
            if all(gcd(a, b) == 1 for i, a in enumerate(selected) for b in selected[i+1:]):
                return selected
            else:
                # Fallback if solver returned invalid (should not happen with binary integrality)
                pass
        elif res.status == 9:
            # Time limit reached, but might have a feasible solution
            if res.x is not None:
                 x = res.x
                 selected = [i + 1 for i in range(N) if x[i] > 0.5]
                 if all(gcd(a, b) == 1 for i, a in enumerate(selected) for b in selected[i+1:]):
                     return selected
    except Exception as e:
        # If MILP fails, fall back to a simple greedy
        pass

    # Fallback: Greedy
    chosen = []
    for a in range(N, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    return chosen
