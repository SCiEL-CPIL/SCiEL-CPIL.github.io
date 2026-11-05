import time
from math import lcm
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    n = 128
    elements = list(range(1, n + 1))
    
    # 1. Identify all "bad" triples
    bad_triples = []
    # Optimization: We can precompute lcms or just compute on the fly.
    # 340k triples * 3 lcms each is ~1M lcm calls.
    # Python's math.lcm is C-implemented and fast.
    # Let's check time.
    
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            for k in range(j + 1, n):
                if time.time() > deadline:
                    break
                a, b, c = i + 1, j + 1, k + 1
                if lcm(a, b) == lcm(a, c) == lcm(b, c):
                    bad_triples.append((i, j, k))
    
    # If no bad triples, we can select all
    if not bad_triples:
        return elements
        
    # 2. Set up ILP
    # Maximize sum(x_i)
    # s.t. x_i + x_j + x_k <= 2 for all (i,j,k) in bad_triples
    #      0 <= x_i <= 1, x_i integer
    
    # Objective
    c = np.ones(n)
    
    # Constraints
    if len(bad_triples) == 0:
        return elements
        
    # Build sparse matrix A
    rows = []
    cols = []
    data = []
    b_vals = []
    
    for idx, (i, j, k) in enumerate(bad_triples):
        rows.extend([idx, idx, idx])
        cols.extend([i, j, k])
        data.extend([1, 1, 1])
        b_vals.append(2)
        
    A = csr_matrix((data, (rows, cols)), shape=(len(bad_triples), n))
    b = np.array(b_vals)
    
    # Bounds
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality
    integrality = np.ones(n)
    
    # Linear Constraint: A @ x <= b
    # scipy.optimize.milp expects LinearConstraint(A, lb, ub)
    # We want A @ x <= b, so lb=-inf, ub=b
    constraint = LinearConstraint(A, ub=b)
    
    # Solve
    try:
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.success and res.x is not None:
            # res.x contains the optimal values
            # Due to floating point, round to nearest integer
            x = np.round(res.x).astype(int)
            selected = [elements[i] for i in range(n) if x[i] == 1]
            if selected:
                return selected
            else:
                # Fallback if solver returns empty or fails
                return [1]
        else:
            # Fallback
            pass
    except Exception:
        pass
        
    # Fallback: Greedy approach (same as seed but maybe slightly better ordering?)
    # Or just return the seed result to ensure validity
    selected = []
    for x in elements:
        valid = True
        for a, b in combinations(selected, 2):
            if lcm(a, b) == lcm(a, x) == lcm(b, x):
                valid = False
                break
        if valid:
            selected.append(x)
            
    return selected if selected else [1]
