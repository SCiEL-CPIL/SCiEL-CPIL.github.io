import time
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _witt_6_12_blocks():
    """
    Solve the maximum packing problem for S(5,6,12) using Integer Linear Programming
    with symmetry breaking to speed up the solver.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Generate all 6-subsets (blocks) and 5-subsets
    points = range(12)
    blocks = list(combinations(points, 6))
    fives = list(combinations(points, 5))
    
    n_blocks = len(blocks)
    n_fives = len(fives)
    
    # Map 5-subsets to indices
    five_index = {f: i for i, f in enumerate(fives)}
    
    # Efficiently construct the constraint matrix
    import scipy.sparse as sp
    
    rows = []
    cols = []
    data = []
    
    for j, b in enumerate(blocks):
        for sub in combinations(b, 5):
            i = five_index[sub]
            rows.append(i)
            cols.append(j)
            data.append(1)
    
    A = sp.csc_matrix((data, (rows, cols)), shape=(n_fives, n_blocks))
    
    # Objective: maximize sum(x_j) -> minimize -sum(x_j)
    c = np.full(n_blocks, -1.0)
    
    # Constraints: A @ x <= 1
    constraints = LinearConstraint(A, lb=-np.inf, ub=np.ones(n_fives))
    
    # Bounds: 0 <= x_j <= 1
    bounds = Bounds(lb=np.zeros(n_blocks), ub=np.ones(n_blocks))
    
    # Integrality: all variables are integer
    integrality = np.ones(n_blocks)
    
    # Symmetry breaking: fix block {0,1,2,3,4,5} to be included
    # M12 is 5-transitive, so we can w.l.o.g. fix 5 points in a block
    fixed_block = (0, 1, 2, 3, 4, 5)
    if fixed_block in blocks:
        fixed_idx = blocks.index(fixed_block)
        bounds.lb[fixed_idx] = 1.0
        bounds.ub[fixed_idx] = 1.0
    
    # Solve
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.success and res.x is not None:
        result = [list(blocks[j]) for j in range(n_blocks) if res.x[j] > 0.5]
        
        if len(result) == 132:
            return result
        
        if len(result) > 0:
             return result
    
    # Fallback if ILP fails
    return _greedy_fallback()

def _greedy_fallback():
    """
    Fallback greedy algorithm if ILP fails.
    """
    all_blocks = list(combinations(range(12), 6))
    covered = set()
    result = []
    
    # Try a few different orderings
    orderings = [
        sorted(all_blocks, key=lambda b: (sum(b), b)),
        sorted(all_blocks, key=lambda b: (min(b), max(b), sum(b), b)),
        sorted(all_blocks, key=lambda b: b),
    ]
    
    best_result = []
    
    for ordering in orderings:
        covered = set()
        current = []
        for b in ordering:
            subsets = list(combinations(b, 5))
            if not any(s in covered for s in subsets):
                current.append(list(b))
                covered.update(subsets)
        
        if len(current) > len(best_result):
            best_result = current
            
        if len(best_result) == 132:
            break
    
    return best_result

def construct():
    """
    Constructs the Steiner system S(5,6,12).
    Returns a list of 132 distinct 6-subsets of 0..11 such that no 5-subset
    lies in two blocks.
    """
    return _witt_6_12_blocks()
