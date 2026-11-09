import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def generate_forbidden_subsets(target, max_val):
    subsets = []
    path = []
    
    def backtrack(start, sum_so_far):
        if sum_so_far == target:
            if path:
                subsets.append(tuple(path))
            return
        if sum_so_far > target:
            return
        
        for i in range(start, max_val + 1):
            if sum_so_far + i > target:
                break
            path.append(i)
            backtrack(i + 1, sum_so_far + i)
            path.pop()
            
    backtrack(1, 0)
    return subsets

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Generate forbidden subsets
    forbidden = generate_forbidden_subsets(200, 100)
    
    if time.time() > deadline or not forbidden:
        return fallback_construct()
        
    # 2. Setup ILP
    n_vars = 100
    c = np.ones(n_vars)
    
    m_constraints = len(forbidden)
    
    # If constraints are too many, fallback to avoid memory/time issues
    if m_constraints > 200000:
        return fallback_construct()

    rows = []
    cols = []
    data = []
    b_ub = np.zeros(m_constraints, dtype=np.float64)
    
    for idx, subset in enumerate(forbidden):
        b_ub[idx] = len(subset) - 1
        for val in subset:
            rows.append(idx)
            cols.append(val - 1)
            data.append(1.0)
            
    if time.time() > deadline:
        return fallback_construct()

    A = csr_matrix((data, (rows, cols)), shape=(m_constraints, n_vars))
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n_vars)
    
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, ub=b_ub, lb=0),
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0 and res.x is not None:
            x_int = np.round(res.x).astype(int)
            selected = [i + 1 for i in range(n_vars) if x_int[i] == 1]
            if selected:
                return selected
            else:
                return fallback_construct()
        else:
            return fallback_construct()
            
    except Exception:
        return fallback_construct()

def fallback_construct():
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected
