import math
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin for 90s limit
    
    N = 256
    # Candidates are 2..256
    # Index i corresponds to number i+2
    n_nodes = 255
    numbers = list(range(2, 257))
    
    # Objective: Maximize sum(1/x_i)
    # milp minimizes c @ x, so we set c = -1/x_i
    c = np.array([-1.0 / x for x in numbers])
    
    # Constraints: x_i + x_j <= 1 for all conflicts (i, j)
    # We need to build the A matrix and b vector.
    # To avoid building a huge dense matrix, we can use sparse matrices if needed, 
    # but n_nodes=255, so a dense matrix of size E x 255 is fine if E is not huge.
    
    # First, count conflicts to estimate memory
    conflict_pairs = []
    for i in range(n_nodes):
        a = i + 2
        for j in range(i + 1, n_nodes):
            b = j + 2
            # lcm(a, b) = a * b // gcd(a, b)
            # Check if lcm <= 256
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                conflict_pairs.append((i, j))
                
    E = len(conflict_pairs)
    
    # If no conflicts, take all
    if E == 0:
        return numbers
        
    # Build constraint matrix A
    # Each constraint is x_i + x_j <= 1
    # A is E x n_nodes
    # b is E x 1 with all 1s
    
    # For efficiency, we can use a sparse matrix representation if E is large,
    # but scipy.optimize.milp accepts sparse matrices.
    import scipy.sparse as sp
    
    # Rows and cols for sparse matrix
    row_idx = []
    col_idx = []
    data = []
    
    for k, (i, j) in enumerate(conflict_pairs):
        row_idx.append(k)
        col_idx.append(i)
        data.append(1)
        
        row_idx.append(k)
        col_idx.append(j)
        data.append(1)
        
    A = sp.csr_matrix((data, (row_idx, col_idx)), shape=(E, n_nodes))
    b = np.ones(E)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(0, 1)
    
    # Integrality: 1 means integer
    integrality = np.ones(n_nodes)
    
    # Linear Constraint: A @ x <= b
    # scipy.optimize.milp uses LinearConstraint(A, lb, ub)
    # We want A @ x <= b, so lb = -inf, ub = b
    constraints = LinearConstraint(A, lb=-np.inf, ub=b)
    
    # Solve
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            # Extract solution
            x = res.x
            # x should be 0 or 1 (or very close)
            selected_indices = [i for i in range(n_nodes) if x[i] > 0.5]
            result = [idx + 2 for idx in selected_indices]
            
            if result:
                return result
            else:
                # Fallback if empty (shouldn't happen)
                return [2]
                
    except Exception as e:
        # Fallback to a simple greedy if MIP fails or times out
        # Greedy: sort by weight desc, add if no conflict
        pass

    # Fallback: Greedy
    # Sort candidates by weight descending (value 1/x, so x ascending)
    # But we need to check conflicts.
    # This is a valid independent set, though not necessarily optimal.
    
    # Precompute conflict sets for fast lookup
    conflict_sets = [set() for _ in range(n_nodes)]
    for i, j in conflict_pairs:
        conflict_sets[i].add(j)
        conflict_sets[j].add(i)
        
    selected = []
    selected_set = set()
    
    # Order by 1/x descending => x ascending
    # But wait, maximizing 1/x means we want small x.
    # So we iterate x from 2 upwards.
    for i in range(n_nodes):
        num = i + 2
        # Check if i conflicts with any selected
        if not conflict_sets[i].intersection(selected_set):
            selected.append(i)
            selected_set.add(i)
            
    return [idx + 2 for idx in selected]
