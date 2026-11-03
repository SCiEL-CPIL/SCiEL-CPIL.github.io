"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the QAP cost for permutation p."""
    # p[i] = location assigned to facility i
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Use matrix multiplication for speed
    bp = b[np.ix_(p, p)]
    return np.sum(a * bp)

def _k_opt(a, b, p, k, max_iter=1000, time_limit=5.0):
    """
    K-opt local search: try swapping pairs of facility locations.
    For k=2, we try all pairs (i,j) and swap p[i] and p[j].
    """
    n = len(a)
    p = np.array(p, dtype=np.int64)
    best_cost = _qap_cost(a, b, p)
    
    # Precompute b rows for faster access
    b_rows = b  # b[p[i]] is the row
    
    start_time = time.time()
    no_improve_count = 0
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Try all pairs (i, j) with i < j
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Swap p[i] and p[j]
                pi, pj = p[i], p[j]
                p[i], p[j] = pj, pi
                
                new_cost = _qap_cost(a, b, p)
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    improved = True
                    no_improve_count = 0
                else:
                    # Revert
                    p[i], p[j] = pi, pj
        
        if not improved:
            break
    
    return p.tolist(), best_cost

def _simulated_annealing(a, b, p, time_limit=10.0, start_temp=1000.0, cool_rate=0.999):
    """
    Simulated annealing for QAP.
    """
    n = len(a)
    p = np.array(p, dtype=np.int64)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    temp = start_temp
    
    while time.time() - start_time < time_limit:
        # Generate a random swap
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        pi, pj = p[i], p[j]
        p[i], p[j] = pj, pi
        
        new_cost = _qap_cost(a, b, p)
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = p.copy()
        else:
            # Revert
            p[i], p[j] = pi, pj
        
        temp *= cool_rate
    
    return best_p.tolist(), best_cost

def solve(a, b):
    n = len(a)
    deadline = 140.0  # Leave 10s margin before 150s kill
    start_time = time.time()
    
    # Step 1: Generate initial permutation using rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    p_init = p_init.tolist()
    
    best_p = p_init
    best_cost = _qap_cost(a, b, p_init)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_p
    
    # Step 2: Try Hungarian-based initialization
    # Use a linearized surrogate: minimize sum_i a_row_sum[i] * b_col_sum[p[i]]
    # This is a linear assignment problem
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=0)  # For symmetric b, same as row sums
    
    # Cost matrix for linear assignment: C[i, j] = a_row_sums[i] * b_col_sums[j]
    C = np.outer(a_row_sums, b_col_sums)
    
    # Solve linear assignment
    row_ind, col_ind = linear_sum_assignment(C)
    p_hungarian = np.empty(n, dtype=np.int64)
    p_hungarian[row_ind] = col_ind
    p_hungarian = p_hungarian.tolist()
    
    cost_hungarian = _qap_cost(a, b, p_hungarian)
    if cost_hungarian < best_cost:
        best_cost = cost_hungarian
        best_p = p_hungarian
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        # Not enough time for local search, do quick 2-opt
        best_p, best_cost = _k_opt(a, b, best_p, k=2, max_iter=100, time_limit=deadline - elapsed - 5)
        return best_p
    
    # Step 3: Local search on best initial solution
    remaining = deadline - elapsed - 5
    if remaining > 5:
        best_p, best_cost = _k_opt(a, b, best_p, k=2, max_iter=500, time_limit=remaining)
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 5
    if remaining > 10:
        # Step 4: Simulated annealing
        np.random.seed(42)
        best_p, best_cost = _simulated_annealing(a, b, best_p, time_limit=remaining)
    
    return best_p
# EVOLVE-BLOCK-END
