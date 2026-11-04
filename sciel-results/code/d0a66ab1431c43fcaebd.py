"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def solve(a, b):
    n = len(a)
    
    # Convert to float64 for the continuous relaxation
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    
    # Precompute row sums for a fast greedy initialization
    a_rs = A.sum(axis=1)
    b_rs = B.sum(axis=1)
    
    # Greedy initialization: assign heavy flow nodes to low distance nodes
    heavy_order = np.argsort(-a_rs, kind="stable")
    light_order = np.argsort(b_rs, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = light_order
    
    def compute_cost(p):
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total
    
    current_cost = compute_cost(p)
    
    # Fast swap delta using vectorized dot products
    def swap_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        # Difference in B rows and cols
        d_row = B[v] - B[u]
        d_col = B[:, v] - B[:, u]
        
        # Dot products with A rows and cols
        s = (np.dot(A[i], d_row[p]) + 
             np.dot(A[:, i], d_col[p]) - 
             np.dot(A[j], d_row[p]) - 
             np.dot(A[:, j], d_col[p]))
        
        # Subtract k=i and k=j terms
        # For k=i:
        s -= A[i, i] * (B[v, u] - B[u, u]) + A[i, i] * (B[u, v] - B[u, u])
        s += A[j, i] * (B[v, u] - B[u, u]) + A[i, j] * (B[u, v] - B[u, u])
        
        # For k=j:
        s -= A[i, j] * (B[v, v] - B[u, v]) + A[j, i] * (B[v, v] - B[u, v])
        s += A[j, j] * (B[v, v] - B[u, v]) + A[j, j] * (B[v, v] - B[u, v])
        
        # Direct i-j and j-i terms
        s += A[i, i] * (B[v, v] - B[u, u])
        s += A[j, j] * (B[u, u] - B[v, v])
        s += (A[i, j] + A[j, i]) * (B[v, u] - B[u, v])
        
        return s
    
    # Local search with multiple passes
    start_time = time.time()
    time_budget = 115.0
    
    improved = True
    pass_count = 0
    max_passes = 50
    
    while improved and pass_count < max_passes and (time.time() - start_time) < time_budget:
        improved = False
        pass_count += 1
        
        # Check time periodically
        for i in range(n):
            if (time.time() - start_time) > time_budget:
                break
            for j in range(i + 1, n):
                if (time.time() - start_time) > time_budget:
                    break
                d = swap_delta(i, j, p)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += d
                    improved = True
    
    return p.tolist()
# EVOLVE-BLOCK-END
