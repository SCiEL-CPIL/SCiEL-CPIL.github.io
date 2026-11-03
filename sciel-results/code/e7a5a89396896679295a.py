import numpy as np
import scipy.linalg as sla
from scipy.optimize import linear_sum_assignment
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # 1. Spectral Initialization
    # Compute eigenvalues and eigenvectors
    # A is symmetric, so use eigh
    eigvals_a, eigvecs_a = sla.eigh(a)
    eigvals_b, eigvecs_b = sla.eigh(b)
    
    # Sort A's eigenvectors by descending eigenvalues (strongest flows)
    idx_a = np.argsort(eigvals_a)[::-1]
    eigvecs_a = eigvecs_a[:, idx_a]
    
    # Sort B's eigenvectors by ascending eigenvalues (shortest distances)
    idx_b = np.argsort(eigvals_b)
    eigvecs_b = eigvecs_b[:, idx_b]
    
    best_p = None
    best_cost = np.inf
    
    # Try different numbers of components k
    for k in [1, 2, 3, 5, 10]:
        if time.time() - start_time > time_limit * 0.1:
            break
            
        ma = eigvecs_a[:, :k]
        mb = eigvecs_b[:, :k]
        
        # Cost matrix C_ij = - ma[i] dot mb[j]
        # We want to maximize ma[i] dot mb[j] to minimize cost
        c = -ma @ mb.T
        
        # Solve Linear Assignment Problem
        row_ind, col_ind = linear_sum_assignment(c)
        p = np.zeros(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        # Calculate cost
        cost = np.sum(a * b[np.ix_(p, p)])
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
    if best_p is None:
        # Fallback to random or identity
        best_p = np.arange(n)
        best_cost = np.sum(a * b)

    p = best_p
    current_cost = best_cost
    
    # 2. Local Search
    # We use a 2-swap local search
    # Delta for swapping i and j:
    # u = A[i] - A[j]
    # w = B[p[j]] - B[p[i]]
    # delta = u dot w
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                # Compute delta
                u = a[i] - a[j]
                w = b[p[j]] - b[p[i]]
                delta = np.dot(u, w)
                
                if delta < -1e-9: # Improvement
                    # Swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
    
    return p.tolist()
