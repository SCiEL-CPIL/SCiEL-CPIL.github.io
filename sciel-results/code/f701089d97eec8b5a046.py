import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Heuristic Rank Matching
    # Assign facilities with high flow to locations with low distance sum (central)
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    fac_order = np.argsort(-flow_sums, kind="stable")
    loc_order = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    # Calculate initial cost
    B_perm = b[np.ix_(p, p)]
    cost = np.sum(a * B_perm)
    
    best_p = p.copy()
    best_cost = cost
    
    start_time = time.time()
    time_limit = 115  # Leave buffer for overhead
    
    # Iterated Local Search
    while time.time() - start_time < time_limit:
        improved = True
        # Local Search Phase: 2-opt
        while improved:
            improved = False
            # Iterate through all pairs
            for i in range(n):
                for k in range(i + 1, n):
                    u = p[i]
                    v = p[k]
                    
                    # Vectorized Delta Calculation
                    # S = sum_{j} (A[i,j] - A[k,j]) * (B[v, p[j]] - B[u, p[j]])
                    # Note: b[v, p] extracts the row v of b, indexed by p.
                    # This is equivalent to sum_j (A[i,j] - A[k,j]) * (B[v, p[j]] - B[u, p[j]])
                    
                    # Optimization: Precompute diff_a? No, i,k change.
                    # But we can compute diff_a = a[i] - a[k]
                    # And diff_b = b[v, p] - b[u, p]
                    
                    diff_a = a[i] - a[k]
                    diff_b = b[v, p] - b[u, p]
                    S = np.dot(diff_a, diff_b)
                    
                    # Subtract self and cross terms included in S
                    # j=i term: (a[i,i]-a[k,i]) * (b[v, p[i]] - b[u, p[i]]) = (a[i,i]-a[k,i]) * (b[v, u] - b[u, u])
                    # j=k term: (a[i,k]-a[k,k]) * (b[v, p[k]] - b[u, p[k]]) = (a[i,k]-a[k,k]) * (b[v, v] - b[u, v])
                    
                    S -= (a[i, i] - a[k, i]) * (b[v, u] - b[u, u])
                    S -= (a[i, k] - a[k, k]) * (b[v, v] - b[u, v])
                    
                    # Add correct new terms for i, k interaction and self loops
                    # New - Old:
                    # (a[i,k]*b[v,u] + a[k,i]*b[u,v] + a[i,i]*b[v,v] + a[k,k]*b[u,u])
                    # - (a[i,k]*b[u,v] + a[k,i]*b[v,u] + a[i,i]*b[u,u] + a[k,k]*b[v,v])
                    
                    delta = S + (a[i, k] * (b[v, u] - b[u, v]) + 
                                 a[k, i] * (b[u, v] - b[v, u]) + 
                                 a[i, i] * (b[v, v] - b[u, u]) + 
                                 a[k, k] * (b[u, u] - b[v, v]))
                    
                    if delta < 0:
                        p[i], p[k] = p[k], p[i]
                        cost += delta
                        improved = True
                        if cost < best_cost:
                            best_cost = cost
                            best_p = p.copy()
        
        # Perturbation Phase
        # Perform a few random swaps to escape local optimum
        # Number of swaps can be adaptive, but fixed small number is robust
        for _ in range(5):
            i = random.randint(0, n - 1)
            k = random.randint(0, n - 1)
            if i != k:
                p[i], p[k] = p[k], p[i]
        
        # Recalculate cost from scratch to avoid accumulation errors
        # N^2 operations, negligible
        B_perm = b[np.ix_(p, p)]
        cost = np.sum(a * B_perm)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    return best_p.tolist()
