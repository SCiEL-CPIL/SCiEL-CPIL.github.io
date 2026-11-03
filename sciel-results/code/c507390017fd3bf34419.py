"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Convert to float64 for consistent behavior and speed in some BLAS routines
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    
    # 1. Initialization: Rank-matching
    # Heavy flows (high row sums in A) go to short distances (low row sums in B)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_sums, kind="stable")
    central = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Calculate initial cost
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Using einsum for efficiency
    cost = np.einsum('ij,ik,jk->', a, b[np.ix_(p, p)])
    
    best_p = p.copy()
    best_cost = cost
    
    # Safety margin: 10s before the 120s limit
    deadline = time.time() + 110.0
    
    # 2. Local Search: 2-opt (First Improvement)
    # We restart the scan after each improvement to ensure thorough exploration of the new neighborhood.
    
    while time.time() < deadline:
        improved = False
        
        for i in range(n - 1):
            if time.time() > deadline:
                break
                
            u = p[i]
            a_i = a[i, :]
            b_u_p = b[u, p]  # Vector: b[u, p[j]] for all j
            
            for k in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                v = p[k]
                b_v_p = b[v, p]  # Vector: b[v, p[j]] for all j
                
                # Calculate delta
                # D = dot(a_i - a_k, b_v_p - b_u_p)
                # Subtract terms for j=i and j=k
                
                diff_a = a_i - a[k, :]
                diff_b = b_v_p - b_u_p
                
                full_sum = np.dot(diff_a, diff_b)
                
                # Term for j=i:
                # p[i] = u
                # diff_a[i] = a_i[i] - a[k, i]
                # diff_b[i] = b[v, u] - b[u, u]
                term_i = (a_i[i] - a[k, i]) * (b[v, u] - b[u, u])
                
                # Term for j=k:
                # p[k] = v
                # diff_a[k] = a_i[k] - a[k, k]
                # diff_b[k] = b[v, v] - b[u, v]
                term_k = (a_i[k] - a[k, k]) * (b[v, v] - b[u, v])
                
                delta = full_sum - term_i - term_k
                
                if delta < -1e-9:
                    # Improvement found
                    p[i], p[k] = p[k], p[i]
                    cost += delta
                    improved = True
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p.copy()
                    
                    # Restart i loop to re-evaluate with new permutation
                    break
            
            if improved:
                break
                
        if not improved:
            break
            
    return best_p.tolist()

# EVOLVE-BLOCK-END
