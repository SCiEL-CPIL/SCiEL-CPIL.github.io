import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    
    # 1. Hungarian Algorithm Initialization
    W = a @ b.T
    row_ind, col_ind = linear_sum_assignment(W)
    p = np.full(n, -1, dtype=np.int64)
    p[row_ind] = col_ind
    
    # Calculate initial cost
    B_perm = b[:, p]
    S = a @ B_perm
    current_cost = int(np.sum(S[np.arange(n), p]))
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute row and column sums of a for faster delta computation
    a_row_sums = a.sum(axis=1)  # a_row_sums[i] = sum_j a[i,j]
    a_col_sums = a.sum(axis=0)  # a_col_sums[j] = sum_i a[i,j]
    
    # Helper function to compute delta for swapping positions i and k
    def compute_delta(i, k, p):
        u, v = p[i], p[k]
        # For swap of facilities at positions i and k (locations u and v)
        # delta = sum over all j of contributions
        # Using precomputed sums where possible
        
        # Contribution from rows i and k interacting with all columns
        # When we swap, facility i goes to location v, facility k goes to location u
        
        # Sum of a[i,j] * b[v, p[j]] for all j - sum of a[i,j] * b[u, p[j]] for all j
        # = sum_j a[i,j] * (b[v, p[j]] - b[u, p[j]])
        
        # Similarly for facility k
        # = sum_j a[k,j] * (b[u, p[j]] - b[v, p[j]])
        
        # And for the column interactions:
        # sum_j a[j,i] * (b[p[j], v] - b[p[j], u]) + sum_j a[j,k] * (b[p[j], u] - b[p[j], v])
        
        # Let's compute this efficiently
        diff_b_vu = b[v, p] - b[u, p]  # shape (n,)
        diff_b_pvu = b[p, v] - b[p, u]  # shape (n,)
        
        # Row contributions
        row_contrib = np.dot(a[i, :], diff_b_vu) - np.dot(a[k, :], diff_b_vu)
        
        # Column contributions
        col_contrib = np.dot(a[:, i], diff_b_pvu) - np.dot(a[:, k], diff_b_pvu)
        
        # Now we need to correct for the fact that when j=i or j=k, the terms change
        # because p[i] and p[k] themselves change
        
        # When j=i: p[i] changes from u to v
        # Old term: a[i,i]*b[u,u] + a[i,i]*b[u,u] (from row i and col i, both using p[i]=u)
        # New term: a[i,i]*b[v,v] + a[i,i]*b[v,v] (both using p[i]=v)
        # But we already counted a[i,i]*b[v,p[i]] and a[i,i]*b[p[i],v] in the sums above
        # with p[i]=u (old value), so we need to correct
        
        # Actually, let me reconsider. The formula sum_j a[i,j]*b[v,p[j]] uses the CURRENT p,
        # but after swap p[i] will be v, not u. So for j=i, we're using b[v, p[i]] = b[v, u]
        # but it should be b[v, v]. Similarly for other terms.
        
        # Let me use a cleaner approach: compute delta directly
        
        # The change in cost when swapping assignments at positions i and k:
        # Old: a[i,i]*b[u,u] + a[i,k]*b[u,v] + a[k,i]*b[v,u] + a[k,k]*b[v,v]
        #       + sum_{j!=i,k} (a[i,j]*b[u,p[j]] + a[j,i]*b[p[j],u] + a[k,j]*b[v,p[j]] + a[j,k]*b[p[j],v])
        # New: a[i,i]*b[v,v] + a[i,k]*b[v,u] + a[k,i]*b[u,v] + a[k,k]*b[u,u]
        #       + sum_{j!=i,k} (a[i,j]*b[v,p[j]] + a[j,i]*b[p[j],v] + a[k,j]*b[u,p[j]] + a[j,k]*b[p[j],u])
        
        # Delta = New - Old
        
        # Diagonal terms (i,i), (k,k):
        diag_ii = a[i,i] * (b[v,v] - b[u,u])
        diag_kk = a[k,k] * (b[u,u] - b[v,v])
        
        # Cross terms (i,k) and (k,i):
        cross_ik = a[i,k] * (b[v,u] - b[u,v])
        cross_ki = a[k,i] * (b[u,v] - b[v,u])
        
        # Off-diagonal terms: sum over j != i,k
        # For each j != i,k:
        # (a[i,j] - a[k,j]) * (b[v,p[j]] - b[u,p[j]]) + (a[j,i] - a[j,k]) * (b[p[j],v] - b[p[j],u])
        
        # Compute for all j, then subtract j=i and j=k
        diff_a_row = a[i, :] - a[k, :]
        diff_a_col = a[:, i] - a[:, k]
        
        # b[v, p[j]] - b[u, p[j]] for all j
        bp_vu = b[v, p] - b[u, p]
        # b[p[j], v] - b[p[j], u] for all j  
        bp_pvu = b[p, v] - b[p, u]
        
        all_terms = np.dot(diff_a_row, bp_vu) + np.dot(diff_a_col, bp_pvu)
        
        # Subtract j=i term: p[i]=u
        # (a[i,i]-a[k,i])*(b[v,u]-b[u,u]) + (a[i,i]-a[i,k])*(b[u,v]-b[u,u])
        term_i = (a[i,i]-a[k,i])*(b[v,u]-b[u,u]) + (a[i,i]-a[i,k])*(b[u,v]-b[u,u])
        
        # Subtract j=k term: p[k]=v
        # (a[i,k]-a[k,k])*(b[v,v]-b[u,v]) + (a[k,i]-a[k,k])*(b[v,v]-b[v,u])
        term_k = (a[i,k]-a[k,k])*(b[v,v]-b[u,v]) + (a[k,i]-a[k,k])*(b[v,v]-b[v,u])
        
        off_diag = all_terms - term_i - term_k
        
        delta = diag_ii + diag_kk + cross_ik + cross_ki + off_diag
        return delta
    
    # 2. Simulated Annealing
    # Estimate initial temperature
    deltas = []
    for _ in range(100):
        i, k = np.random.choice(n, 2, replace=False)
        delta = compute_delta(i, k, p)
        deltas.append(abs(delta))
    
    T = np.mean(deltas) * 3.0 if deltas else 10000.0
    T_min = 1.0
    alpha = 0.9995
    
    start_time = time.time()
    time_limit = 90.0
    
    while time.time() - start_time < time_limit:
        i, k = np.random.choice(n, 2, replace=False)
        delta = compute_delta(i, k, p)
        
        if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
            p[i], p[k] = p[k], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    # 3. 2-opt local search on best solution (first pass)
    p = best_p.copy()
    current_cost = best_cost
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for k in range(i+1, n):
                delta = compute_delta(i, k, p)
                if delta < 0:
                    p[i], p[k] = p[k], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    # 4. Second round of simulated annealing from improved solution
    p = best_p.copy()
    current_cost = best_cost
    
    # Re-estimate temperature
    deltas = []
    for _ in range(100):
        i, k = np.random.choice(n, 2, replace=False)
        delta = compute_delta(i, k, p)
        deltas.append(abs(delta))
    
    T = np.mean(deltas) * 2.0 if deltas else 1000.0
    T_min = 1.0
    alpha = 0.9997
    
    start_time = time.time()
    time_limit = 25.0
    
    while time.time() - start_time < time_limit:
        i, k = np.random.choice(n, 2, replace=False)
        delta = compute_delta(i, k, p)
        
        if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
            p[i], p[k] = p[k], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    # 5. Final 2-opt local search
    p = best_p.copy()
    current_cost = best_cost
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for k in range(i+1, n):
                delta = compute_delta(i, k, p)
                if delta < 0:
                    p[i], p[k] = p[k], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    return best_p.tolist()
