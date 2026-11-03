"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])

def _delta_swap(a, b, p, i, j):
    """
    Compute delta cost for swapping positions i and j in permutation p.
    Correct formula for QAP 2-opt.
    """
    n = len(p)
    u, v = p[i], p[j]
    # D is b[p, p], so D[i, k] = b[p[i], p[k]]
    # We need b[p[i], p[k]] and b[p[j], p[k]]
    
    # To avoid recomputing D, we pass D as well? 
    # For simplicity in this new version, we recompute the necessary vectors.
    # Since n=256, fetching rows is O(1) if we have b.
    
    # Vectorized delta calculation
    # Delta = sum_{k != i, j} (a[u, k] - a[v, k]) * (b[p[i], p[k]] - b[p[j], p[k]])
    #        + (a[u, v] - a[v, u]) * (b[p[i], p[j]] - b[p[j], p[i]])
    #        + (a[u, u] - a[v, v]) * (b[p[i], p[i]] - b[p[j], p[j]])
    
    # Let's compute S = sum_{k=0}^{n-1} (a[u, k] - a[v, k]) * (b[p[i], p[k]] - b[p[j], p[k]])
    # diff_a = a[u, :] - a[v, :]
    # diff_b = b[p[i], :] - b[p[j], :]  -- NO, this is not correct.
    # b[p[i], p[k]] is element (p[i], p[k]) of b.
    # So diff_b_k = b[p[i], p[k]] - b[p[j], p[k]].
    
    # We can compute this as:
    # row_i = b[p[i], :]  (length n)
    # row_j = b[p[j], :]  (length n)
    # But we need b[p[i], p[k]] which is row_i[p[k]].
    # So diff_b = row_i[p] - row_j[p].
    
    # This requires indexing.
    # Let's pass D = b[p, p] to avoid re-indexing b.
    # D[i, k] = b[p[i], p[k]].
    # So diff_b = D[i, :] - D[j, :].
    
    # We will modify _local_search to pass D.
    pass

def _local_search_2opt(a, b, p, time_limit, start_time):
    """
    Local search with 2-opt swaps.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = np.sum(a * D)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            u = p[i]
            for j in range(i + 1, n):
                v = p[j]
                
                # Compute Delta
                # S = sum_k (a[u, k] - a[v, k]) * (D[i, k] - D[j, k])
                diff_a = a[u, :] - a[v, :]
                diff_D = D[i, :] - D[j, :]
                S = np.dot(diff_a, diff_D)
                
                # Subtract k=i and k=j terms
                # Term i: (a[u, i] - a[v, i]) * (D[i, i] - D[j, i])
                # Note: D[i, i] = b[p[i], p[i]], D[j, i] = b[p[j], p[i]]
                # a[u, i] is flow between facility u and facility at pos i (which is u). So a[u, u].
                # a[v, i] is flow between facility v and facility at pos i (which is u). So a[v, u].
                # So term_i = (a[u, u] - a[v, u]) * (D[i, i] - D[j, i])
                
                # Term j: (a[u, j] - a[v, j]) * (D[i, j] - D[j, j])
                # a[u, j] is a[u, v]. a[v, j] is a[v, v].
                # So term_j = (a[u, v] - a[v, v]) * (D[i, j] - D[j, j])
                
                term_i = (a[u, u] - a[v, u]) * (D[i, i] - D[j, i])
                term_j = (a[u, v] - a[v, v]) * (D[i, j] - D[j, j])
                
                # Correction terms
                # (a[u, v] - a[v, u]) * (D[i, j] - D[j, i])
                # (a[u, u] - a[v, v]) * (D[i, i] - D[j, j])
                
                corr_1 = (a[u, v] - a[v, u]) * (D[i, j] - D[j, i])
                corr_2 = (a[u, u] - a[v, v]) * (D[i, i] - D[j, j])
                
                delta = S - term_i - term_j + corr_1 + corr_2
                
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    
                    # Update D
                    # Swap rows i and j in D
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    # Swap cols i and j in D
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                    
                    current_cost += delta
                    improved = True
                    
    return p, current_cost

def _perturb(p, k=5):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115
    
    best_p = None
    best_cost = np.inf
    
    rng = np.random.default_rng(42)
    
    # Multiple random restarts
    for restart in range(10):
        if time.time() - start_time > time_budget - 5:
            break
            
        p_init = rng.permutation(n)
        
        # Local search
        remaining = time_budget - (time.time() - start_time)
        if remaining < 1:
            break
            
        p_ls, cost_ls = _local_search_2opt(a, b, p_init, remaining - 1, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
