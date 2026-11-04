"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    b_p = b[np.ix_(p, p)]
    return np.sum(a * b_p)

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in permutation p.
    p[i] is the location of facility i.
    """
    li = p[i]
    lj = p[j]
    
    # Vectorized dot products for the sums over k
    a_i = a[i, :]
    a_j = a[j, :]
    a_col_i = a[:, i]
    a_col_j = a[:, j]
    
    b_row_li = b[li, :]
    b_row_lj = b[lj, :]
    b_col_li = b[:, li]
    b_col_lj = b[:, lj]
    
    delta_terms = (
        np.dot(a_i, b_row_lj[p] - b_row_li[p]) +
        np.dot(a_col_i, b_col_lj[p] - b_col_li[p]) +
        np.dot(a_j, b_row_li[p] - b_row_lj[p]) +
        np.dot(a_col_j, b_col_li[p] - b_col_lj[p])
    )
    
    # Subtract the k=i and k=j terms from the dot products
    subtract_i_i_1 = a[i, i] * (b[lj, li] - b[li, li])
    subtract_i_j_1 = a[i, j] * (b[lj, lj] - b[li, lj])
    
    subtract_i_i_2 = a[i, i] * (b[li, lj] - b[li, li])
    subtract_j_i_2 = a[j, i] * (b[lj, lj] - b[lj, li])
    
    subtract_j_i_3 = a[j, i] * (b[li, li] - b[lj, li])
    subtract_j_j_3 = a[j, j] * (b[li, lj] - b[lj, lj])
    
    subtract_i_j_4 = a[i, j] * (b[li, li] - b[li, lj])
    subtract_j_j_4 = a[j, j] * (b[lj, li] - b[lj, lj])
    
    # Add the correct diagonal terms for the swap
    add_diag_1 = a[i, i] * (b[lj, lj] - b[li, li])
    add_diag_2 = a[i, j] * (b[lj, li] - b[li, lj])
    add_diag_3 = a[j, i] * (b[li, lj] - b[lj, li])
    add_diag_4 = a[j, j] * (b[li, li] - b[lj, lj])
    
    delta = delta_terms - (subtract_i_i_1 + subtract_i_j_1 + subtract_i_i_2 + subtract_j_i_2 + 
                           subtract_j_i_3 + subtract_j_j_3 + subtract_i_j_4 + subtract_j_j_4) + \
            (add_diag_1 + add_diag_2 + add_diag_3 + add_diag_4)
            
    return delta

def solve(a, b):
    n = len(a)
    # Rank-matching start
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    max_time = 115.0
    
    # Simulated Annealing Parameters
    T = 1000.0
    alpha = 0.999
    min_T = 1.0
    
    while (time.time() - start_time) < max_time and T > min_T:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        delta = _swap_delta(a, b, p, i, j)
        
        if delta < 0:
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            if random.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
        
        T *= alpha
        
    # Final polish with greedy hill climbing
    p = best_p
    current_cost = best_cost
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        i_list, j_list = np.triu_indices(n, k=1)
        perm = np.random.permutation(len(i_list))
        i_list = i_list[perm]
        j_list = j_list[perm]
        
        for idx in range(len(i_list)):
            if (time.time() - start_time) > max_time:
                break
            i = i_list[idx]
            j = j_list[idx]
            
            delta = _swap_delta(a, b, p, i, j)
            if delta < 0:
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                improved = True
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    
    return best_p.tolist()
# EVOLVE-BLOCK-END
