"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _qap_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    b_p = b[np.ix_(p, p)]
    return np.sum(a * b_p)

def _swap_cost_delta(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in permutation p.
    p[i] is the location of facility i.
    """
    li = p[i]
    lj = p[j]
    
    # Vectorized dot products for the sums over k
    # s1: sum_{k} a[i,k] * (b[lj, p[k]] - b[li, p[k]])
    # We subtract the k=i and k=j terms to avoid double counting with diagonal/cross terms
    # Actually, it's safer to compute the full sums and then adjust.
    
    # Let's use the explicit formula derived in the seed but cleaned up.
    # Delta = sum_{u,v in {i,j}} a[u,v](b[p'[u], p'[v]] - b[p[u], p[v]]) 
    #       + sum_{k not in {i,j}} [ a[i,k](b[lj, p[k]] - b[li, p[k]]) + a[k,i](b[p[k], lj] - b[p[k], li]) 
    #                               + a[j,k](b[li, p[k]] - b[lj, p[k]]) + a[k,j](b[p[k], li] - b[p[k], lj]) ]
    
    # The "sum over k not in {i,j}" can be computed as "sum over all k" minus "k=i" and "k=j" terms.
    
    # Sum over all k for term 1: a[i,k] * (b[lj, p[k]] - b[li, p[k]])
    term1_all = np.dot(a[i, :], b[lj, p] - b[li, p])
    # Subtract k=i: a[i,i] * (b[lj, li] - b[li, li])
    # Subtract k=j: a[i,j] * (b[lj, lj] - b[li, lj])
    term1 = term1_all - a[i, i] * (b[lj, li] - b[li, li]) - a[i, j] * (b[lj, lj] - b[li, lj])
    
    # Sum over all k for term 2: a[k,i] * (b[p[k], lj] - b[p[k], li])
    term2_all = np.dot(a[:, i], b[p, lj] - b[p, li])
    # Subtract k=i: a[i,i] * (b[li, lj] - b[li, li])
    # Subtract k=j: a[j,i] * (b[lj, lj] - b[lj, li])
    term2 = term2_all - a[i, i] * (b[li, lj] - b[li, li]) - a[j, i] * (b[lj, lj] - b[lj, li])
    
    # Sum over all k for term 3: a[j,k] * (b[li, p[k]] - b[lj, p[k]])
    term3_all = np.dot(a[j, :], b[li, p] - b[lj, p])
    # Subtract k=i: a[j,i] * (b[li, li] - b[lj, li])
    # Subtract k=j: a[j,j] * (b[li, lj] - b[lj, lj])
    term3 = term3_all - a[j, i] * (b[li, li] - b[lj, li]) - a[j, j] * (b[li, lj] - b[lj, lj])
    
    # Sum over all k for term 4: a[k,j] * (b[p[k], li] - b[p[k], lj])
    term4_all = np.dot(a[:, j], b[p, li] - b[p, lj])
    # Subtract k=i: a[i,j] * (b[li, li] - b[li, lj])
    # Subtract k=j: a[j,j] * (b[lj, li] - b[lj, lj])
    term4 = term4_all - a[i, j] * (b[li, li] - b[li, lj]) - a[j, j] * (b[lj, li] - b[lj, lj])
    
    # Diagonal and cross terms (u,v in {i,j})
    # Before: a[i,i]b[li,li] + a[i,j]b[li,lj] + a[j,i]b[lj,li] + a[j,j]b[lj,lj]
    # After:  a[i,i]b[lj,lj] + a[i,j]b[lj,li] + a[j,i]b[li,lj] + a[j,j]b[li,li]
    delta_diag = (a[i, i] * (b[lj, lj] - b[li, li]) + 
                  a[i, j] * (b[lj, li] - b[li, lj]) + 
                  a[j, i] * (b[li, lj] - b[lj, li]) + 
                  a[j, j] * (b[li, li] - b[lj, lj]))
    
    return delta_diag + term1 + term2 + term3 + term4

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Rank Matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    current_cost = _qap_cost(a, b, p)
    
    # 2. Simulated Annealing
    start_time = time.time()
    max_time = 115.0
    
    # Temperature parameters
    # Estimate typical delta magnitude
    # Random swap delta is roughly proportional to cost / n^2 * n = cost/n ?
    # Cost ~ 4e7. Delta ~ 1e5.
    t_start = 50000.0
    t_end = 1.0
    cooling_rate = 0.9999 # Adjust based on number of steps
    
    # We will do a fixed number of steps or until time runs out.
    # Let's do steps in batches to check time.
    steps_per_batch = 10000
    rng = np.random.default_rng(42)
    
    while (time.time() - start_time) < max_time:
        # Check time before batch
        if (time.time() - start_time) > max_time:
            break
            
        # Perform steps_per_batch swaps
        for _ in range(steps_per_batch):
            if (time.time() - start_time) > max_time:
                break
                
            # Pick random pair i, j
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
                
            delta = _swap_cost_delta(a, b, p, i, j)
            
            # Metropolis criterion
            if delta < 0:
                # Always accept improving moves
                p[i], p[j] = p[j], p[i]
                current_cost += delta
            else:
                # Accept worsening moves with probability exp(-delta / T)
                if t_start > t_end:
                    prob = math.exp(-delta / t_start)
                    if rng.random() < prob:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
            
            # Cool down
            t_start *= cooling_rate
            if t_start < t_end:
                t_start = t_end
    
    return p.tolist()
# EVOLVE-BLOCK-END
