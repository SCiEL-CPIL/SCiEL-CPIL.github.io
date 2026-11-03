"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])

def _fast_delta(a, D, i, j):
    """
    Compute delta cost for swapping positions i and j in permutation p.
    D is the current distance matrix b[p, p].
    """
    # S1 = a[i] dot D[j] - a[i] dot D[i]
    # S2 = a[:, i] dot D[:, j] - a[:, i] dot D[:, i]
    # S3 = a[j] dot D[i] - a[j] dot D[j]
    # S4 = a[:, j] dot D[:, i] - a[:, j] dot D[:, j]
    
    s1 = np.dot(a[i], D[j]) - np.dot(a[i], D[i])
    s2 = np.dot(a[:, i], D[:, j]) - np.dot(a[:, i], D[:, i])
    s3 = np.dot(a[j], D[i]) - np.dot(a[j], D[j])
    s4 = np.dot(a[:, j], D[:, i]) - np.dot(a[:, j], D[:, j])
    
    total = s1 + s2 + s3 + s4
    
    # Subtract k=i contribution
    ki = a[i,i]*(D[j,i]-D[i,i]) + a[i,i]*(D[i,j]-D[i,i]) + a[j,i]*(D[i,i]-D[j,i]) + a[i,j]*(D[i,i]-D[i,j])
    
    # Subtract k=j contribution
    kj = a[i,j]*(D[j,j]-D[i,j]) + a[j,i]*(D[j,j]-D[j,i]) + a[j,j]*(D[i,j]-D[j,j]) + a[j,j]*(D[j,i]-D[j,j])
    
    total -= ki + kj
    
    # Add direct i,j interaction change
    direct = a[i,j]*D[j,i] + a[j,i]*D[i,j] - a[i,j]*D[i,j] - a[j,i]*D[j,i]
    
    total += direct
    
    return total

def _local_search_2opt_fast(a, b, p, time_limit, start_time):
    """
    Fast local search with 2-opt swaps.
    Returns (p, cost).
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
            for j in range(i + 1, n):
                delta = _fast_delta(a, D, i, j)
                if delta < 0:
                    # Perform swap in p
                    p[i], p[j] = p[j], p[i]
                    
                    # Update D
                    # Swap rows i and j
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    # Swap cols i and j
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                    
                    current_cost += delta
                    improved = True
                    # Continue searching, don't restart the pass immediately for speed
                    # but re-checking with new D is correct
                    
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
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Initial random permutation
    rng = random.Random(42)
    p_init = np.arange(n, dtype=np.int64)
    rng.shuffle(p_init.tolist())
    p_init = np.array(p_init, dtype=np.int64)
    
    # First local search on random init
    p_ls, cost_ls = _local_search_2opt_fast(a, b, p_init, time_budget - 5, start_time)
    best_p = p_ls
    best_cost = cost_ls
    
    # Iterated Local Search
    while time.time() - start_time < time_budget - 5:
        # Perturb
        p_pert = _perturb(best_p, k=5)
        
        # Local search
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break
        
        p_ls, cost_ls = _local_search_2opt_fast(a, b, p_pert, remaining - 1, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
