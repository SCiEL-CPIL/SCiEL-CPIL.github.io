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
    # Check time every 10 swaps to reduce overhead
    swap_count = 0
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if swap_count % 10 == 0 and time.time() - start_time > time_limit:
                return p, current_cost
            for j in range(i + 1, n):
                delta = _fast_delta(a, D, i, j)
                if delta < 0:
                    # Perform swap in p
                    p[i], p[j] = p[j], p[i]
                    
                    # Update D efficiently
                    temp_row = D[i].copy()
                    D[i] = D[j]
                    D[j] = temp_row
                    
                    temp_col = D[:, i].copy()
                    D[:, i] = D[:, j]
                    D[:, j] = temp_col
                    
                    current_cost += delta
                    improved = True
                    swap_count += 1
                    # Continue searching
                    
    return p, current_cost

def _eigen_init(a, b):
    """
    Initialize using principal eigenvectors of A and B.
    """
    n = len(a)
    try:
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        
        # Use eigh for symmetric matrices (faster and real)
        eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
        
        # Principal eigenvector (largest eigenvalue, last column)
        v_a = eigvecs_a[:, -1]
        v_b = eigvecs_b[:, -1]
        
        idx_a = np.argsort(v_a)
        idx_b = np.argsort(v_b)
        
        # Check correlation to determine if we need to reverse
        # If v_a and v_b are negatively correlated, high v_a should map to low v_b
        corr = np.corrcoef(v_a, v_b)[0, 1]
        if corr < 0:
            idx_b = idx_b[::-1]
            
        p = np.zeros(n, dtype=np.int64)
        for k in range(n):
            p[idx_a[k]] = idx_b[k]
            
        # Validate
        if len(set(p.tolist())) != n:
            return None
        return p
    except Exception:
        return None

def _perturb(p, k):
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
    
    # Try Eigen Init
    p_init = _eigen_init(a, b)
    if p_init is None:
        rng = random.Random(42)
        p_init = np.arange(n, dtype=np.int64)
        rng.shuffle(p_init.tolist())
        p_init = np.array(p_init, dtype=np.int64)
    
    # First local search on init
    p_ls, cost_ls = _local_search_2opt_fast(a, b, p_init, time_budget - 10, start_time)
    best_p = p_ls
    best_cost = cost_ls
    
    # Iterated Local Search with adaptive perturbation
    k_perturb = 5
    while time.time() - start_time < time_budget - 5:
        # Perturb
        p_pert = _perturb(best_p, k_perturb)
        
        # Local search
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break
        
        p_ls, cost_ls = _local_search_2opt_fast(a, b, p_pert, remaining - 1, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
            # Reduce perturbation if we found a better solution
            k_perturb = max(3, k_perturb - 1)
        else:
            # Increase perturbation if stuck
            k_perturb = min(20, k_perturb + 1)
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
