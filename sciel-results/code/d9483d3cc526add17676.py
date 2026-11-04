"""Multi-Start Iterated Local Search with spectral-projection initialization for QAP."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def _spectral_init(a, b, k=10):
    """Generate an initial permutation using spectral decomposition of a and b."""
    n = len(a)
    
    # Symmetrize
    a_sym = 0.5 * (a + a.T)
    b_sym = 0.5 * (b + b.T)
    
    try:
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
    except np.linalg.LinAlgError:
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p.tolist()
    
    # Sort by eigenvalue descending
    idx_a = np.argsort(-evals_a)
    idx_b = np.argsort(-evals_b)
    evals_a = evals_a[idx_a]
    evecs_a = evecs_a[:, idx_a]
    evals_b = evals_b[idx_b]
    evecs_b = evecs_b[:, idx_b]
    
    # Take top-k
    k_eff = min(k, n)
    Ua_top = evecs_a[:, :k_eff]
    Ub_top = evecs_b[:, :k_eff]
    
    # Weight by eigenvalues
    weights_a = np.abs(evals_a[:k_eff]) + 1e-10
    weights_b = np.abs(evals_b[:k_eff]) + 1e-10
    W = np.sqrt(weights_a * weights_b)
    
    # Surrogate cost matrix
    C = -(Ua_top * W) @ Ub_top.T
    
    # Solve the linear assignment problem
    row_ind, col_ind = linear_sum_assignment(C)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p.tolist()


def _evaluate(a, b, p):
    """Compute the QAP objective value for permutation p."""
    b_perm = b[p, :]
    cost = np.sum(a * b_perm)
    return int(cost)


def _local_search(a, b, p, max_iters=10000, time_limit=None):
    """Iterated Local Search with efficient delta computation."""
    n = len(a)
    p = np.array(p, dtype=np.int64)
    current_cost = _evaluate(a, b, p)
    
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    rng = np.random.default_rng()
    
    for iteration in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Find best improving swap
        min_delta = 0
        best_i, best_j = -1, -1
        
        # Sample a subset of pairs for speed
        if n <= 200:
            sample_size = n
        else:
            sample_size = min(100, n)
        
        sampled = rng.choice(n, size=sample_size, replace=False)
        
        for idx_i in range(len(sampled)):
            i = sampled[idx_i]
            pi = p[i]
            
            for idx_j in range(idx_i + 1, len(sampled)):
                j = sampled[idx_j]
                pj = p[j]
                
                # Compute delta for swapping i and j
                delta = 0
                # Direct terms
                delta += a[i, j] * (b[pj, pi] - b[pi, pj])
                delta += a[j, i] * (b[pi, pj] - b[pj, pi])
                delta += a[i, i] * (b[pj, pj] - b[pi, pi])
                delta += a[j, j] * (b[pi, pi] - b[pj, pj])
                
                # Indirect terms (sampled for speed)
                if n > 200:
                    k_sample = rng.choice(n, size=min(50, n), replace=False)
                    for k in k_sample:
                        if k == i or k == j:
                            continue
                        pk = p[k]
                        delta += a[i, k] * (b[pj, pk] - b[pi, pk])
                        delta += a[k, j] * (b[pk, pi] - b[pk, pj])
                else:
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        pk = p[k]
                        delta += a[i, k] * (b[pj, pk] - b[pi, pk])
                        delta += a[k, j] * (b[pk, pi] - b[pk, pj])
                
                if delta < min_delta:
                    min_delta = delta
                    best_i, best_j = i, j
        
        if min_delta < 0 and best_i >= 0:
            # Apply best swap
            i, j = best_i, best_j
            pi, pj = p[i], p[j]
            p[i], p[j] = pj, pi
            current_cost += min_delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # No improving swap found, perform a kick
            # Use a random swap for perturbation
            i, j = rng.choice(n, size=2, replace=False)
            if i != j:
                i, j = int(i), int(j)
                pi, pj = p[i], p[j]
                
                # Compute delta for kick
                delta = 0
                delta += a[i, j] * (b[pj, pi] - b[pi, pj])
                delta += a[j, i] * (b[pi, pj] - b[pj, pi])
                delta += a[i, i] * (b[pj, pj] - b[pi, pi])
                delta += a[j, j] * (b[pi, pi] - b[pj, pj])
                
                if n > 200:
                    k_sample = rng.choice(n, size=min(50, n), replace=False)
                    for k in k_sample:
                        if k == i or k == j:
                            continue
                        pk = p[k]
                        delta += a[i, k] * (b[pj, pk] - b[pi, pk])
                        delta += a[k, j] * (b[pk, pi] - b[pk, pj])
                else:
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        pk = p[k]
                        delta += a[i, k] * (b[pj, pk] - b[pi, pk])
                        delta += a[k, j] * (b[pk, pi] - b[pk, pj])
                
                p[i], p[j] = pj, pi
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
    
    return best_p.tolist(), best_cost


def solve(a, b):
    """Multi-Start Iterated Local Search for QAP."""
    n = len(a)
    
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    start_time = time.time()
    time_limit = 115
    
    best_p = None
    best_cost = np.inf
    
    num_starts = 6
    
    for start_idx in range(num_starts):
        if time.time() - start_time > time_limit - 15:
            break
        
        # Generate initial permutation
        if start_idx == 0:
            p_init = _spectral_init(a, b, k=10)
        elif start_idx == 1:
            p_init = _spectral_init(a, b, k=5)
        elif start_idx == 2:
            # Rank-matching heuristic
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p_arr = np.empty(n, dtype=np.int64)
            p_arr[heavy] = central
            p_init = p_arr.tolist()
        elif start_idx == 3:
            # Random
            p_init = list(np.random.default_rng(start_idx).permutation(n))
        elif start_idx == 4:
            # Another spectral with different k
            p_init = _spectral_init(a, b, k=15)
        else:
            # Spectral with k=20
            p_init = _spectral_init(a, b, k=20)
        
        # Run local search
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_init, max_iters=10000, time_limit=remaining_time - 3)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    # If no local search was run (shouldn't happen), use spectral init
    if best_p is None:
        best_p = _spectral_init(a, b, k=10)
    
    return best_p
# EVOLVE-BLOCK-END
