"""Multi-Start Iterated Local Search with spectral-projection initialization for QAP."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def _spectral_init(a, b, k=10, rng=None):
    """Generate an initial permutation using spectral decomposition of a and b."""
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(a)
    
    # Symmetrize
    a_sym = 0.5 * (a + a.T)
    b_sym = 0.5 * (b + b.T)
    
    # Center
    a_c = a_sym - a_sym.mean(axis=1, keepdims=True)
    a_c = a_c - a_c.mean(axis=0, keepdims=True)
    b_c = b_sym - b_sym.mean(axis=1, keepdims=True)
    b_c = b_c - b_c.mean(axis=0, keepdims=True)
    
    try:
        evals_a, evecs_a = np.linalg.eigh(a_c)
        evals_b, evecs_b = np.linalg.eigh(b_c)
    except np.linalg.LinAlgError:
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p.tolist()
    
    idx_a = np.argsort(-evals_a)
    idx_b = np.argsort(-evals_b)
    evals_a = evals_a[idx_a]
    evecs_a = evecs_a[:, idx_a]
    evals_b = evals_b[idx_b]
    evecs_b = evecs_b[:, idx_b]
    
    k_eff = min(k, n)
    Ua = evecs_a[:, :k_eff]
    Ub = evecs_b[:, :k_eff]
    
    wa = np.abs(evals_a[:k_eff]) + 1e-10
    wb = np.abs(evals_b[:k_eff]) + 1e-10
    W = np.sqrt(wa * wb)
    
    C = -(Ua * W) @ Ub.T
    
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p.tolist()


def _evaluate(a, b, p):
    """Compute the QAP objective value for permutation p."""
    b_perm = b[p, :]
    return int(np.sum(a * b_perm))


def _delta_swap(a, b, p, i, j):
    """Compute the cost delta of swapping positions i and j."""
    pi = p[i]
    pj = p[j]
    
    # Direct terms
    delta = a[i, j] * (b[pj, pi] - b[pi, pj])
    delta += a[j, i] * (b[pi, pj] - b[pj, pi])
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Indirect terms
    # For all k != i, j:
    # delta += a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
    # delta += a[k,j] * (b[p[k], pi] - b[p[k], pj])
    
    # Vectorized:
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    ks = np.where(mask)[0]
    
    if len(ks) > 0:
        pk = p[ks]
        delta += np.sum(a[i, ks] * (b[pj, pk] - b[pi, pk]))
        delta += np.sum(a[ks, j] * (b[pk, pi] - b[pk, pj]))
    
    return delta


def _local_search(a, b, p, max_iters=10000, time_limit=None):
    """Iterated Local Search with sampled swaps and reversal kicks."""
    n = len(a)
    p = np.array(p, dtype=np.int64)
    current_cost = _evaluate(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    rng = np.random.default_rng()
    
    # Precompute row sums for quick delta estimation
    b_perm = b[p, :]
    row_sum = np.sum(a * b_perm, axis=1)
    
    def _apply_swap(i, j):
        nonlocal current_cost, p
        d = _delta_swap(a, b, p, i, j)
        p[i], p[j] = p[j], p[i]
        current_cost += d
        # Update row_sum
        b_perm = b[p, :]
        row_sum[i] = np.sum(a[i, :] * b[p, i, :]) if False else np.sum(a[i, :] * b[p[i], :])
        row_sum[j] = np.sum(a[j, :] * b[p[j], :])
        return d
    
    for iteration in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Sample random pairs to find improving swaps
        found_improvement = False
        n_samples = 200
        for _ in range(n_samples):
            i, j = rng.choice(n, size=2, replace=False)
            i, j = int(i), int(j)
            if i == j:
                continue
            d = _delta_swap(a, b, p, i, j)
            if d < 0:
                _apply_swap(i, j)
                found_improvement = True
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                break
        
        if not found_improvement:
            # Kick: random reversal
            i, j = sorted(rng.choice(n, size=2, replace=False))
            i, j = int(i), int(j)
            if j - i >= 2:
                # Reverse segment
                old_cost = current_cost
                p[i:j+1] = p[i:j+1][::-1]
                new_cost = _evaluate(a, b, p)
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            else:
                # Single swap kick
                i2, j2 = rng.choice(n, size=2, replace=False)
                i2, j2 = int(i2), int(j2)
                if i2 != j2:
                    _apply_swap(i2, j2)
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
    
    num_starts = 8
    
    for start_idx in range(num_starts):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 2
        if remaining < 3:
            break
        
        # Generate diverse initial permutations
        if start_idx == 0:
            p_init = _spectral_init(a, b, k=10)
        elif start_idx == 1:
            p_init = _spectral_init(a, b, k=5)
        elif start_idx == 2:
            p_init = _spectral_init(a, b, k=15)
        elif start_idx == 3:
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p_arr = np.empty(n, dtype=np.int64)
            p_arr[heavy] = central
            p_init = p_arr.tolist()
        elif start_idx == 4:
            heavy = np.argsort(a.sum(axis=1), kind="stable")
            central = np.argsort(-b.sum(axis=1), kind="stable")
            p_arr = np.empty(n, dtype=np.int64)
            p_arr[heavy] = central
            p_init = p_arr.tolist()
        elif start_idx == 5:
            p_init = _spectral_init(-a, b, k=10)
        elif start_idx == 6:
            p_init = _spectral_init(a, -b, k=10)
        else:
            p_init = list(np.random.default_rng(start_idx * 12345).permutation(n))
        
        # Allocate time based on start index
        time_frac = max(0.1, 1.0 - start_idx * 0.1)
        ls_time = remaining * time_frac
        
        p_ls, cost_ls = _local_search(a, b, p_init, max_iters=10000, time_limit=ls_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    if best_p is None:
        best_p = _spectral_init(a, b, k=10)
    
    return best_p
# EVOLVE-BLOCK-END
