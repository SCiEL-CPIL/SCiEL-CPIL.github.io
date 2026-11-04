"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))

def _delta_swap(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in the assignment p.
    Vectorized for speed.
    """
    pi = p[i]
    pj = p[j]
    n = len(p)
    
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    
    diff1 = b[pj, p[mask]] - b[pi, p[mask]]
    diff2 = b[p[mask], pj] - b[p[mask], pi]
    
    a_diff_row = a[i, mask] - a[j, mask]
    a_diff_col = a[mask, i] - a[mask, j]
    
    delta_k = np.sum(a_diff_row * diff1) + np.sum(a_diff_col * diff2)
    
    delta_self = (a[i, j] * (b[pj, pi] - b[pi, pj])
                  + a[j, i] * (b[pi, pj] - b[pj, pi])
                  + a[i, i] * (b[pj, pj] - b[pi, pi])
                  + a[j, j] * (b[pi, pi] - b[pj, pj]))
    
    return int(delta_k + delta_self)

def _delta_3opt(a, b, p, i, j, k):
    """
    Compute the change in cost if we cycle facilities i -> j -> k -> i.
    p[i] -> p[j], p[j] -> p[k], p[k] -> p[i].
    """
    pi = p[i]
    pj = p[j]
    pk = p[k]
    n = len(p)
    
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    mask[k] = False
    mask_idx = np.where(mask)[0]
    p_mask = p[mask_idx]
    
    b_pj_pm = b[pj, p_mask]
    b_pi_pm = b[pi, p_mask]
    b_pk_pm = b[pk, p_mask]
    
    b_pm_pj = b[p_mask, pj]
    b_pm_pi = b[p_mask, pi]
    b_pm_pk = b[p_mask, pk]
    
    d_i = b_pj_pm - b_pi_pm
    d_j = b_pk_pm - b_pj_pm
    d_k = b_pi_pm - b_pk_pm
    
    d_li = b_pm_pj - b_pm_pi
    d_lj = b_pm_pk - b_pm_pj
    d_lk = b_pm_pi - b_pm_pk
    
    a_i = a[i, mask_idx]
    a_j = a[j, mask_idx]
    a_k = a[k, mask_idx]
    
    a_li = a[mask_idx, i]
    a_lj = a[mask_idx, j]
    a_lk = a[mask_idx, k]
    
    delta_k = np.sum(a_i * d_i) + np.sum(a_j * d_j) + np.sum(a_k * d_k)
    delta_k += np.sum(a_li * d_li) + np.sum(a_lj * d_lj) + np.sum(a_lk * d_lk)
    
    delta_self = 0
    delta_self += a[i,j] * (b[pj,pk] - b[pi,pj])
    delta_self += a[j,i] * (b[pk,pj] - b[pj,pi])
    delta_self += a[i,k] * (b[pj,pi] - b[pi,pk])
    delta_self += a[k,i] * (b[pi,pj] - b[pk,pi])
    delta_self += a[j,k] * (b[pk,pi] - b[pj,pk])
    delta_self += a[k,j] * (b[pi,pk] - b[pk,pj])
    delta_self += a[i,i] * (b[pj,pj] - b[pi,pi])
    delta_self += a[j,j] * (b[pk,pk] - b[pj,pj])
    delta_self += a[k,k] * (b[pi,pi] - b[pk,pk])
    
    return int(delta_k + delta_self)

def _local_search(a, b, p, deadline, max_passes=100):
    """
    First-improvement local search with 2-opt swaps.
    """
    n = len(p)
    cost = _cost(a, b, p)

    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p, cost
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    break  # First improvement per i, then move to next i
        if not improved:
            break

    return p, cost

def _spectral_init(a, b, n, rng):
    """
    Spectral initialization using multi-dimensional embeddings and linear assignment.
    """
    try:
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
    except np.linalg.LinAlgError:
        return rng.permutation(n).astype(np.int64)

    k = min(10, n)
    idx_a = np.argsort(-np.abs(evals_a))[:k]
    fac_emb = evecs_a[:, idx_a]
    
    idx_b = np.argsort(-np.abs(evals_b))[:k]
    loc_emb = evecs_b[:, idx_b]

    # Normalize embeddings
    fac_norm = np.linalg.norm(fac_emb, axis=1, keepdims=True)
    fac_norm[fac_norm == 0] = 1
    fac_emb = fac_emb / fac_norm
    
    loc_norm = np.linalg.norm(loc_emb, axis=1, keepdims=True)
    loc_norm[loc_norm == 0] = 1
    loc_emb = loc_emb / loc_norm

    # Cost matrix for assignment: minimize sum fac_emb[i] . loc_emb[p[i]]
    # This is equivalent to maximizing sum fac_emb[i] . loc_emb[p[i]] if we flip sign
    # Or minimizing - (fac_emb @ loc_emb.T)
    cost_matrix = -fac_emb @ loc_emb.T
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind

    # Optional: slight perturbation to break symmetry if needed
    if rng.random() < 0.3:
        subset_size = rng.integers(5, 20)
        subset = rng.choice(n, size=subset_size, replace=False)
        p[subset] = p[subset][::-1]

    return p

def _quantum_perturb(p, rng, n, temperature=0.5):
    """
    Quantum-inspired perturbation: probabilistic multi-swap moves.
    """
    perturbed_p = p.copy()

    base_swaps = 2
    max_swaps = 8
    n_swaps = base_swaps + int(temperature * (max_swaps - base_swaps))
    n_swaps = max(1, min(n_swaps, n // 2))

    for _ in range(n_swaps):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i != j:
            perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]

    if rng.random() < 0.1 * temperature:
        block_len = rng.integers(3, min(10, n // 3))
        start = rng.integers(0, n - block_len + 1)
        perturbed_p[start:start + block_len] = perturbed_p[start:start + block_len][::-1]

    return perturbed_p

def _surrogate_perturb(a, b, p, current_cost, rng, n):
    """
    Surrogate-guided perturbation: identify facilities/locations with high
    contribution to cost and perturb those positions.
    """
    perturbed_p = p.copy()

    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    contrib = a_row_sums * b_row_sums[p]

    k = 3 + int(rng.integers(0, 4))
    top_idx = np.argsort(-contrib)[:k]

    for idx in top_idx:
        if rng.random() < 0.5:
            j = rng.integers(0, n)
            if j != idx:
                perturbed_p[idx], perturbed_p[j] = perturbed_p[j], perturbed_p[idx]
        else:
            low_idx = np.argsort(contrib)[:k]
            j = low_idx[rng.integers(0, k)]
            if j != idx:
                perturbed_p[idx], perturbed_p[j] = perturbed_p[j], perturbed_p[idx]

    return perturbed_p

def _tabu_search(a, b, p, deadline, tabu_tenure=15):
    """
    Tabu Search with 2-opt and 3-opt moves.
    """
    n = len(p)
    cost = _cost(a, b, p)
    best_cost = cost
    best_p = p.copy()
    
    # Tabu list: stores (i, j) and the iteration when it expires
    tabu_list = {}
    current_iter = 0
    
    while time.time() < deadline:
        current_iter += 1
        
        # Find best 2-opt move
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                
                # Check tabu status
                key = (i, j)
                is_tabu = False
                if key in tabu_list:
                    if tabu_list[key] > current_iter:
                        is_tabu = True
                else:
                    key_rev = (j, i)
                    if key_rev in tabu_list:
                        if tabu_list[key_rev] > current_iter:
                            is_tabu = True
                
                # Aspiration criterion
                if is_tabu and delta < 0 and (cost + delta) < best_cost:
                    is_tabu = False
                elif is_tabu:
                    continue
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i != -1:
            # Apply move
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += best_delta
            
            # Update tabu list
            tabu_list[(best_i, best_j)] = current_iter + tabu_tenure
            tabu_list[(best_j, best_i)] = current_iter + tabu_tenure
            
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        else:
            # No improving 2-opt move (or all are tabu).
            # Try 3-opt to escape.
            best_3_delta = 0
            best_3_i = -1
            best_3_j = -1
            best_3_k = -1
            
            # Sample some 3-cycles to keep it fast
            for _ in range(100):
                idx = rng.integers(0, n, size=3)
                if len(set(idx)) < 3:
                    continue
                i, j, k = idx
                delta = _delta_3opt(a, b, p, i, j, k)
                
                # Check tabu for pairs involved
                is_tabu = False
                for pair in [(i,j), (j,k), (k,i)]:
                    key = (pair[0], pair[1])
                    key_rev = (pair[1], pair[0])
                    if (key in tabu_list and tabu_list[key] > current_iter) or \
                       (key_rev in tabu_list and tabu_list[key_rev] > current_iter):
                        is_tabu = True
                        break
                
                if is_tabu:
                    if delta < 0 and (cost + delta) < best_cost:
                        is_tabu = False
                    else:
                        continue
                
                if delta < best_3_delta:
                    best_3_delta = delta
                    best_3_i = i
                    best_3_j = j
                    best_3_k = k
            
            if best_3_i != -1:
                # Apply 3-cycle: i -> j -> k -> i
                pi = p[best_3_i]
                pj = p[best_3_j]
                pk = p[best_3_k]
                p[best_3_i] = pj
                p[best_3_j] = pk
                p[best_3_k] = pi
                cost += best_3_delta
                
                # Update tabu for all pairs
                for pair in [(best_3_i, best_3_j), (best_3_j, best_3_k), (best_3_k, best_3_i)]:
                    tabu_list[pair] = current_iter + tabu_tenure
                    tabu_list[(pair[1], pair[0])] = current_iter + tabu_tenure
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
            else:
                # Stuck. Do a random perturbation to escape.
                # Use surrogate-guided or quantum perturbation
                if rng.random() < 0.5:
                    p = _surrogate_perturb(a, b, p, cost, rng, n)
                else:
                    p = _quantum_perturb(p, rng, n, temperature=0.5)
                
                # Recompute cost after perturbation
                cost = _cost(a, b, p)
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()

    return best_p, best_cost

def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin

    rng = np.random.default_rng(42)
    
    # Phase 1: Initialization
    p = _spectral_init(a, b, n, rng)
    p, cost = _local_search(a, b, p, deadline)
    
    # Phase 2: Tabu Search
    p, cost = _tabu_search(a, b, p, deadline, tabu_tenure=20)
    
    # Phase 3: Final Polish
    if time.time() < deadline - 2.0:
        p, cost = _local_search(a, b, p, deadline)
        
    return p.tolist()
# EVOLVE-BLOCK-END
