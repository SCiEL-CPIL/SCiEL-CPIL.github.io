"""
Iterated Local Search with Benders-style surrogate for QAPLIB wil100.
Uses a linearized surrogate cost to guide perturbations and acceptance
decisions, while the exact quadratic cost drives the local search.
Multi-start with adaptive perturbation strength.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _surrogate_cost(a, b, p):
    """
    Linearized surrogate: sum_i (a_row_sum[i]) * (b_col_sum[p[i]])
    This is a relaxation that ignores quadratic interactions.
    """
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=0)
    return int(np.sum(a_row_sums * b_col_sums[p]))

def _greedy_initial(a, b):
    """
    Greedy initial assignment: match high-flow facilities to central locations.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _spectral_initial(a, b):
    """
    Spectral initialization using eigenvectors of the matrices.
    """
    n = len(a)
    # Use the leading eigenvector of A as facility ordering
    # and leading eigenvector of B as location ordering
    try:
        eigvals_a, eigvecs_a = np.linalg.eigh(a)
        eigvals_b, eigvecs_b = np.linalg.eigh(b)
        
        # Take the eigenvector corresponding to the largest eigenvalue
        # (for symmetric matrices, this gives the principal component)
        idx_a = np.argmax(eigvals_a)
        idx_b = np.argmax(eigvals_b)
        
        v_a = eigvecs_a[:, idx_a]
        v_b = eigvecs_b[:, idx_b]
        
        # Sort facilities by descending spectral component
        facility_order = np.argsort(-v_a, kind="stable")
        # Sort locations by ascending spectral component (to match high flow to low distance)
        location_order = np.argsort(v_b, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[facility_order] = location_order
        return p
    except Exception:
        return _greedy_initial(a, b)

def _random_initial(n):
    """Generate a random permutation."""
    return np.random.permutation(n).astype(np.int64)

def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j in permutation p.
    """
    pi = p[i]
    pj = p[j]
    
    # Term 1: column differences
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)
    
    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j
    
    # Term 2: row differences
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)
    
    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j
    
    # Term 3: cross term
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return int(T1 + T2 + T3)

def _local_search(a, b, p, current_cost, time_limit=5.0):
    """
    Steepest-descent local search: repeatedly find the best improving swap.
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    
    while time.time() - start_time < time_limit:
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Compute all swap deltas
        for i in range(n):
            for j in range(i + 1, n):
                delta = _compute_swap_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            break  # Local minimum
        
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += best_delta
    
    return p, cost

def _perturb(a, b, p, k, surrogate_guided=True):
    """
    Perturb the permutation by k random swaps.
    If surrogate_guided, prefer swaps that improve the surrogate cost.
    """
    n = len(p)
    p = p.copy()
    
    if surrogate_guided:
        a_row_sums = a.sum(axis=1)
        b_col_sums = b.sum(axis=0)
        curr_surrogate = int(np.sum(a_row_sums * b_col_sums[p]))
        
        for _ in range(k):
            # Generate a pool of candidate swaps
            candidates = []
            for _ in range(min(20, n * (n-1) // 2)):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    # Estimate surrogate delta
                    si = a_row_sums[i] * b_col_sums[p[j]] + a_row_sums[j] * b_col_sums[p[i]]
                    sj = a_row_sums[i] * b_col_sums[p[i]] + a_row_sums[j] * b_col_sums[p[j]]
                    surr_delta = si - sj
                    candidates.append((surr_delta, i, j))
            
            # Sort by surrogate improvement and pick from the best ones
            candidates.sort(key=lambda x: x[0])
            # Pick from the top half with some randomness
            top_half = candidates[:len(candidates)//2] if len(candidates) > 1 else candidates
            _, i, j = top_half[np.random.randint(len(top_half))]
            
            p[i], p[j] = p[j], p[i]
    else:
        for _ in range(k):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
    
    return p

def solve(a, b):
    n = len(a)
    total_deadline = 145.0
    start_time = time.time()
    
    # Phase 1: Generate multiple initial solutions
    p_greedy = _greedy_initial(a, b)
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    p_spectral = _spectral_initial(a, b)
    cost_spectral = _qap_cost(a, b, p_spectral)
    
    p_random1 = _random_initial(n)
    cost_random1 = _qap_cost(a, b, p_random1)
    
    p_random2 = _random_initial(n)
    cost_random2 = _qap_cost(a, b, p_random2)
    
    # Select the best initial solution
    candidates = [(p_greedy, cost_greedy), (p_spectral, cost_spectral), 
                  (p_random1, cost_random1), (p_random2, cost_random2)]
    candidates.sort(key=lambda x: x[1])
    best_p, best_cost = candidates[0]
    best_p = best_p.copy()
    best_surrogate = _surrogate_cost(a, b, best_p)
    
    # Phase 2: Iterated Local Search with Benders-style guidance
    # Use surrogate to guide perturbations and acceptance
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=0)
    
    # Adaptive perturbation strength
    base_perturb = 8
    max_perturb = 20
    
    # Main ILS loop
    while time.time() - start_time < total_deadline - 10.0:
        # Local search to convergence
        polish_time = min(5.0, max(0.5, total_deadline - 10.0 - (time.time() - start_time)))
        curr_p, curr_cost = _local_search(a, b, best_p, best_cost, time_limit=polish_time)
        
        if curr_cost < best_cost:
            best_cost = curr_cost
            best_p = curr_p.copy()
            best_surrogate = _surrogate_cost(a, b, best_p)
        
        # Check time
        if time.time() - start_time > total_deadline - 10.0:
            break
        
        # Perturbation: use surrogate-guided perturbation
        # Adaptive strength: increase if stuck, decrease if improving
        elapsed = time.time() - start_time
        progress_ratio = elapsed / total_deadline
        
        if progress_ratio < 0.3:
            k = base_perturb
        elif progress_ratio < 0.7:
            k = base_perturb + 5
        else:
            k = max_perturb
        
        # Surrogate-guided perturbation
        perturbed_p = _perturb(a, b, best_p, k, surrogate_guided=True)
        perturbed_cost = _qap_cost(a, b, perturbed_p)
        
        # Acceptance criterion: 
        # 1. If perturbed is better in exact cost, accept
        # 2. If perturbed is better in surrogate, accept with high probability
        # 3. Otherwise, accept with temperature-based probability
        if perturbed_cost < best_cost:
            best_p = perturbed_p
            best_cost = perturbed_cost
            best_surrogate = _surrogate_cost(a, b, best_p)
        else:
            perturbed_surrogate = _surrogate_cost(a, b, perturbed_p)
            surr_delta = perturbed_surrogate - best_surrogate
            cost_delta = perturbed_cost - best_cost
            
            # Temperature decays over time
            T = 100.0 * (1.0 - progress_ratio) + 10.0
            
            # If surrogate improves, be more accepting
            if surr_delta < 0:
                accept_prob = np.exp(-cost_delta / (T * 0.5))
            else:
                accept_prob = np.exp(-cost_delta / T)
            
            if np.random.rand() < accept_prob:
                best_p = perturbed_p
                best_cost = perturbed_cost
                best_surrogate = perturbed_surrogate
    
    # Phase 3: Final local search polish
    elapsed = time.time() - start_time
    polish_time = min(8.0, max(0.5, total_deadline - elapsed - 1.0))
    if polish_time > 0.5:
        best_p, best_cost = _local_search(a, b, best_p, best_cost, time_limit=polish_time)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
