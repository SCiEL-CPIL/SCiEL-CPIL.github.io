"""
Spectral-initialized Iterated Local Search solver for QAPLIB wil100.
Strategy:
1. Spectral initialization: align eigenvectors of flow and distance matrices
   to create a geometrically informed initial permutation.
2. Iterated Local Search: repeatedly perform steepest-descent local search,
   then perturb the solution to escape local optima.
3. Final polish with a tighter local search pass.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_initial(a, b):
    """
    Create an initial permutation using spectral decomposition.
    Aligns the principal eigenvectors of the flow matrix with those of
    the distance matrix to create a geometrically informed assignment.
    """
    n = len(a)
    
    # Symmetrize for eigenvalue computation
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Get eigenvectors (sorted by eigenvalue descending)
    eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
    eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
    
    # Sort by eigenvalue descending
    idx_a = np.argsort(-eigvals_a)
    idx_b = np.argsort(-eigvals_b)
    eigvecs_a = eigvecs_a[:, idx_a]
    eigvecs_b = eigvecs_b[:, idx_b]
    
    # Project facilities onto the principal eigenvectors of flow
    # and locations onto the principal eigenvectors of distance
    # Then match by projection similarity
    
    # Use the top few eigenvectors for a more robust matching
    k = min(n, 10)
    proj_a = eigvecs_a[:, :k]
    proj_b = eigvecs_b[:, :k]
    
    # For each facility, compute its "signature" in the spectral space
    # and match to the location with the most similar signature
    # We want high-flow facilities to go to central (low-distance) locations
    
    # Actually, let's use a different approach:
    # The cost is sum a[i,j] * b[p[i], p[j]]
    # We want to minimize this. If a has high values where b has low values,
    # we want to align them.
    
    # Let's try: sort facilities by their projection on the top eigenvector of A
    # and sort locations by their projection on the top eigenvector of B
    # then assign in a way that minimizes the product.
    
    # Top eigenvector of A (facilities with high mutual flow)
    v_a_top = eigvecs_a[:, 0]
    # Top eigenvector of B (locations with high mutual distance)
    v_b_top = eigvecs_b[:, 0]
    
    # We want high-flow facilities to be at low-distance locations
    # So sort facilities by descending projection on v_a_top
    # and locations by ascending projection on v_b_top
    fac_order = np.argsort(-v_a_top, kind="stable")
    loc_order = np.argsort(v_b_top, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    # But this is just a rank-matching. Let's also try the second eigenvector
    # and combine.
    
    # Alternative: use the full spectral signature
    # For each facility i, signature = [v_a_top[i], v_a_2nd[i], ...]
    # For each location j, signature = [v_b_top[j], v_b_2nd[j], ...]
    # Match facility i to location j that minimizes the "cost" in spectral space
    
    # Let's try a simpler but effective approach:
    # Use the greedy heuristic combined with spectral info
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Weighted combination: spectral score + row sum score
    spectral_score_fac = -v_a_top  # higher = more "central" in flow space
    spectral_score_loc = v_b_top   # higher = more "peripheral" in distance space
    
    # Normalize
    spectral_score_fac = (spectral_score_fac - spectral_score_fac.mean()) / (spectral_score_fac.std() + 1e-10)
    spectral_score_loc = (spectral_score_loc - spectral_score_loc.mean()) / (spectral_score_loc.std() + 1e-10)
    
    row_sum_score_fac = a_row_sums / (a_row_sums.max() + 1e-10)
    row_sum_score_loc = b_row_sums / (b_row_sums.max() + 1e-10)
    
    # Combined score
    combined_fac = 0.5 * spectral_score_fac + 0.5 * row_sum_score_fac
    combined_loc = 0.5 * spectral_score_loc + 0.5 * row_sum_score_loc
    
    # Assign high combined_fac to low combined_loc
    fac_order2 = np.argsort(-combined_fac, kind="stable")
    loc_order2 = np.argsort(combined_loc, kind="stable")
    
    p2 = np.empty(n, dtype=np.int64)
    p2[fac_order2] = loc_order2
    
    # Pick the better of the two initial solutions
    cost1 = _qap_cost(a, b, p)
    cost2 = _qap_cost(a, b, p2)
    
    if cost1 <= cost2:
        return p
    else:
        return p2

def _compute_swap_delta(a, b, p, i, j):
    """Compute the cost delta for swapping positions i and j in permutation p."""
    pi = p[i]
    pj = p[j]
    
    # Term 1: sum_k (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)
    
    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j
    
    # Term 2: sum_k (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)
    
    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j
    
    # Term 3: (a[i,j] - a[j,i]) * (b[pj,pi] - b[pi,pj])
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return T1 + T2 + T3

def _local_search(a, b, p, current_cost, max_iter=5000, time_limit=None, start_time=None):
    """
    Steepest descent local search: repeatedly find the best improving swap
    and apply it until no improvement is found or limits are reached.
    """
    n = len(p)
    improved = True
    iter_count = 0
    
    while improved:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Check all pairs (i, j) with i < j
        for i in range(n):
            for j in range(i + 1, n):
                delta = _compute_swap_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i >= 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            improved = True
            iter_count += 1
        
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        if max_iter is not None and iter_count >= max_iter:
            break
    
    return p, current_cost, iter_count

def _perturb(p, strength=3):
    """
    Perturb the permutation by performing a random number of swaps.
    strength controls how many random swaps to perform.
    """
    n = len(p)
    p_perturbed = p.copy()
    num_swaps = max(1, strength + np.random.randint(0, 3))
    
    for _ in range(num_swaps):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
    
    return p_perturbed

def _simulated_annealing(a, b, p, current_cost, time_limit, start_time):
    """
    Simulated Annealing for polishing the solution.
    """
    n = len(p)
    T_start = 500.0
    T_end = 0.1
    total_deadline = time_limit
    
    while time.time() - start_time < total_deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _compute_swap_delta(a, b, p, i, j)
        new_cost = current_cost + delta
        
        elapsed = time.time() - start_time
        T = T_start * (T_end / T_start) ** (elapsed / total_deadline)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost = new_cost
    
    return p, current_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Phase 1: Spectral initialization
    p = _spectral_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Phase 2: Iterated Local Search
    # We'll run ILS until time runs out, then do a final polish
    ils_time_limit = total_deadline - 5.0  # Leave 5 seconds for final polish
    
    while time.time() - start_time < start_time + ils_time_limit:
        # Local search to convergence
        p, current_cost, _ = _local_search(
            a, b, p, current_cost,
            max_iter=2000,
            time_limit=ils_time_limit - (time.time() - start_time),
            start_time=start_time
        )
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Check if we have time for perturbation and another round
        if time.time() - start_time >= start_time + ils_time_limit - 3:
            break
        
        # Perturb the solution
        p_perturbed = _perturb(p, strength=3)
        p_perturbed_cost = _qap_cost(a, b, p_perturbed)
        
        # Accept the perturbed solution if it's not too bad, or always for diversity
        # Use a simple acceptance criterion
        if p_perturbed_cost < best_cost * 1.5:  # Not too far from best
            p = p_perturbed
            current_cost = p_perturbed_cost
        else:
            # Keep the best solution and perturb it
            p = best_p.copy()
            current_cost = best_cost
            p = _perturb(p, strength=2)
            current_cost = _qap_cost(a, b, p)
    
    # Phase 3: Final polish with local search + SA
    p_final = best_p.copy()
    cost_final = best_cost
    
    # Local search polish
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 2:
        p_final, cost_final, _ = _local_search(
            a, b, p_final, cost_final,
            max_iter=5000,
            time_limit=remaining_time - 1,
            start_time=start_time
        )
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
    
    # SA polish with remaining time
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 1:
        p_sa, cost_sa = _simulated_annealing(
            a, b, best_p.copy(), best_cost,
            time_limit=remaining_time - 0.5,
            start_time=start_time
        )
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
