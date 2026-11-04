"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _hungarian_init(a, b):
    """
    Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
    C[i, j] = sum_k a[i, k] * b[j, k] + sum_k a[k, i] * b[k, j]
    """
    n = len(a)
    cost_matrix = a @ b + a.T @ b.T
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=int)
    p[row_ind] = col_ind
    return p

def _eigen_init(a, b, n, rng):
    """
    Generate initial permutation using eigenvalue-based heuristic.
    """
    try:
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        
        eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
        
        k = min(32, n)
        idx_a = np.argsort(np.abs(eigvals_a))[::-1][:k]
        idx_b = np.argsort(np.abs(eigvals_b))[::-1][:k]
        
        score_a = np.zeros(n)
        for idx in idx_a:
            score_a += np.abs(eigvals_a[idx]) * np.abs(eigvecs_a[:, idx])
        
        score_b = np.zeros(n)
        for idx in idx_b:
            score_b += np.abs(eigvals_b[idx]) * np.abs(eigvecs_b[:, idx])
        
        fac_order = np.argsort(score_a)[::-1]
        loc_order = np.argsort(score_b)
        
        p = np.zeros(n, dtype=int)
        for i in range(n):
            p[fac_order[i]] = loc_order[i]
        
        p_rev = np.zeros(n, dtype=int)
        for i in range(n):
            p[fac_order[i]] = loc_order[n - 1 - i]
        
        cost1 = np.sum(a * b[np.ix_(p, p)])
        cost2 = np.sum(a * b[np.ix_(p_rev, p_rev)])
        
        if cost1 <= cost2:
            return p
        else:
            return p_rev
    except:
        return rng.permutation(n)

def _greedy_init(a, b, n, rng):
    """
    Greedy construction: place facilities one by one at the location that
    minimizes incremental cost given already-placed facilities.
    """
    p = np.full(n, -1, dtype=int)
    remaining_fac = np.arange(n)
    remaining_loc = np.arange(n)
    
    # Order facilities by total flow (most important first)
    flow_order = np.argsort(np.sum(a, axis=1))[::-1]
    
    D = b.copy()  # distance matrix in current assignment order
    
    for step in range(n):
        fac = flow_order[step]
        # For each remaining location, compute incremental cost
        best_loc = -1
        best_cost = np.inf
        
        for loc in remaining_loc:
            # Cost of placing fac at loc given already placed
            inc_cost = 0
            for j in range(step):
                placed_fac = flow_order[j]
                inc_cost += a[fac, placed_fac] * b[loc, p[placed_fac]] + a[placed_fac, fac] * b[p[placed_fac], loc]
            if inc_cost < best_cost:
                best_cost = inc_cost
                best_loc = loc
        
        p[fac] = best_loc
        remaining_loc = remaining_loc[remaining_loc != best_loc]
    
    return p

def _sa_search(a, b, p_init, start_time, time_budget, rng):
    """
    Simulated Annealing with vectorized 2-swap moves.
    Uses geometric cooling with adaptive temperature adjustment.
    """
    n = len(p_init)
    p = p_init.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute symmetric part
    a_sym = a + a.T
    
    # Temperature schedule
    T0 = current_cost * 0.01  # Initial temperature
    T_min = 1e-6
    # Number of iterations per temperature level
    iters_per_level = 200
    # Cooling factor
    alpha = 0.97
    
    T = T0
    level = 0
    
    # Track acceptance for adaptive tuning
    accept_count = 0
    total_count = 0
    
    deadline = start_time + time_budget
    
    while time.time() < deadline - 2:
        # Check time budget
        if time.time() > deadline - 2:
            break
        
        # At each temperature level, do iters_per_level moves
        for _ in range(iters_per_level):
            if time.time() > deadline - 2:
                break
            
            # Pick two random positions to swap
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            
            # Compute delta for swapping i and j
            # delta = sum_k (a_sym[i,k] - a_sym[j,k]) * (D[k,j] - D[k,i])
            #       + correction terms for diagonal
            diff = a_sym[i] - a_sym[j]
            dD = D[:, j] - D[:, i]
            delta = np.dot(diff, dD)
            # Correction for the (i,j) and (j,i) terms being double-counted
            delta -= a_sym[i, j] * (D[j, j] - D[i, i])
            delta -= a_sym[j, i] * (D[i, i] - D[j, j])
            # Actually let me recompute more carefully
            # When we swap p[i] and p[j]:
            # Old contribution involving i: sum_k a[i,k]*D[i,k] + sum_k a[k,i]*D[k,i]
            # New contribution involving i (now at old j's position): sum_k a[i,k]*D[j,k] + sum_k a[k,i]*D[k,j]
            # Similarly for j.
            # delta = sum_k [a[i,k]*(D[j,k]-D[i,k]) + a[k,i]*(D[k,j]-D[k,i])
            #              + a[j,k]*(D[i,k]-D[j,k]) + a[k,j]*(D[k,i]-D[k,j])]
            #       = sum_k [(a[i,k]-a[j,k])*(D[j,k]-D[i,k]) + (a[k,i]-a[k,j])*(D[k,j]-D[k,i])]
            #       = sum_k [(a[i,k]-a[j,k])*(D[j,k]-D[i,k]) + (a_sym[k,i]-a_sym[k,j])*(D[k,j]-D[k,i])]
            # Note D[k,j]-D[k,i] = -(D[i,k]-D[j,k]) only if D is symmetric, which b may not be
            # Let me just use the direct formula
            delta = 0.0
            # Terms where row index is i or j
            for k in range(n):
                if k == i or k == j:
                    continue
                delta += a[i, k] * (D[j, k] - D[i, k])
                delta += a[k, i] * (D[k, j] - D[k, i])
                delta += a[j, k] * (D[i, k] - D[j, k])
                delta += a[k, j] * (D[k, i] - D[k, j])
            # Diagonal terms (k==i or k==j)
            # When k==i: a[i,i]*(D[j,i]-D[i,i]) + a[i,i]*(D[i,j]-D[j,i]) ... this gets complex
            # Let me just compute it directly for the two special cases
            # k == i:
            delta += a[i, i] * (D[j, i] - D[i, i])  # a[i,i] term with D[i,i] -> D[j,i]
            delta += a[i, i] * (D[i, j] - D[j, j])  # a[i,i] term... wait
            # This is getting error-prone. Let me use the vectorized approach properly.
            
            # Actually, let me use a cleaner vectorized delta:
            # delta(i,j) for swapping positions i and j:
            # = (a[i] - a[j]) @ (D[j] - D[i]) + (a[:,i] - a[:,j]) @ (D[:,j] - D[:,i])
            #   - a[i,j]*(D[j,j]-D[i,i]) - a[j,i]*(D[i,i]-D[j,j])
            #   + a[i,j]*(D[i,j]-D[j,i]) + a[j,i]*(D[j,i]-D[i,j])
            # Simplified with a_sym:
            # = a_sym[i] @ (D[j] - D[i]) + a_sym[:,i] @ (D[:,j] - D[:,i])
            #   - a[i,i]*(D[j,j]-D[i,i]) - a[j,j]*(D[i,i]-D[j,j])
            #   + (a[i,j]-a[j,i])*(D[i,j]-D[j,i])
            
            delta = np.dot(a_sym[i], D[j] - D[i]) + np.dot(a_sym[:, i], D[:, j] - D[:, i])
            delta -= a[i, i] * (D[j, j] - D[i, i])
            delta -= a[j, j] * (D[i, i] - D[j, j])
            delta += (a[i, j] - a[j, i]) * (D[i, j] - D[j, i])
            
            delta = int(delta)
            total_count += 1
            
            if delta <= 0 or rng.random() < np.exp(-delta / max(T, T_min)):
                # Accept the move
                accept_count += 1
                p[i], p[j] = p[j], p[i]
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        
        # Update temperature
        # Adaptive: if acceptance rate is too low, cool slower
        if total_count > 0:
            accept_rate = accept_count / total_count
            if accept_rate < 0.05:
                alpha = min(0.995, alpha + 0.005)
            elif accept_rate > 0.5:
                alpha = max(0.95, alpha - 0.005)
        
        T *= alpha
        if T < T_min:
            T = T_min
        
        # Reset counters
        accept_count = 0
        total_count = 0
    
    return best_p, best_cost

def _vectorized_sa_search(a, b, p_init, start_time, time_budget, rng):
    """
    Simulated Annealing with vectorized batch evaluation of all 2-swap moves.
    At each step, evaluate all n*(n-1)/2 possible swaps and pick the best
    (or a random one among improving moves) to accept.
    """
    n = len(p_init)
    p = p_init.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    best_cost = current_cost
    best_p = p.copy()
    
    a_sym = a + a.T
    
    # Temperature schedule
    T = current_cost * 0.005
    T_min = 1e-6
    alpha = 0.985
    
    deadline = start_time + time_budget
    
    # Precompute indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    num_pairs = len(i_idx)
    
    step = 0
    while time.time() < deadline - 3:
        if time.time() > deadline - 3:
            break
        
        step += 1
        
        # Vectorized delta computation for all pairs
        # delta(i,j) = a_sym[i] @ (D[j] - D[i]) + a_sym[:,i] @ (D[:,j] - D[:,i])
        #              - a[i,i]*(D[j,j]-D[i,i]) - a[j,j]*(D[i,i]-D[j,j])
        #              + (a[i,j]-a[j,i])*(D[i,j]-D[j,i])
        
        diff_D = D[j_idx] - D[i_idx]  # (num_pairs, n)
        term1 = np.sum(a_sym[i_idx] * diff_D, axis=1)  # (num_pairs,)
        
        # term2: a_sym[:,i] @ (D[:,j] - D[:,i])
        # This is: sum_k a_sym[k,i] * (D[k,j] - D[k,i])
        # = sum_k a_sym[k,i]*D[k,j] - sum_k a_sym[k,i]*D[k,i]
        term2 = np.sum(a_sym[:, i_idx] * (D[:, j_idx] - D[:, i_idx]), axis=0)  # wrong shape
        
        # Let me redo this properly
        # For pair (i,j): term2 = sum_k a_sym[k,i] * (D[k,j] - D[k,i])
        # = (a_sym[:,i]) @ (D[:,j] - D[:,i])
        
        # Vectorized: for all pairs simultaneously
        # a_sym[:, i_idx] has shape (n, num_pairs)
        # D[:, j_idx] has shape (n, num_pairs)
        # D[:, i_idx] has shape (n, num_pairs)
        
        a_col_i = a_sym[:, i_idx]  # (n, num_pairs)
        D_col_j = D[:, j_idx]  # (n, num_pairs)
        D_col_i = D[:, i_idx]  # (n, num_pairs)
        
        term2 = np.sum(a_col_i * (D_col_j - D_col_i), axis=0)  # (num_pairs,)
        
        # Diagonal corrections
        diag_i = a[i_idx, i_idx]  # (num_pairs,)
        diag_j = a[j_idx, j_idx]  # (num_pairs,)
        D_ii = D[i_idx, i_idx]  # (num_pairs,)
        D_jj = D[j_idx, j_idx]  # (num_pairs,)
        D_ij = D[i_idx, j_idx]  # (num_pairs,)
        D_ji = D[j_idx, i_idx]  # (num_pairs,)
        a_ij = a[i_idx, j_idx]  # (num_pairs,)
        a_ji = a[j_idx, i_idx]  # (num_pairs,)
        
        term3 = -diag_i * (D_jj - D_ii)
        term4 = -diag_j * (D_ii - D_jj)
        term5 = (a_ij - a_ji) * (D_ij - D_ji)
        
        deltas = term1 + term2 + term3 + term4 + term5
        deltas = deltas.astype(np.int64)
        
        # Find improving moves
        improving = deltas < 0
        if np.any(improving):
            # Among improving moves, pick the best (most negative delta)
            min_idx = np.argmin(deltas)
            delta_val = int(deltas[min_idx])
            i_val = i_idx[min_idx]
            j_val = j_idx[min_idx]
            
            # Accept with probability (always accept if improving, or use SA)
            if delta_val <= 0:
                p[i_val], p[j_val] = p[j_val], p[i_val]
                D[i_val], D[j_val] = D[j_val].copy(), D[i_val].copy()
                D[:, i_val], D[:, j_val] = D[:, j_val].copy(), D[:, i_val].copy()
                current_cost += delta_val
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            else:
                # Shouldn't happen since we filtered, but just in case
                if rng.random() < np.exp(-delta_val / max(T, T_min)):
                    p[i_val], p[j_val] = p[j_val], p[i_val]
                    D[i_val], D[j_val] = D[j_val].copy(), D[i_val].copy()
                    D[:, i_val], D[:, j_val] = D[:, j_val].copy(), D[:, i_val].copy()
                    current_cost += delta_val
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
        else:
            # No improving moves; pick a random move and accept with SA probability
            rand_idx = rng.integers(0, num_pairs)
            delta_val = int(deltas[rand_idx])
            i_val = i_idx[rand_idx]
            j_val = j_idx[rand_idx]
            
            if rng.random() < np.exp(-delta_val / max(T, T_min)):
                p[i_val], p[j_val] = p[j_val], p[i_val]
                D[i_val], D[j_val] = D[j_val].copy(), D[i_val].copy()
                D[:, i_val], D[:, j_val] = D[:, j_val].copy(), D[:, i_val].copy()
                current_cost += delta_val
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 145  # Leave 5s safety margin
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse initial solutions
    p_hung = _hungarian_init(a, b)
    p_eigen = _eigen_init(a, b, n, rng)
    p_random = rng.permutation(n)
    
    initial_solutions = [p_hung, p_eigen, p_random]
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Quick local search on each initial solution to get good starting points
    # Use a fast first-improvement local search
    for p_init in initial_solutions:
        elapsed = time.time() - start_time
        remaining = time_budget - 60 - elapsed
        if remaining < 10:
            break
        
        p_ls, cost_ls = _quick_local_search(a, b, p_init, remaining / len(initial_solutions), start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    if best_p is None:
        best_p = p_hung.copy()
        best_cost = _compute_cost(a, b, best_p)
    
    # Phase 2: Simulated Annealing from the best solution
    elapsed = time.time() - start_time
    remaining = time_budget - 5 - elapsed
    if remaining > 10:
        p_sa, cost_sa = _sa_search(a, b, best_p, start_time, remaining, rng)
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
    
    # Phase 3: If time remains, try SA from other initial solutions
    elapsed = time.time() - start_time
    remaining = time_budget - 5 - elapsed
    if remaining > 20:
        for p_init in initial_solutions:
            if time.time() - start_time > time_budget - 10:
                break
            rem = time_budget - 10 - (time.time() - start_time)
            if rem < 10:
                break
            p_sa, cost_sa = _sa_search(a, b, p_init, start_time, rem, rng)
            if cost_sa < best_cost:
                best_cost = cost_sa
                best_p = p_sa.copy()
    
    # Final: quick local search polish on the best solution
    elapsed = time.time() - start_time
    remaining = time_budget - 2 - elapsed
    if remaining > 5 and best_p is not None:
        p_final, cost_final = _quick_local_search(a, b, best_p, remaining, start_time)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
    
    return best_p.tolist()

def _quick_local_search(a, b, p, time_limit, start_time):
    """
    Fast first-improvement 2-swap local search.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    a_sym = a + a.T
    
    deadline = start_time + time_limit
    
    while time.time() < deadline - 1:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline - 1:
                return p, current_cost
            
            js = np.arange(i + 1, n)
            
            # Vectorized delta computation
            diff_a = a_sym[i] - a_sym[js]
            diff_D = D[js] - D[i]
            S = np.sum(diff_a * diff_D, axis=1)
            
            coeff_i = a_sym[i, i] - a_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            coeff_j = a_sym[i, js] - np.diag(a_sym[js])
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
            
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S - term_i - term_j + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                p[i], p[j] = p[j], p[i]
                
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost
# EVOLVE-BLOCK-END
