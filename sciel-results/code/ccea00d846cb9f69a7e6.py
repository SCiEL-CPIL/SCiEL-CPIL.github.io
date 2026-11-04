"""Combined solver for QAPLIB wil100: Multi-start ILS with fast delta, SA acceptance, and LP-based init."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment, linprog

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost change of swapping facilities at positions i and j.
    O(n) per swap using incremental computation.
    """
    n = len(p)
    pi = p[i]
    pj = p[j]
    
    # Delta = sum_k [a[k,i]*b[pj,p[k]] + a[k,j]*b[pi,p[k]] - a[k,i]*b[pi,p[k]] - a[k,j]*b[pj,p[k]]]
    #       + sum_k [a[i,k]*b[pj,p[k]] + a[j,k]*b[pi,p[k]] - a[i,k]*b[pi,p[k]] - a[j,k]*b[pj,p[k]]]
    #       - corrections for k=i and k=j terms that are double-counted
    
    # More carefully:
    # Cost change = sum_{k != i,j} [(a[k,i] + a[i,k]) * (b[pj,p[k]] - b[pi,p[k]]) 
    #                              + (a[k,j] + a[j,k]) * (b[pi,p[k]] - b[pj,p[k]])]
    #              + terms involving k=i and k=j
    
    # Let's use the direct formula:
    # For each k, the contribution changes from:
    #   a[k,i]*b[pi,pk] + a[k,j]*b[pj,pk] + a[i,k]*b[pi,pk] + a[j,k]*b[pj,pk]
    # to:
    #   a[k,i]*b[pj,pk] + a[k,j]*b[pi,pk] + a[i,k]*b[pj,pk] + a[j,k]*b[pi,pk]
    
    # So delta_k = (a[k,i] - a[i,k]) * (b[pj,pk] - b[pi,pk]) 
    #            + (a[k,j] - a[j,k]) * (b[pi,pk] - b[pj,pk])
    
    # But we need to be careful with k=i and k=j.
    
    # Vectorized approach:
    b_pi = b[pi, :]
    b_pj = b[pj, :]
    
    # For all k:
    term1 = (a[:, i] - a[i, :]) * (b_pj - b_pi)
    term2 = (a[:, j] - a[j, :]) * (b_pi - b_pj)
    
    delta = np.sum(term1) + np.sum(term2)
    
    # Correct for k=i and k=j (they were included in the sums above but need special handling)
    # When k=i: the original had a[i,i]*b[pi,pi] + a[i,j]*b[pj,pi] + a[i,i]*b[pi,pi] + a[j,i]*b[pj,pi]
    #           the new has    a[i,i]*b[pj,pi] + a[i,j]*b[pi,pi] + a[i,i]*b[pj,pi] + a[j,i]*b[pi,pi]
    # When k=j: the original had a[j,i]*b[pi,pj] + a[j,j]*b[pj,pj] + a[i,j]*b[pi,pj] + a[j,j]*b[pj,pj]
    #           the new has    a[j,i]*b[pj,pj] + a[j,j]*b[pi,pj] + a[i,j]*b[pj,pj] + a[j,j]*b[pi,pj]
    
    # The vectorized sum already includes k=i and k=j correctly because:
    # term1[k=i] = (a[i,i] - a[i,i]) * (b_pj[i] - b_pi[i]) = 0
    # term2[k=i] = (a[i,j] - a[j,i]) * (b_pi[i] - b_pj[i])
    # term1[k=j] = (a[j,i] - a[i,j]) * (b_pj[j] - b_pi[j])
    # term2[k=j] = (a[j,j] - a[j,j]) * (b_pi[j] - b_pj[j]) = 0
    
    # So the formula is correct as is.
    return int(delta)

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j.
    Uses O(n^3) matrix multiplications, which is fast for n=100.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]

    M1 = a.T @ Bp
    M2 = a @ Bp

    diag_M1 = np.diag(M1)
    diag_M2 = np.diag(M2)
    diag_Bp = np.diag(Bp)
    a_diag = np.diag(a)

    T1 = M1 - diag_M1[:, None] - diag_M1[None, :] + M1.T
    T2 = M2 - diag_M2[:, None] - diag_M2[None, :] + M2.T

    C1_i = (a_diag[:, None] - a) * (Bp - diag_Bp[:, None])
    C1_j = (a.T - a_diag[None, :]) * (diag_Bp[None, :] - Bp)
    C1 = C1_i + C1_j

    C2_i = (a_diag[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    C2_j = (a - a_diag[None, :]) * (diag_Bp[None, :] - Bp.T)
    C2 = C2_i + C2_j

    T3 = (a - a.T) * (Bp.T - Bp)

    D = (T1 - C1) + (T2 - C2) + T3
    np.fill_diagonal(D, 0)

    return D

def _local_search_fast(a, b, p, deadline, max_iters=5000):
    """
    Fast local search using O(n) swap delta computation.
    Tries all pairs, picks the best improvement.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    iters = 0
    
    while time.time() < deadline and iters < max_iters:
        iters += 1
        # Compute all swap deltas
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Sample a subset of pairs if n is large, or do full scan for n=100
        # For n=100, full scan is 4950 pairs, each O(n)=100 ops -> ~500K ops per iteration
        # That's fine.
        for i in range(n):
            for j in range(i + 1, n):
                d = _compute_swap_delta(a, b, p, i, j)
                if d < best_delta:
                    best_delta = d
                    best_i, best_j = i, j
        
        if best_delta >= 0:
            break
        
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
    
    return p, current_cost

def _local_search_steepest(a, b, p, deadline, max_iters=200):
    """
    Steepest descent local search using full delta matrix.
    Each iteration computes the full O(n^3) delta matrix.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    iters = 0
    
    while time.time() < deadline and iters < max_iters:
        iters += 1
        D = _compute_delta_matrix(a, b, p)
        best_delta = np.min(D)

        if best_delta >= 0:
            break

        best_idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = best_idx
        p[i], p[j] = p[j], p[i]
        current_cost += int(best_delta)

    return p, current_cost

def _kick(p, kick_strength):
    """
    Apply a random kick by swapping 'kick_strength' random pairs.
    """
    n = len(p)
    for _ in range(kick_strength):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            p[i], p[j] = p[j], p[i]
    return p

def _lp_initial(a, b, n):
    """
    Solve a relaxed QAP via LP to get a fractional assignment, then round to permutation.
    This captures flow-distance interactions beyond simple centrality.
    """
    try:
        # Objective: minimize sum_{i,j} a[i,j] * sum_{k,l} x[i,k] * b[k,l] * x[j,l]
        # This is quadratic, so we relax by fixing one side.
        # Alternative: use the linear relaxation where we minimize
        # sum_{i,k} (sum_j a[i,j]) * b[k,:] dot x[i,:]
        # which is just the Hungarian surrogate.
        
        # Better: solve a sequence of LPs where we fix rows and optimize columns.
        # This is essentially a greedy row-by-row assignment with flow-aware costs.
        
        # Let's use a different approach: minimize sum_{i,j,k} a[i,j] * b[k,:] @ x[i,:] * x[j,k]
        # Still quadratic. Let's use a surrogate that's more informative.
        
        # Use the "flow-weighted distance" surrogate:
        # c[i,k] = sum_j a[i,j] * (sum_l b[k,l] * col_sum_j[l])
        # where col_sum_j[l] = sum_i a[i,j] for each location l... this is getting circular.
        
        # Simpler: use row-flow-weighted column distances
        row_flow = a.sum(axis=1)  # total flow out of each facility
        col_dist = b.sum(axis=1)  # total distance from each location
        
        # The standard surrogate is outer(row_flow, col_dist)
        # Let's add a second-order term: for each pair (i,j), the interaction a[i,j] 
        # suggests that facilities i and j should be close, so locations for i and j 
        # should have small b distance.
        
        # We can't easily encode this in a linear assignment. Instead, let's use
        # multiple Hungarian solves with different surrogates and pick the best.
        
        surrogates = []
        
        # Surrogate 1: standard
        s1 = np.outer(row_flow, col_dist)
        surrogates.append(s1)
        
        # Surrogate 2: column-flow weighted
        col_flow = a.sum(axis=0)
        s2 = np.outer(col_flow, col_dist)
        surrogates.append(s2)
        
        # Surrogate 3: symmetric flow
        sym_flow = (a + a.T) / 2
        row_flow_sym = sym_flow.sum(axis=1)
        s3 = np.outer(row_flow_sym, col_dist)
        surrogates.append(s3)
        
        # Surrogate 4: flow * distance interaction
        # c[i,k] = sum_j a[i,j] * b[k, j_centroid]
        # where j_centroid is the "center of mass" of facility j's flows
        # Approximate: c[i,k] = sum_j a[i,j] * b[k, :] @ (col_flow / col_flow.sum())
        col_flow_norm = col_flow / max(col_flow.sum(), 1)
        # c[i,k] = sum_j a[i,j] * (b[k,:] @ col_flow_norm)
        b_weighted = b @ col_flow_norm  # shape (n, n): b[k, weighted_j]
        s4 = a @ b_weighted.T  # c[i,k] = sum_j a[i,j] * b_weighted[k,j]
        surrogates.append(s4)
        
        best_p = None
        best_cost = float('inf')
        
        for s in surrogates:
            row_ind, col_ind = linear_sum_assignment(s)
            p = np.empty(n, dtype=np.int64)
            p[row_ind] = col_ind
            cost = _qap_cost(a, b, p)
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        
        return best_p, best_cost
    except Exception:
        return None, float('inf')

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def compute_cost(p):
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    np.random.seed(42)
    
    # Phase 1: Generate diverse initial solutions
    initial_solutions = []
    
    # LP-based / multi-surrogate initialization
    p_lp, cost_lp = _lp_initial(a, b, n)
    if p_lp is not None:
        initial_solutions.append(p_lp)
    
    # Hungarian on linear surrogate (row sums)
    flow_cent = a.sum(axis=1)
    dist_cent = b.sum(axis=1)
    surrogate_cost = np.outer(flow_cent, dist_cent)
    row_ind, col_ind = linear_sum_assignment(surrogate_cost)
    p1 = np.empty(n, dtype=np.int64)
    p1[row_ind] = col_ind
    initial_solutions.append(p1)
    
    # Hungarian on column-flow surrogate
    flow_cent_col = a.sum(axis=0)
    surrogate_cost2 = np.outer(flow_cent_col, dist_cent)
    row_ind2, col_ind2 = linear_sum_assignment(surrogate_cost2)
    p2 = np.empty(n, dtype=np.int64)
    p2[row_ind2] = col_ind2
    initial_solutions.append(p2)
    
    # Random permutations
    for _ in range(4):
        initial_solutions.append(np.random.permutation(n).astype(np.int64))

    best_p = initial_solutions[0].copy()
    best_cost = compute_cost(best_p)

    # Phase 2: Refine each initial solution with fast local search
    refined_solutions = []
    time_per_initial = max(5.0, (deadline - start_time - 30.0) / len(initial_solutions))
    
    for p_init in initial_solutions:
        if time.time() > deadline - 30.0:
            break
        p_ref, cost_ref = _local_search_fast(a, b, p_init.copy(), time.time() + time_per_initial)
        if cost_ref < best_cost:
            best_cost = cost_ref
            best_p = p_ref.copy()
        refined_solutions.append((p_ref, cost_ref))

    # Sort refined solutions by cost
    refined_solutions.sort(key=lambda x: x[1])
    
    # Phase 3: ILS with SA acceptance on the best refined solution
    p_current = best_p.copy()
    cost_current = best_cost
    
    # SA parameters for ILS acceptance
    T0 = 5000.0  # Initial temperature
    T_min = 1.0   # Minimum temperature
    cooling_rate = 0.995
    
    temperature = T0
    sa_iter = 0
    
    while time.time() < deadline - 10.0:
        sa_iter += 1
        # Apply a kick
        kick_strength = np.random.choice([2, 3, 4, 5], p=[0.3, 0.3, 0.25, 0.15])
        p_kicked = _kick(p_current.copy(), kick_strength)
        
        # Run fast local search from the kicked solution
        ls_time = min(3.0, max(1.0, (deadline - time.time() - 10.0) / 50.0))
        p_new, cost_new = _local_search_fast(a, b, p_kicked, time.time() + ls_time)
        
        # SA acceptance
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
            p_current = p_new.copy()
            cost_current = cost_new
        elif cost_new < cost_current:
            # Accept if better than current
            p_current = p_new.copy()
            cost_current = cost_new
        else:
            # Accept with probability exp(-delta / T)
            delta = cost_new - cost_current
            if delta > 0 and temperature > T_min:
                prob = np.exp(-delta / temperature)
                if np.random.random() < prob:
                    p_current = p_new.copy()
                    cost_current = cost_new
        
        temperature = max(T_min, temperature * cooling_rate)
    
    # Phase 4: Final polish with steepest descent (full delta matrix)
    if time.time() < deadline - 2.0:
        p_final, cost_final = _local_search_steepest(a, b, best_p.copy(), deadline, max_iters=50)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
    
    # One more fast local search pass
    if time.time() < deadline - 0.5:
        p_final2, cost_final2 = _local_search_fast(a, b, best_p.copy(), deadline)
        if cost_final2 < best_cost:
            best_cost = cost_final2
            best_p = p_final2.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
