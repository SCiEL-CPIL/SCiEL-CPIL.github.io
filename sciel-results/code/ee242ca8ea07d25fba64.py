"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Ensure int64 for exact arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    def calc_cost(p):
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        b_perm = b[:, p][:, p]
        return int(np.sum(a * b_perm))
    
    # Check symmetry
    is_symmetric_a = np.allclose(a, a.T)
    is_symmetric_b = np.allclose(b, b.T)
    use_fast_path = is_symmetric_a and is_symmetric_b
    
    # GRASP: Greedy Randomized Adaptive Search Procedure
    # Multiple diverse initializations via randomized greedy construction
    
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Precompute for fast delta calculation
    # For symmetric case, we can use a simpler formula
    
    def grasp_construct(alpha, rng):
        """
        Vectorized GRASP construction.
        """
        p = np.full(n, -1, dtype=np.int64)
        assigned_loc_mask = np.zeros(n, dtype=bool)
        assigned_fac_mask = np.zeros(n, dtype=bool)
        
        facility_weights = a.sum(axis=1)
        
        # Track cumulative cost contributions for fast delta updates
        # When facility i is placed at location j, the marginal cost involves
        # interactions with all previously assigned facilities
        
        for step in range(n):
            if time.time() > deadline - 10.0:
                break
            
            unassigned_fac = np.where(~assigned_fac_mask)[0]
            unassigned_loc = np.where(~assigned_loc_mask)[0]
            
            if len(unassigned_fac) == 0 or len(unassigned_loc) == 0:
                break
            
            # Consider top facilities by weight
            fac_weights = facility_weights[unassigned_fac]
            sorted_idx = np.argsort(-fac_weights)
            top_k = min(30, len(unassigned_fac))
            candidate_facs = unassigned_fac[sorted_idx[:top_k]]
            
            best_fac = -1
            best_loc = -1
            best_min_cost = np.inf
            
            # Get assigned facilities and their locations for cost computation
            assigned_facs = np.where(assigned_fac_mask)[0]
            if len(assigned_facs) > 0:
                assigned_locs = p[assigned_facs]
            else:
                assigned_locs = np.array([], dtype=np.int64)
            
            for fac in candidate_facs:
                # Vectorized cost computation for all unassigned locations
                # cost[loc] = sum_k (a[fac,k]*b[loc,p[k]] + a[k,fac]*b[p[k],loc]) + a[fac,fac]*b[loc,loc]
                
                if len(assigned_facs) > 0:
                    # b[loc, assigned_locs] for all unassigned locs at once
                    b_loc_assigned = b[np.ix_(unassigned_loc, assigned_locs)]  # shape (n_loc, n_assigned)
                    a_fac_assigned = a[fac, assigned_facs]  # shape (n_assigned,)
                    a_assigned_fac = a[assigned_facs, fac]  # shape (n_assigned,)
                    
                    # cost from cross terms
                    cost_cross = b_loc_assigned @ a_fac_assigned + b_loc_assigned.T @ a_assigned_fac
                    # Wait, that's not right. Let me think again.
                    # For each unassigned location loc, cost = sum_k a[fac,k]*b[loc,p[k]] + a[k,fac]*b[p[k],loc]
                    # b_loc_assigned[loc_idx, k_idx] = b[unassigned_loc[loc_idx], assigned_locs[k_idx]]
                    # So sum_k a[fac,k]*b[loc,p[k]] = b_loc_assigned[loc_idx,:] @ a_fac_assigned
                    # And sum_k a[k,fac]*b[p[k],loc] = b_loc_assigned[loc_idx,:] @ a_assigned_fac
                    # But b[p[k],loc] = b[assigned_locs[k_idx], unassigned_loc[loc_idx]] which is the transpose
                    b_assigned_loc = b[np.ix_(assigned_locs, unassigned_loc)]  # shape (n_assigned, n_loc)
                    cost_cross1 = b_loc_assigned @ a_fac_assigned  # shape (n_loc,)
                    cost_cross2 = b_assigned_loc.T @ a_assigned_fac  # shape (n_loc,)
                    cost_cross = cost_cross1 + cost_cross2
                else:
                    cost_cross = np.zeros(len(unassigned_loc))
                
                # Self term
                b_loc_diag = b[unassigned_loc, unassigned_loc]
                cost_self = a[fac, fac] * b_loc_diag
                
                loc_costs = cost_cross + cost_self
                
                min_cost = np.min(loc_costs)
                max_cost = np.max(loc_costs)
                rcl_threshold = min_cost + alpha * (max_cost - min_cost)
                
                rcl_mask = loc_costs <= rcl_threshold
                if not np.any(rcl_mask):
                    rcl_mask = np.zeros(len(unassigned_loc), dtype=bool)
                    rcl_mask[np.argmin(loc_costs)] = True
                
                rcl_locs = unassigned_loc[rcl_mask]
                selected_loc = rcl_locs[rng.integers(len(rcl_locs))]
                
                if min_cost < best_min_cost:
                    best_min_cost = min_cost
                    best_fac = fac
                    best_loc = selected_loc
            
            if best_fac == -1:
                best_fac = unassigned_fac[0]
                best_loc = unassigned_loc[0]
            
            p[best_fac] = best_loc
            assigned_fac_mask[best_fac] = True
            assigned_loc_mask[best_loc] = True
        
        # Fill any remaining unassigned (shouldn't happen)
        remaining_fac = np.where(~assigned_fac_mask)[0]
        remaining_loc = np.where(~assigned_loc_mask)[0]
        if len(remaining_fac) > 0 and len(remaining_loc) > 0:
            for i in range(min(len(remaining_fac), len(remaining_loc))):
                p[remaining_fac[i]] = remaining_loc[i]
                assigned_fac_mask[remaining_fac[i]] = True
                assigned_loc_mask[remaining_loc[i]] = True
        
        return p
    
    def local_search_2swap(p, max_iterations=1000):
        """
        Optimized 2-swap local search with vectorized delta computation.
        """
        current_cost = calc_cost(p)
        best_cost = current_cost
        best_p = p.copy()
        
        no_improve = 0
        max_no_improve = 80
        
        # Precompute row sums for faster delta calculations
        a_row_sums = a.sum(axis=1)
        a_col_sums = a.sum(axis=0)
        
        while no_improve < max_no_improve and time.time() < deadline - 5.0:
            B_p = b[:, p]  # B_p[k, m] = b[k, p[m]]
            B_pT = b[p, :]  # B_pT[m, k] = b[p[m], k]
            
            best_delta = np.inf
            best_i = -1
            best_j = -1
            
            for i in range(n):
                u = p[i]
                # Precompute for facility i
                a_i = a[i]
                a_i_col = a[:, i]
                B_u = B_p[:, u]
                B_u_row = b[u, :]
                
                for j in range(i + 1, n):
                    v = p[j]
                    
                    if use_fast_path:
                        # Symmetric case: delta = 2 * [(a[i]-a[j]) . (B_p[v]-B_p[u]) - corrections]
                        diff_a = a_i - a[j]
                        diff_b = B_p[:, v] - B_u
                        full_dot = np.dot(diff_a, diff_b)
                        
                        # Correction terms for diagonal/self interactions
                        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                        
                        delta = 2 * (full_dot - term_i - term_j)
                    else:
                        # General case
                        diff_a_row = a_i - a[j]
                        diff_b_row = B_p[:, v] - B_u
                        sum1_full = np.dot(diff_a_row, diff_b_row)
                        term1_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                        term1_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                        sum1 = sum1_full - term1_i - term1_j
                        
                        diff_a_col = a_i_col - a[:, j]
                        diff_b_col = B_pT[v, :] - B_pT[u, :]
                        sum2_full = np.dot(diff_a_col, diff_b_col)
                        term2_i = (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
                        term2_j = (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
                        sum2 = sum2_full - term2_i - term2_j
                        
                        direct = (a[i, j] * (b[v, u] - b[u, v]) +
                                  a[j, i] * (b[u, v] - b[v, u]) +
                                  a[i, i] * (b[v, v] - b[u, u]) +
                                  a[j, j] * (b[u, u] - b[v, v]))
                        
                        delta = sum1 + sum2 + direct
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i == -1 or best_delta >= 0:
                no_improve += 1
                # Perturb to escape local optimum - use larger perturbation
                n_perturb = min(8, n // 2)
                for _ in range(n_perturb):
                    idx1 = rng.integers(n)
                    idx2 = rng.integers(n)
                    if idx1 != idx2:
                        p[idx1], p[idx2] = p[idx2], p[idx1]
                current_cost = calc_cost(p)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                no_improve = 0
            else:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += int(best_delta)
                no_improve = 0
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        
        return best_p, best_cost
    
    # Main GRASP loop
    best_p = None
    best_cost = np.inf
    rng = np.random.default_rng()
    
    # First, try the rank-matching heuristic as a baseline
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_row_sums, kind="stable")
    central_order = np.argsort(b_row_sums, kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy_order] = central_order
    p_improved, cost_improved = local_search_2swap(p_init)
    if cost_improved < best_cost:
        best_cost = cost_improved
        best_p = p_improved
    
    # Now run multiple GRASP iterations with different alpha values
    alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    grasp_iterations = 0
    
    while time.time() < deadline - 15.0:
        # Cycle through alpha values
        alpha = alpha_values[grasp_iterations % len(alpha_values)]
        grasp_iterations += 1
        
        p_grasp = grasp_construct(alpha, rng)
        p_improved, cost_improved = local_search_2swap(p_grasp)
        
        if cost_improved < best_cost:
            best_cost = cost_improved
            best_p = p_improved
    
    # Final polishing phase: use remaining time for intensive local search
    if best_p is not None and time.time() < deadline - 3.0:
        p_polish = best_p.copy()
        # Run local search with more iterations in the remaining time
        while time.time() < deadline - 2.0:
            p_polish, cost_polish = local_search_2swap(p_polish)
            if cost_polish < best_cost:
                best_cost = cost_polish
                best_p = p_polish
            else:
                # No improvement, break to avoid wasting time
                break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
