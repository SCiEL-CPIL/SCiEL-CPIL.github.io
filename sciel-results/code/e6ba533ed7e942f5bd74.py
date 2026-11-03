"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Safety margin: 120s limit, aim to finish by 115s
    deadline = time.time() + 115.0

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def swap_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        # Vectorized delta calculation
        # S1 = sum_k (a[i, k] - a[j, k]) * (b[v, p[k]] - b[u, p[k]])
        # S2 = sum_k (a[k, i] - a[k, j]) * (b[p[k], v] - b[p[k], u])
        
        diff_a_rows = a[i, :] - a[j, :]
        diff_b_rows = b[v, p] - b[u, p]
        s1 = np.dot(diff_a_rows, diff_b_rows)
        
        diff_a_cols = a[:, i] - a[:, j]
        diff_b_cols = b[p, v] - b[p, u]
        s2 = np.dot(diff_a_cols, diff_b_cols)
        
        # Correction for k=i and k=j
        # Terms in s1 for k=i: (a[i,i] - a[j,i]) * (b[v, u] - b[u, u])
        # Terms in s2 for k=i: (a[i,i] - a[j,i]) * (b[u, v] - b[u, u])
        # Terms in s1 for k=j: (a[i,j] - a[j,j]) * (b[v, v] - b[u, v])
        # Terms in s2 for k=j: (a[i,j] - a[j,j]) * (b[u, v] - b[v, v])
        
        ai_i = a[i, i]
        aj_i = a[j, i]
        ai_j = a[i, j]
        aj_j = a[j, j]
        
        bu_u = b[u, u]
        bu_v = b[u, v]
        bv_u = b[v, u]
        bv_v = b[v, v]
        
        corr_1 = (ai_i - aj_i) * (bv_u - bu_u)
        corr_2 = (ai_i - ai_j) * (bu_v - bu_u)
        corr_3 = (ai_j - aj_j) * (bv_v - bu_v)
        corr_4 = (aj_i - aj_j) * (bv_v - bv_u)
        
        # Correct interaction term for i, j
        # New interaction: a[i,j]b[v,u] + a[j,i]b[u,v]
        # Old interaction: a[i,j]b[u,v] + a[j,i]b[v,u]
        # Delta interaction: (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
        corr_interaction = (ai_j - aj_i) * (bv_u - bu_v)
        
        return s1 + s2 - (corr_1 + corr_2 + corr_3 + corr_4) + corr_interaction

    def local_search(p, current_cost, deadline):
        improved = True
        max_iter = 500 
        iter_count = 0
        
        while improved and time.time() < deadline - 2.0 and iter_count < max_iter:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Precompute b indexed by p for faster access
            bp = b[np.ix_(p, p)]
            
            # Scan all pairs to find the steepest descent move
            for i in range(n - 1):
                if time.time() >= deadline - 2.0:
                    return p, current_cost
                u = p[i]
                for j in range(i + 1, n):
                    v = p[j]
                    
                    # Vectorized delta using precomputed bp
                    # S1 = sum_k (a[i,k] - a[j,k]) * (bp[v,k] - bp[u,k])
                    diff_a = a[i, :] - a[j, :]
                    diff_bp = bp[v, :] - bp[u, :]
                    s1 = np.dot(diff_a, diff_bp)
                    
                    # S2 = sum_k (a[:,i] - a[:,j]) * (bp[:,v] - bp[:,u])
                    diff_aT = a[:, i] - a[:, j]
                    diff_bpT = bp[:, v] - bp[:, u]
                    s2 = np.dot(diff_aT, diff_bpT)
                    
                    # Corrections for k=i and k=j
                    ai_i = a[i, i]
                    aj_i = a[j, i]
                    ai_j = a[i, j]
                    aj_j = a[j, j]
                    
                    bu_u = bp[u, u]
                    bu_v = bp[u, v]
                    bv_u = bp[v, u]
                    bv_v = bp[v, v]
                    
                    corr_1 = (ai_i - aj_i) * (bv_u - bu_u)
                    corr_2 = (ai_i - ai_j) * (bu_v - bu_u)
                    corr_3 = (ai_j - aj_j) * (bv_v - bu_v)
                    corr_4 = (aj_i - aj_j) * (bv_v - bv_u)
                    corr_interaction = (ai_j - aj_i) * (bv_u - bu_v)
                    
                    d = s1 + s2 - (corr_1 + corr_2 + corr_3 + corr_4) + corr_interaction
                    
                    if d < best_delta:
                        best_delta = d
                        best_i = i
                        best_j = j
            
            if best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
                iter_count += 1
            else:
                break
                
        return p, current_cost

    def greedy_init():
        """
        Greedy construction: assign facilities in order of flow sum 
        to the location that minimizes the incremental cost.
        """
        p = np.full(n, -1, dtype=np.int64)
        flow_sums = a.sum(axis=1)
        order = np.argsort(-flow_sums, kind='stable')
        
        assigned = np.zeros(n, dtype=bool)
        
        for idx in order:
            best_loc = -1
            best_inc = np.inf
            
            # For each unassigned location, calculate the incremental cost
            # of placing facility idx there.
            # Incremental cost = sum_{j: assigned} (a[idx, j] + a[j, idx]) * b[loc, p[j]]
            
            # We can vectorize this if we maintain a list of assigned facilities and their locations.
            # Let assigned_facilities be the list of facilities already placed.
            # Let assigned_locations be p[assigned_facilities].
            
            # To keep it simple and fast enough for n=150, we can just loop through unassigned locations
            # and compute the dot product with the already assigned part.
            
            # Optimization: Precompute the interaction vector for facility idx with all other facilities
            # inter_vec = a[idx, :] + a[:, idx]
            # Then for a location loc, inc = sum_{j in assigned} inter_vec[j] * b[loc, p[j]]
            
            inter_vec = a[idx, :] + a[:, idx]
            
            # Get indices of assigned facilities
            assigned_mask = assigned
            # We need to sum inter_vec[j] * b[loc, p[j]] for all j where assigned[j] is True
            
            # b[loc, p] is a vector of length n.
            # We can multiply inter_vec by a mask of assigned, then dot with b[loc, p]?
            # No, b[loc, p[j]] depends on j.
            
            # Let's just loop over unassigned locations. n is 150.
            # Inner loop is over assigned facilities (up to 150).
            # Total ops ~ n^3 / 2 = 1.7M. Very fast.
            
            for loc in range(n):
                if assigned[loc]:
                    continue
                
                # Calculate incremental cost
                # Sum over all j such that assigned[j] is True
                # term_j = inter_vec[j] * b[loc, p[j]]
                
                # Vectorized way:
                # mask = assigned.astype(float)
                # terms = inter_vec * mask
                # inc = np.dot(terms, b[loc, p]) # This is wrong because p[j] is not j.
                
                # Correct vectorized way:
                # We have b[loc, :] which is a row of B.
                # We want sum_{j: assigned} inter_vec[j] * b[loc, p[j]]
                # This is equivalent to:
                # Create a vector v where v[p[j]] = inter_vec[j] if assigned[j] else 0?
                # That's complicated.
                
                # Let's just use the assigned indices.
                assigned_indices = np.where(assigned)[0]
                if len(assigned_indices) == 0:
                    inc = 0.0
                else:
                    # b[loc, p[assigned_indices]] gives the distances from loc to the locations of assigned facilities
                    # inter_vec[assigned_indices] gives the flow interactions
                    inc = np.dot(inter_vec[assigned_indices], b[loc, p[assigned_indices]])
                
                if inc < best_inc:
                    best_inc = inc
                    best_loc = loc
            
            p[idx] = best_loc
            assigned[best_loc] = True
            
        return p

    # 1. Initialization
    best_cost = np.inf
    best_p = None
    
    # Generate diverse initial solutions
    inits = []
    
    # Greedy Init
    inits.append(greedy_init())
    
    # Rank Init (Forward)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    inits.append(p_rank)
    
    # Rank Init (Reversed)
    p_rank_rev = np.empty(n, dtype=np.int64)
    p_rank_rev[heavy] = central[::-1]
    inits.append(p_rank_rev)
    
    # Multiple Random Inits with different seeds
    for seed in [42, 123, 456]:
        rng = np.random.RandomState(seed)
        inits.append(rng.permutation(n))
    
    # Hungarian-based Init: linear assignment relaxation
    # C[i, loc] = sum_j (a[i,j] + a[j,i]) * b[loc, j]
    # This captures pairwise affinity structure in a linear form
    try:
        from scipy.optimize import linear_sum_assignment
        # C[i, loc] = sum_j (a[i,j] + a[j,i]) * b[loc, j]
        # = sum_j a[i,j]*b[loc,j] + sum_j a[j,i]*b[loc,j]
        # = (a @ b.T)[i, loc] + (a.T @ b.T)[i, loc]
        # Wait: (a @ b.T)[i, loc] = sum_j a[i,j] * b[loc, j] ✓
        # (a.T @ b.T)[i, loc] = sum_j a[j,i] * b[loc, j] ✓
        C = (a @ b.T) + (a.T @ b.T)
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        inits.append(p_hung)
    except ImportError:
        pass
    
    # Second Hungarian variant: asymmetric
    # C2[i, loc] = sum_j a[i,j] * b[loc, j]  (only forward flows)
    try:
        from scipy.optimize import linear_sum_assignment
        C2 = a @ b.T
        row_ind2, col_ind2 = linear_sum_assignment(C2)
        p_hung2 = np.empty(n, dtype=np.int64)
        p_hung2[row_ind2] = col_ind2
        inits.append(p_hung2)
    except ImportError:
        pass
    
    # Run local search from each initialization
    for p_init in inits:
        if time.time() > deadline - 15.0:
            break
        
        p_work = p_init.copy()
        cost = compute_cost(p_work)
        
        p_work, cost = local_search(p_work, cost, deadline)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p_work.copy()
    
    if best_p is None:
        best_p = inits[0].copy()
        best_cost = compute_cost(best_p)
    
    # 2. Iterated Local Search (ILS)
    # Use random perturbations to escape local minima
    no_improve_count = 0
    rng_ils = np.random.RandomState(123)
    
    while time.time() < deadline - 8.0:
        # Perturb the best solution
        p = best_p.copy()
        
        # Adaptive perturbation: increase swaps if stuck
        base_swaps = 5
        extra_swaps = min(no_improve_count, 5) * 2
        num_swaps = base_swaps + extra_swaps
        num_swaps = min(num_swaps, 25) # Cap to avoid destroying the solution completely
        
        for _ in range(num_swaps):
            i = rng_ils.randint(0, n)
            j = rng_ils.randint(0, n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        current_cost = compute_cost(p)
        
        # Run local search from the perturbed state
        p, current_cost = local_search(p, current_cost, deadline)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Reset counter occasionally to allow larger perturbations if stuck for long
        if no_improve_count > 15:
            no_improve_count = 0

    # Final verification
    if best_p is not None:
        final_cost = compute_cost(best_p)
        if final_cost < best_cost:
            best_cost = final_cost
        return best_p.tolist()
    
    return list(range(n))

# EVOLVE-BLOCK-END
