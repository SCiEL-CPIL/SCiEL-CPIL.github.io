import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Convert to float64 for faster arithmetic in deltas, but keep int for final cost if needed
    # Actually, a and b are int64. Operations on int64 can be slow. 
    # Let's cast to float64 for the search process, but be careful with precision?
    # QAP costs can be large. 150*150*max_val.
    # Let's check max value. If max is 1e6, cost is ~1e12. float64 has 15-16 digits. Safe.
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    def compute_cost(p):
        """Compute cost using vectorized matrix multiplication."""
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        bp = b_f[np.ix_(p, p)]
        return int(np.sum(a_f * bp))

    def get_delta(p, i, j):
        """
        Compute the change in cost if we swap facilities i and j.
        p is the current permutation.
        delta = new_cost - old_cost
        """
        li, lj = p[i], p[j]
        if li == lj:
            return 0.0

        # Vectorized delta calculation
        # Terms involving i and j directly:
        # a[i,i] * b[li,li] -> a[i,i] * b[lj,lj]
        # a[i,j] * b[li,lj] -> a[i,j] * b[lj,li]
        # a[j,i] * b[lj,li] -> a[j,i] * b[li,lj]
        # a[j,j] * b[lj,lj] -> a[j,j] * b[li,li]
        
        # Terms involving i and k (k != i, j):
        # a[i,k] * b[li,p[k]] -> a[i,k] * b[lj,p[k]]
        # a[k,i] * b[p[k],li] -> a[k,i] * b[p[k],lj]
        
        # Terms involving j and k (k != i, j):
        # a[j,k] * b[lj,p[k]] -> a[j,k] * b[li,p[k]]
        # a[k,j] * b[p[k],lj] -> a[k,j] * b[p[k],li]
        
        # We can combine the k terms:
        # sum_{k != i,j} [ a[i,k] * (b[lj,p[k]] - b[li,p[k]]) + a[k,i] * (b[p[k],lj] - b[p[k],li]) ]
        # + sum_{k != i,j} [ a[j,k] * (b[li,p[k]] - b[lj,p[k]]) + a[k,j] * (b[p[k],li] - b[p[k],lj]) ]
        
        # Let's compute using full arrays and subtract i, j contributions to avoid masking overhead?
        # Or just use fancy indexing. n=150 is small.
        
        # diff_b_col = b[lj, p] - b[li, p]
        # diff_b_row = b[p, lj] - b[p, li]
        
        # delta_row_i = np.dot(a[i], diff_b_col)
        # delta_col_i = np.dot(a[:, i], diff_b_row)
        
        # delta_row_j = np.dot(a[j], -diff_b_col)  # since b[li,p] - b[lj,p] = -diff_b_col
        # delta_col_j = np.dot(a[:, j], -diff_b_row)
        
        # Total delta for all k (including i, j):
        # delta_all = delta_row_i + delta_col_i + delta_row_j + delta_col_j
        
        # But this double counts or miscounts the i-j interactions?
        # Let's verify.
        # The term for k=i in delta_row_i is a[i,i]*(b[lj,li]-b[li,li]). Correct.
        # The term for k=j in delta_row_i is a[i,j]*(b[lj,p[j]]-b[li,p[j]]) = a[i,j]*(b[lj,lj]-b[li,lj]). Correct.
        # The term for k=i in delta_col_i is a[i,i]*(b[p[i],lj]-b[p[i],li]) = a[i,i]*(b[li,lj]-b[li,li]). Correct.
        # The term for k=j in delta_col_i is a[j,i]*(b[p[j],lj]-b[p[j],li]) = a[j,i]*(b[lj,lj]-b[lj,li]). Correct.
        
        # Similarly for j.
        # So summing over all k is valid.
        
        diff_b_col = b_f[lj, p] - b_f[li, p]
        diff_b_row = b_f[p, lj] - b_f[p, li]
        
        delta = (np.dot(a_f[i], diff_b_col) + 
                 np.dot(a_f[:, i], diff_b_row) + 
                 np.dot(a_f[j], -diff_b_col) + 
                 np.dot(a_f[:, j], -diff_b_row))
        
        return delta

    def local_search(p, current_cost, time_limit):
        """Perform 2-swap local search until no improvement or time limit."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, n):
                    if time.time() > time_limit:
                        break
                    delta = get_delta(p, i, j)
                    if delta < -1e-9:  # Small epsilon for float
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        # Restart loop to ensure we don't miss better moves
                        break
                if improved:
                    continue
        return p, current_cost

    def grasp_construct(alpha):
        """
        GRASP construction:
        Iteratively assign facilities to locations.
        At each step, for each unassigned facility, compute the cost of assigning it to each unassigned location.
        Select the best ones (RCL) and pick one randomly.
        """
        assigned_f = np.zeros(n, dtype=bool)
        assigned_l = np.zeros(n, dtype=bool)
        p = np.empty(n, dtype=np.int64)
        current_cost = 0.0
        
        # Precompute? No, costs change as we assign.
        # We need to update the "potential" cost.
        # Cost contribution of assigning facility i to location l, given current assignments:
        # C(i, l) = sum_{k assigned} (a[i,k]*b[l,p[k]] + a[k,i]*b[p[k],l]) + a[i,i]*b[l,l]
        # We can maintain a vector `cost_to_loc` for each facility i, representing the cost to assign i to each location l.
        # Initially 0.
        # When we assign facility k to location lk:
        # For all unassigned i:
        #   cost_to_loc[i, :] += a[i,k]*b[lk,:] + a[k,i]*b[:,lk] + a[i,i]*b[lk,lk] (only for l=lk? No, b[l,lk] is for l)
        #   Specifically:
        #   term1: a[i,k] * b[lk, l] for all l. So add a[i,k] * b[lk, :] to cost_to_loc[i, :]
        #   term2: a[k,i] * b[l, lk] for all l. So add a[k,i] * b[:, lk] to cost_to_loc[i, :]
        #   term3: a[i,i] * b[lk, lk] only for l=lk. So add a[i,i] * b[lk, lk] to cost_to_loc[i, lk]
        
        cost_to_loc = np.zeros((n, n), dtype=np.float64)
        
        for step in range(n):
            # Build RCL for unassigned facilities
            # Actually, GRASP usually picks one facility at a time.
            # Which facility? Usually the one with highest cost variance or just random?
            # Standard GRASP: Pick a facility i (maybe random or greedy).
            # Let's pick the facility with the highest max cost? Or just iterate 0..n-1?
            # If we iterate 0..n-1, it's not fully randomized.
            # Let's pick a random unassigned facility.
            
            unassigned_f = np.where(~assigned_f)[0]
            i = unassigned_f[np.random.randint(len(unassigned_f))]
            
            # Get costs for this facility i to all unassigned locations
            unassigned_l = np.where(~assigned_l)[0]
            costs = cost_to_loc[i, unassigned_l]
            
            # RCL: Select locations with cost <= min_cost + alpha * (max_cost - min_cost)
            min_c = np.min(costs)
            max_c = np.max(costs)
            threshold = min_c + alpha * (max_c - min_c)
            
            rcl_indices = np.where(costs <= threshold)[0]
            # Pick one randomly from RCL
            chosen_idx = rcl_indices[np.random.randint(len(rcl_indices))]
            l = unassigned_l[chosen_idx]
            
            # Assign i to l
            p[i] = l
            assigned_f[i] = True
            assigned_l[l] = True
            
            # Update current cost
            current_cost += cost_to_loc[i, l]
            
            # Update cost_to_loc for remaining unassigned facilities
            # For all unassigned k:
            # cost_to_loc[k, :] += a[k,i]*b[l,:] + a[i,k]*b[:,l]
            # cost_to_loc[k, l] += a[k,k]*b[l,l]
            
            remaining_f = np.where(~assigned_f)[0]
            if len(remaining_f) > 0:
                # Vectorized update
                # a[remaining_f, i] is shape (m, 1)
                # b[l, :] is shape (n,)
                # Outer product: (m, n)
                add1 = np.outer(a_f[remaining_f, i], b_f[l, :])
                add2 = np.outer(a_f[i, remaining_f], b_f[:, l])
                
                cost_to_loc[remaining_f, :] += add1 + add2
                
                # Diagonal term
                diag_val = a_f[remaining_f, remaining_f] * b_f[l, l]
                # Add to cost_to_loc[k, l]
                cost_to_loc[remaining_f, l] += diag_val
        
        return p, current_cost

    # 1. Initial Heuristic: Rank Matching
    a_row_sums = a_f.sum(axis=1)
    b_row_sums = b_f.sum(axis=1)
    
    heavy_order = np.argsort(-a_row_sums, kind="stable")
    central_order = np.argsort(b_row_sums, kind="stable")
    
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy_order] = central_order
    
    best_p = p0.copy()
    best_cost = compute_cost(best_p)
    
    # Local search from rank matching
    p_ls, cost_ls = local_search(p0.copy(), best_cost, deadline - 10)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()

    # 2. Spectral / Linear Assignment Initialization
    if time.time() < deadline - 10:
        A = a_f @ b_f.T
        row_ind, col_ind = linear_sum_assignment(-A)
        p_spectral = np.empty(n, dtype=np.int64)
        p_spectral[row_ind] = col_ind
        
        cost_spectral = compute_cost(p_spectral)
        if cost_spectral < best_cost:
            best_cost = cost_spectral
            best_p = p_spectral.copy()
        
        p_spectral_ls, cost_spectral_ls = local_search(p_spectral.copy(), cost_spectral, deadline - 10)
        if cost_spectral_ls < best_cost:
            best_cost = cost_spectral_ls
            best_p = p_spectral_ls.copy()

    # 3. GRASP Phase
    # Use remaining time for GRASP restarts
    # Each GRASP run: Construct + Local Search
    # We need to estimate time per run.
    # Construction is O(n^3) roughly? No, O(n^2) updates per step, n steps -> O(n^3).
    # n=150, n^3 = 3.3M. Very fast.
    # Local search is the bottleneck.
    
    while time.time() < deadline - 5:
        # Randomize alpha
        # alpha in [0, 1]. Lower alpha is more greedy.
        alpha = np.random.uniform(0.1, 0.5)
        
        p_grasp, cost_grasp = grasp_construct(alpha)
        
        # Local search
        time_left_for_ls = deadline - 5 - time.time()
        if time_left_for_ls < 2:
            break
            
        p_grasp_ls, cost_grasp_ls = local_search(p_grasp, cost_grasp, time.time() + time_left_for_ls)
        
        if cost_grasp_ls < best_cost:
            best_cost = cost_grasp_ls
            best_p = p_grasp_ls.copy()

    return best_p.tolist()
