"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute row sums for fast cost evaluation
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    def compute_cost(p):
        """Compute the QAP cost for permutation p."""
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # = sum_i (a[i,:] . b[p[i], :])
        # Vectorized: for each i, we need b[p[i], :] dotted with a[i, :]
        # This is equivalent to: sum over i of a[i, :] @ b[p[i], :]
        # We can do this as: (a * b[np.ix_(p, p)]).sum() but that's O(n^2) memory
        # Better: use matrix multiplication
        # cost = trace(a^T @ b[p][:, p]) ... let me think
        # Actually: sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Let's create a matrix M where M[i,j] = b[p[i], p[j]]
        # Then cost = sum(a * M)
        # M = b[np.ix_(p, p)]
        # This is O(n^2) memory which for n=150 is 22500, fine.
        return int(np.sum(a * b[np.ix_(p, p)]))

    def compute_swap_delta(p, i, j):
        """Compute the change in cost if we swap positions i and j in permutation p."""
        # Current: a[i,:] contributes with b[p[i],:], a[j,:] with b[p[j],:]
        # After swap: a[i,:] contributes with b[p[j],:], a[j,:] with b[p[i],:]
        # Also, all other k != i,j: a[k,i]*b[p[k],p[i]] becomes a[k,i]*b[p[k],p[j]], etc.
        
        pi = p[i]
        pj = p[j]
        
        # Change from row i: a[i,:] . (b[pj,:] - b[pi,:])
        delta_i = np.dot(a[i, :], b[pj, :] - b[pi, :])
        # Change from row j: a[j,:] . (b[pi,:] - b[pj,:])
        delta_j = np.dot(a[j, :], b[pi, :] - b[pj, :])
        
        # Change from columns i and j for all other rows k
        # For each k != i, j: a[k,i]*b[p[k],pi] + a[k,j]*b[p[k],pj] -> a[k,i]*b[p[k],pj] + a[k,j]*b[p[k],pi]
        # Delta for column i: a[:,i] . (b[p, pj] - b[p, pi]) for k != i,j
        # Delta for column j: a[:,j] . (b[p, pi] - b[p, pj]) for k != i,j
        
        # b[p, pi] is the vector [b[p[0], pi], b[p[1], pi], ..., b[p[n-1], pi]]
        # = b[:, pi][p]
        
        bp_pi = b[:, pi][p]  # shape (n,)
        bp_pj = b[:, pj][p]  # shape (n,)
        
        diff = bp_pj - bp_pi  # shape (n,)
        
        # For column i: sum over k != i,j of a[k,i] * diff[k]
        col_i_delta = np.dot(a[:, i], diff) - a[i, i] * diff[i] - a[j, i] * diff[j]
        # For column j: sum over k != i,j of a[k,j] * (-diff[k])
        col_j_delta = -np.dot(a[:, j], diff) + a[i, j] * diff[i] + a[j, j] * diff[j]
        
        total_delta = delta_i + delta_j + col_i_delta + col_j_delta
        return int(total_delta)

    def local_search(p, time_limit):
        """2-swap local search. Returns improved permutation."""
        p = np.array(p, dtype=np.int64, copy=True)
        current_cost = compute_cost(p)
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            n = len(p)
            # Try all pairs, but we can be smarter
            for i in range(n):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, n):
                    delta = compute_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        # After a swap, restart the scan from beginning to find more improvements
                        break
                if improved:
                    break
        
        return p, current_cost

    # GRASP construction: randomized greedy
    def grasp_construct(rng, alpha=0.3):
        """Construct a solution using GRASP with restricted candidate list."""
        p = np.full(n, -1, dtype=np.int64)
        assigned = np.zeros(n, dtype=bool)
        # p[i] = location for facility i
        # We assign facilities one by one to locations
        
        # Precompute for fast evaluation
        # When assigning facility i to location loc, the marginal cost is:
        # sum_j a[i,j] * b[loc, p[j]] + sum_j a[j,i] * b[p[j], loc]
        # + b[loc, loc] * a[i,i] (but this is already included if j=i... let's be careful)
        
        # Actually, the full cost is sum_{i,j} a[i,j] * b[p[i], p[j]]
        # When we assign facility i to location loc, the new terms involving i are:
        # sum_j a[i,j] * b[loc, p[j]] + sum_j a[j,i] * b[p[j], loc]
        # But we need to be careful not to double count a[i,i]*b[loc,loc]
        
        # Let's maintain a running sum of the "cross" terms
        # For each unassigned facility i, compute the cost of assigning it to each location
        
        # Vectorized approach:
        # For facility i, cost of assigning to location loc:
        # c(i, loc) = sum_j a[i,j] * b[loc, p[j]] + sum_j a[j,i] * b[p[j], loc]
        #            = b[loc, :] @ (a[i, :] assigned part) ... this is getting complex
        
        # Simpler: just compute it directly for each candidate
        # n=150, so 150 facilities * 150 locations = 22500 evaluations per step, 150 steps
        # That's 3.375 million dot products of length up to 150. Might be slow in Python.
        
        # Let's use a vectorized approach:
        # Maintain matrix B_assigned where B_assigned[loc, j] = b[loc, p[j]] if j assigned, else 0
        # And A_col where A_col[i, j] = a[i, j]
        
        # For facility i, cost of assigning to location loc:
        # = sum_j a[i,j] * b[loc, p[j]] + sum_j a[j,i] * b[p[j], loc]
        # = b[loc, :] @ (A_assigned[:, i]) where A_assigned[j, i] = a[j, i] if j assigned
        #   + (b[:, loc] @ (a[i, :])) where only assigned j contribute
        
        # Let's maintain:
        # Bp: shape (n, n), Bp[loc, j] = b[loc, p[j]] if p[j] != -1 else 0
        # Then for facility i:
        # term1 = Bp[loc, :] @ a[i, :] ... no wait
        # sum_j a[i,j] * b[loc, p[j]] = sum_j a[i,j] * Bp[loc, j] = a[i, :] @ Bp[loc, :]
        
        # sum_j a[j,i] * b[p[j], loc] = sum_j a[j,i] * Bp[p[j], loc] ... this is trickier
        # Let's define BpT: BpT[j, loc] = b[p[j], loc] if p[j] != -1 else 0
        # Then sum_j a[j,i] * BpT[j, loc] = a[:, i] @ BpT[:, loc]
        
        # So cost(i, loc) = a[i, :] @ Bp[loc, :] + a[:, i] @ BpT[:, loc]
        
        # We can vectorize over loc:
        # costs_for_i = Bp @ a[i, :] + BpT.T @ a[:, i]
        # Wait: Bp @ a[i, :] gives vector of shape (n,) where entry loc = Bp[loc, :] @ a[i, :]
        # BpT.T @ a[:, i] gives vector of shape (n,) where entry loc = BpT[:, loc] @ a[:, i]
        
        Bp = np.zeros((n, n), dtype=np.float64)   # Bp[loc, j] = b[loc, p[j]] if assigned
        BpT = np.zeros((n, n), dtype=np.float64)  # BpT[j, loc] = b[p[j], loc] if assigned
        
        for step in range(n):
            # Compute cost of assigning each unassigned facility to each available location
            # Only consider unassigned facilities
            unassigned = np.where(~assigned)[0]
            available_locs = np.where(~assigned)[0]  # same set since it's a permutation
            
            # For each unassigned facility i, compute costs to all available locations
            # costs[i_idx, loc_idx] for i in unassigned, loc in available_locs
            
            # Vectorized:
            # For facility i: cost to location loc = a[i,:] @ Bp[loc,:] + a[:,i] @ BpT[:,loc]
            # = (Bp @ a[i,:])[loc] + (BpT.T @ a[:,i])[loc]
            
            # Bp @ a[i,:] for all i at once: (Bp @ a.T) gives shape (n, n) where [loc, i] = Bp[loc,:] @ a[i,:]
            term1 = Bp @ a.T  # shape (n, n), term1[loc, i] = sum_j Bp[loc,j] * a[i,j]
            
            # BpT.T @ a[:, i] for all i: (BpT.T @ a) gives shape (n, n) where [loc, i] = BpT[:,loc] @ a[:,i]
            term2 = BpT.T @ a  # shape (n, n), term2[loc, i] = sum_j BpT[j,loc] * a[j,i]
            
            # Total cost matrix: cost[loc, i] = term1[loc, i] + term2[loc, i]
            cost_mat = term1 + term2  # shape (n, n)
            
            # Only consider unassigned facilities and available locations
            # cost_mat[loc, i] for loc in available_locs, i in unassigned
            sub_cost = cost_mat[np.ix_(available_locs, unassigned)]
            
            # GRASP: select from restricted candidate list
            min_cost = sub_cost.min()
            max_cost = sub_cost.max()
            threshold = min_cost + alpha * (max_cost - min_cost)
            
            # Find candidates below threshold
            candidate_mask = sub_cost <= threshold
            candidate_indices = np.where(candidate_mask)
            
            # Randomly select one candidate
            num_candidates = len(candidate_indices[0])
            idx = rng.integers(0, num_candidates)
            loc_idx = candidate_indices[0][idx]
            fac_idx = candidate_indices[1][idx]
            
            loc = available_locs[loc_idx]
            fac = unassigned[fac_idx]
            
            p[fac] = loc
            assigned[fac] = True
            assigned[loc] = True
            
            # Update Bp and BpT
            Bp[loc, fac] = b[loc, fac]  # Wait, Bp[loc, j] should be b[loc, p[j]] = b[loc, fac]
            # Actually Bp[loc, j] = b[loc, p[j]], and p[fac] = loc, so Bp[loc, fac] = b[loc, loc]? No.
            # Bp is indexed as Bp[loc, j] where j is the facility index.
            # Bp[loc, j] = b[loc, p[j]] if facility j is assigned.
            # So Bp[loc, fac] = b[loc, p[fac]] = b[loc, loc]
            # That doesn't seem right. Let me re-think.
            
            # I think I confused myself. Let me redefine:
            # Bp[loc, j] = b[loc, p[j]] if facility j is assigned, else 0
            # When we assign facility fac to location loc, p[fac] = loc
            # So for all locations loc2, Bp[loc2, fac] = b[loc2, loc]
            # And for all facilities j, BpT[fac, loc2] = b[loc, loc2]
            
            Bp[:, fac] = b[:, loc]  # For all locations, Bp[loc2, fac] = b[loc2, loc]
            BpT[fac, :] = b[loc, :]  # For all locations, BpT[fac, loc2] = b[loc, loc2]
        
        return p

    # Initial solution using rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central
    p0, cost0 = local_search(p0, start_time + 20)
    
    best_p = p0.copy()
    best_cost = cost0
    
    rng = np.random.default_rng(42)
    
    # GRASP iterations
    while time.time() < deadline - 5:
        p_grasp = grasp_construct(rng, alpha=0.3)
        p_ls, cost_ls = local_search(p_grasp, time.time() + 10)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
        
        if time.time() > deadline - 5:
            break
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
