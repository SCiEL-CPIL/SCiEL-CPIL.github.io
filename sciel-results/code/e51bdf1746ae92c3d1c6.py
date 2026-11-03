"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    # Safety margin: 120s limit, aim to finish by 115s
    deadline = time.time() + 115.0

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    # Precompute transpose of a for column access
    a_t = a.T

    def swap_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        # Vectorized delta calculation
        # term1 = sum_k (a[i, k] - a[j, k]) * (b[v, p[k]] - b[u, p[k]])
        # term2 = sum_k (a[k, i] - a[k, j]) * (b[p[k], v] - b[p[k], u])
        
        diff_a_rows = a[i, :] - a[j, :]
        diff_b_rows = b[v, p] - b[u, p]
        term1 = np.dot(diff_a_rows, diff_b_rows)
        
        diff_a_cols = a_t[:, i] - a_t[:, j]
        diff_b_cols = b[p, v] - b[p, u]
        term2 = np.dot(diff_a_cols, diff_b_cols)
        
        return term1 + term2

    def local_search(p, current_cost, deadline, max_iter=500):
        improved = True
        iter_count = 0
        
        while improved and time.time() < deadline - 2.0 and iter_count < max_iter:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Scan all pairs to find the steepest descent move
            for i in range(n - 1):
                if time.time() >= deadline - 2.0:
                    return p, current_cost
                for j in range(i + 1, n):
                    if time.time() >= deadline - 2.0:
                        return p, current_cost
                    d = swap_delta(i, j, p)
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

    # 1. Initialization strategies
    
    # Strategy A: Rank-matching
    def init_rank():
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    # Strategy B: Greedy Construction
    def init_greedy():
        p = np.full(n, -1, dtype=np.int64)
        flow_sums = a.sum(axis=1)
        order = np.argsort(-flow_sums, kind='stable')
        assigned = np.zeros(n, dtype=bool)
        
        for idx in order:
            best_loc = -1
            best_inc = np.inf
            inter_vec = a[idx, :] + a[:, idx]
            
            for loc in range(n):
                if assigned[loc]:
                    continue
                assigned_indices = np.where(assigned)[0]
                if len(assigned_indices) == 0:
                    inc = 0.0
                else:
                    inc = np.dot(inter_vec[assigned_indices], b[loc, p[assigned_indices]])
                
                if inc < best_inc:
                    best_inc = inc
                    best_loc = loc
            
            p[idx] = best_loc
            assigned[best_loc] = True
            
        return p

    # Strategy C: Hungarian-based Init (Linear Assignment Relaxation)
    def init_hungarian():
        try:
            from scipy.optimize import linear_sum_assignment
            # C[i, loc] = sum_j (a[i,j] + a[j,i]) * b[loc, j]
            # This approximates the pairwise interaction by summing over all j
            # Note: b[loc, j] is the distance from location 'loc' to 'j'.
            # We want to assign facility i to location loc.
            # The term involving i and j is a[i,j]*b[loc_i, loc_j] + a[j,i]*b[loc_j, loc_i].
            # Summing over j, the cost contribution for facility i at loc_i is approx:
            # sum_j (a[i,j] + a[j,i]) * b[loc_i, j] (assuming loc_j ~ j for the average case, or just using the raw matrix)
            
            # Let's construct the cost matrix for the linear assignment.
            # Cost[i, loc] = sum_j (a[i,j] + a[j,i]) * b[loc, j]
            
            # a[i,j] + a[j,i] is the total flow involving i and j.
            # Let flow_i = a[i, :] + a[:, i]
            # Cost[i, loc] = dot(flow_i, b[loc, :])
            
            # flow is n x n
            flow = a + a.T
            # b is n x n
            # We want Cost[i, loc] = sum_j flow[i, j] * b[loc, j]
            # This is (flow @ b.T) ? No.
            # flow[i, j] * b[loc, j]. Sum over j.
            # This is equivalent to (flow @ b.T)[i, loc] ?
            # (flow @ b.T)[i, k] = sum_j flow[i, j] * b.T[j, k] = sum_j flow[i, j] * b[k, j].
            # Yes. So Cost = flow @ b.T
            
            cost_mat = flow @ b.T
            
            # Solve linear assignment
            row_ind, col_ind = linear_sum_assignment(cost_mat)
            p = np.empty(n, dtype=np.int64)
            # row_ind is facility, col_ind is location
            p[row_ind] = col_ind
            return p
        except Exception:
            # Fallback to rank if scipy fails
            return init_rank()

    # 2. Main Search Loop
    
    best_cost = np.inf
    best_p = None
    
    # Generate initial solutions
    inits = []
    
    # 1. Rank
    p_rank = init_rank()
    inits.append(p_rank)
    
    # 2. Greedy
    p_greedy = init_greedy()
    inits.append(p_greedy)
    
    # 3. Hungarian
    p_hung = init_hungarian()
    inits.append(p_hung)
    
    # 4. Random
    rng = np.random.RandomState(42)
    inits.append(rng.permutation(n))
    
    # Evaluate and refine each initial solution
    for p_init in inits:
        if time.time() >= deadline - 5.0:
            break
            
        p = p_init.copy()
        current_cost = compute_cost(p)
        
        # Run local search
        p, current_cost = local_search(p, current_cost, deadline)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    # If no valid best_p found (shouldn't happen), fallback
    if best_p is None:
        best_p = inits[0].copy()
        best_cost = compute_cost(best_p)

    # 3. Iterated Local Search (ILS)
    # Use random perturbations to escape local minima
    # We keep the best solution found so far and perturb it
    
    # Set a safety margin for the final return to ensure we don't crash
    while time.time() < deadline - 5.0: 
        # Perturb the best solution
        p = best_p.copy()
        current_cost = best_cost
        
        # Perform random swaps
        # Number of swaps: typically between 5 and 15 for n=150
        num_swaps = np.random.randint(5, 16)
        for _ in range(num_swaps):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        # Compute cost of perturbed solution
        current_cost = compute_cost(p)
        
        # Run local search from the perturbed state
        p, current_cost = local_search(p, current_cost, deadline)
        
        # Update best if improved
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # If time is running out, we might want to stop early to ensure clean exit
        if time.time() > deadline - 10.0:
            break

    # Final verification
    final_cost = compute_cost(best_p)
    if final_cost < best_cost:
        best_cost = final_cost

    return best_p.tolist()

# EVOLVE-BLOCK-END
