"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 145.0  # Leave 5s safety margin for 150s limit

    # Ensure arrays are int64 for exact integer arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    # 1. Initial Solution: Hungarian Algorithm on Linearized 3D Assignment
    # We approximate the QAP cost by fixing the ordering of facilities and locations.
    # Let r_i be the rank of facility i by "importance" (sum of flows).
    # Let c_j be the rank of location j by "centrality" (sum of distances).
    # We want to assign facility i to location p[i].
    # The cost is sum a[i,j] b[p[i], p[j]].
    # If we assume p is monotonic with respect to ranks, we can simplify.
    # A better heuristic:
    # Construct a cost matrix C where C[i, j] approximates the contribution of assigning facility i to location j.
    # A common linearization is C[i, j] = sum_k a[i, k] * b[j, rank(k)] where rank(k) is the location assigned to k.
    # This is circular.
    
    # Alternative: Use the "rank matching" but optimized via Hungarian.
    # Sort facilities by row sum of A (descending) -> heavy facilities.
    # Sort locations by row sum of B (ascending) -> central locations.
    # The "ideal" assignment is heavy -> central.
    # However, simple sorting ignores interactions.
    
    # Let's use a spectral initialization which is often superior for QAP.
    # But the prompt specifically suggests: "Formulate the QAP as a 3D assignment problem where the objective is linearized by fixing the relative ordering... then solve ... using Hungarian".
    
    # Interpretation:
    # 1. Compute a "linear" cost matrix L for the assignment.
    #    L[i, j] = sum_k a[i, k] * b[j, q(k)] 
    #    where q is an initial guess (e.g., identity or random).
    #    This is effectively one step of the "linearization" or "rank-based" heuristic.
    #    Actually, a standard heuristic is:
    #    Let A_sorted be A sorted by row sums.
    #    Let B_sorted be B sorted by row sums.
    #    The cost of assigning facility i to location j can be approximated by the "interaction" with the "average" or "ranked" partner.
    
    # Let's try a specific "Rank-Based" Linear Assignment:
    # 1. Compute row sums of A: a_sum[i]
    # 2. Compute row sums of B: b_sum[j]
    # 3. The "main" cost driver is matching high a_sum to low b_sum.
    # 4. To capture second-order effects, we can define a cost matrix M for the Hungarian algorithm:
    #    M[i, j] = a_sum[i] * b_sum[j]
    #    This is a rank-1 approximation. Minimizing this via Hungarian (which is just sorting in this case) gives the rank matching.
    
    # To get something "different" and potentially better, we can use the "spectral" initialization which is a known strong heuristic for QAP, or a "linearized" cost based on the current best guess.
    
    # Let's stick to the prompt's suggestion: "linearized by fixing the relative ordering".
    # This implies we assume the optimal permutation preserves the order of some feature.
    # Feature 1: Facility importance (row sum of A).
    # Feature 2: Location centrality (row sum of B).
    
    # Let's implement a "Greedy + Hungarian" hybrid or just a strong Spectral initialization, as it's a different family from simple rank matching.
    # However, the prompt *explicitly* asks for the Hungarian strategy.
    
    # Strategy:
    # 1. Start with a random or rank-based initial permutation p0.
    # 2. Construct a linear cost matrix C for the Hungarian algorithm:
    #    C[i, j] = sum_k a[i, k] * b[j, p0[k]]
    #    This approximates the QAP cost if the assignment of other facilities (k) remains p0[k].
    # 3. Solve the Linear Assignment Problem (LAP) using Hungarian to get a new permutation p1.
    # 4. This is essentially one iteration of the "Linearization" or "Rank-based" heuristic.
    # 5. We can iterate this a few times (Linearized QAP / LQAP).
    
    # Initialization: Rank Matching
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    heavy_idx = np.argsort(-a_sums, kind="stable")
    central_idx = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx
    
    # Iterative Linearization (LQAP)
    # This is a known heuristic: repeatedly solve the LAP with costs linearized around the current solution.
    for _ in range(10):  # Fixed number of iterations, cheap
        if time.time() - start_time > time_limit:
            break
        # Construct cost matrix C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # C = A @ B_p^T ? No.
        # C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # Let B_p be the matrix where B_p[k, :] = b[p[k], :].
        # Then (A @ B_p.T)[i, j] = sum_k A[i, k] * B_p.T[k, j] = sum_k A[i, k] * B_p[j, k] = sum_k A[i, k] * b[p[k], j]
        # We want b[j, p[k]]. Since B is symmetric (QAP distance matrix is symmetric), b[j, p[k]] = b[p[k], j].
        # So C = A @ B_p.T
        B_p = b[p]  # B_p[k] = b[p[k]]
        C = a @ B_p.T
        
        # Solve LAP
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        # Check if improved
        cost_old = np.sum(a * b[p])
        cost_new = np.sum(a * b[p_new])
        
        if cost_new < cost_old:
            p = p_new
        else:
            # If not improved, maybe break or continue?
            # LQAP can oscillate. Let's just take the best found so far.
            pass
        
        if time.time() - start_time > time_limit:
            break

    best_cost = np.sum(a * b[p])
    best_p = p.copy()

    # 2. Local Search: 2-Opt (Pairwise Swap)
    # Now refine with local search.
    
    def get_delta(i, j, p, a, b):
        u = p[i]
        v = p[j]
        diff_a = a[i] - a[j]
        diff_b = b[v, p] - b[u, p]
        delta = np.dot(diff_a, diff_b) + 2 * a[i, j] * b[u, v]
        return delta

    max_passes = 10000
    pass_count = 0
    # Use first-improvement strategy with randomized restarts if stuck
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        # Randomize order of pairs to avoid local bias?
        # Let's do sequential for speed, but if stuck, we perturb.
        
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                delta = get_delta(i, j, p, a, b)
                
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    best_cost += delta
                    improved = True
                    
                    if best_cost < np.sum(a * b[p]): # This check is redundant if we track cost correctly, but safe?
                         # Actually, we update best_cost directly.
                         pass
                         
                    # We don't restart the inner loop immediately in the tight loop for speed,
                    # but we break out to re-scan?
                    # Standard 2-opt often scans the whole list.
                    # Breaking early (first improvement) is faster but can be less effective.
                    # Let's continue scanning for this pass to find multiple improvements?
                    # No, swapping changes the state for subsequent j's.
                    # So we must restart or be careful.
                    # Simplest: break and restart the outer loop? That's O(n^3).
                    # Let's just break the inner loop and continue outer.
                    break
            if improved:
                break
        
        if not improved:
            # Local optimum reached
            elapsed = time.time() - start_time
            if elapsed > time_limit - 10:
                break
            
            # Perturbation: 2-swap or 3-swap
            # Pick 2 random indices and swap
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                p[i], p[j] = p[j], p[i]
                # Recalculate cost from scratch for safety after perturbation
                cost = np.sum(a * b[p])
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
                # Continue loop to improve from here
                continue
            else:
                break

    return best_p.tolist()

# EVOLVE-BLOCK-END
