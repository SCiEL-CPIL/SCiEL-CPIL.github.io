"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit (150s kill)
    
    # Helper: Calculate exact cost
    def calc_cost(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))
    
    # Helper: Calculate delta for swapping facilities i and j in permutation p
    # This is O(n) if we precompute some things, but here we do O(n) per swap
    # For n=256, O(n) is 256 operations, which is very fast in numpy.
    def swap_delta(p, i, j):
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0
        
        # Cost change = sum_k (a[i,k] - a[j,k]) * (b[lj, p[k]] - b[li, p[k]])
        # Note: The standard formula for symmetric QAP is:
        # delta = 2 * ( sum_{k != i,j} (a[i,k] - a[j,k]) * (b[lj, p[k]] - b[li, p[k]]) )
        #       + (a[i,i] - a[j,j]) * (b[lj, lj] - b[li, li])
        #       + (a[i,j] - a[j,i]) * (b[lj, li] - b[li, lj])  -- this term is 0 if symmetric a and b?
        # Let's use the robust O(n) formula derived from expanding the cost function.
        
        # Let C(p) = sum_{u,v} a[u,v] b[p[u], p[v]]
        # Swapping i and j changes p[i] and p[j].
        # Terms involving i or j:
        # 1. a[i,i] b[p[i], p[i]] -> a[i,i] b[lj, lj]
        # 2. a[j,j] b[p[j], p[j]] -> a[j,j] b[li, li]
        # 3. a[i,j] b[p[i], p[j]] -> a[i,j] b[lj, li]
        # 4. a[j,i] b[p[j], p[i]] -> a[j,i] b[li, lj]
        # 5. sum_{k != i,j} a[i,k] b[p[i], p[k]] -> sum_{k != i,j} a[i,k] b[lj, p[k]]
        # 6. sum_{k != i,j} a[j,k] b[p[j], p[k]] -> sum_{k != i,j} a[j,k] b[li, p[k]]
        # 7. sum_{k != i,j} a[k,i] b[p[k], p[i]] -> sum_{k != i,j} a[k,i] b[p[k], lj]
        # 8. sum_{k != i,j} a[k,j] b[p[k], p[j]] -> sum_{k != i,j} a[k,j] b[p[k], li]
        
        # If a and b are symmetric:
        # a[i,j] = a[j,i], b[lj,li] = b[li,lj]
        # Term 3 + 4 = 2 * a[i,j] * b[lj, li]
        # Old Term 3 + 4 = 2 * a[i,j] * b[li, lj]
        # Change = 2 * a[i,j] * (b[lj, li] - b[li, lj]) = 0 if b symmetric.
        
        # Terms 5+7 combine for a symmetric a:
        # sum_{k != i,j} (a[i,k] + a[k,i]) * (b[lj, p[k]] - b[li, p[k]]) 
        # = 2 * sum_{k != i,j} a[i,k] * (b[lj, p[k]] - b[li, p[k]])
        
        # Terms 6+8 combine for a symmetric a:
        # sum_{k != i,j} (a[j,k] + a[k,j]) * (b[li, p[k]] - b[lj, p[k]])
        # = 2 * sum_{k != i,j} a[j,k] * (b[li, p[k]] - b[lj, p[k]])
        # = -2 * sum_{k != i,j} a[j,k] * (b[lj, p[k]] - b[li, p[k]])
        
        # Total off-diagonal change:
        # 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[lj, p[k]] - b[li, p[k]])
        
        # Diagonal change:
        # a[i,i] * (b[lj, lj] - b[li, li]) + a[j,j] * (b[li, li] - b[lj, lj])
        # = (a[i,i] - a[j,j]) * (b[lj, lj] - b[li, li])
        
        # So, delta = (a[i,i] - a[j,j]) * (b[lj, lj] - b[li, li]) 
        #            + 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[lj, p[k]] - b[li, p[k]])
        
        # Vectorized computation for the sum part:
        # Let diff_a = a[i, :] - a[j, :]
        # Let diff_b = b[lj, p] - b[li, p]
        # Sum over all k is dot(diff_a, diff_b)
        # We need to exclude k=i and k=j.
        
        diff_a = a[i] - a[j]
        diff_b = b[lj, p] - b[li, p]
        
        total_dot = np.dot(diff_a, diff_b)
        
        # Subtract term for k=i:
        # diff_a[i] * diff_b[i] = (a[i,i] - a[j,i]) * (b[lj, p[i]] - b[li, p[i]])
        # p[i] = li
        # = (a[i,i] - a[j,i]) * (b[lj, li] - b[li, li])
        
        # Subtract term for k=j:
        # diff_a[j] * diff_b[j] = (a[i,j] - a[j,j]) * (b[lj, p[j]] - b[li, p[j]])
        # p[j] = lj
        # = (a[i,j] - a[j,j]) * (b[lj, lj] - b[li, lj])
        
        term_i = (a[i, i] - a[j, i]) * (b[lj, li] - b[li, li])
        term_j = (a[i, j] - a[j, j]) * (b[lj, lj] - b[li, lj])
        
        off_diag_sum = total_dot - term_i - term_j
        
        diag_delta = (a[i, i] - a[j, j]) * (b[lj, lj] - b[li, li])
        
        return int(diag_delta + 2 * off_diag_sum)

    # Initial Solution: Linear Assignment on Row Sums
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    cost_matrix = np.outer(a_row_sums, b_row_sums)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    p_init = np.empty(n, dtype=np.int64)
    p_init[row_ind] = col_ind
    
    # Multi-Start Strategy
    # We will maintain a list of "islands" or solutions.
    # Each solution undergoes local search (2-opt).
    # If stuck, we perturb it (restart).
    
    best_p = p_init.copy()
    best_cost = calc_cost(best_p)
    
    # We run multiple independent local searches in parallel (sequentially in Python, but logically independent)
    # Or just one strong local search with restarts?
    # Given 115s, we can afford a few restarts.
    
    # Let's do 5 restarts.
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    current_p = p_init.copy()
    current_cost = best_cost
    
    for restart_idx in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # If not the first restart, perturb the current best
        if restart_idx > 0:
            # Perturb: Swap K random pairs
            # K proportional to n, e.g., n/10
            num_swaps = max(1, n // 10)
            for _ in range(num_swaps):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    current_p[i], current_p[j] = current_p[j], current_p[i]
            current_cost = calc_cost(current_p)
            
            # Update best if perturbation accidentally improved (unlikely but possible)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
        
        # Local Search: 2-Opt (Best Improvement or First Improvement?)
        # First Improvement is faster, Best Improvement is better per step.
        # Let's use First Improvement with a limit on passes.
        
        # To avoid getting stuck in a cycle, we can use a simple tabu or just randomize the order.
        # Randomizing the order of i,j pairs in each pass helps.
        
        max_passes = 50  # Limit passes per restart to allow time for other restarts
        
        for pass_num in range(max_passes):
            if time.time() > deadline - 5.0:
                break
                
            improved = False
            # Randomize the order of pairs to explore different directions
            # Generating all pairs is O(n^2) = 32k pairs.
            # We can just iterate i, j but start at a random offset?
            # Or shuffle the list of pairs? That's expensive.
            # Let's just iterate i from 0 to n-1, j from 0 to n-1, i != j.
            # To break symmetry, we can randomize the starting index for i.
            
            start_i = random.randint(0, n - 1)
            
            for k in range(n):
                i = (start_i + k) % n
                if time.time() > deadline - 5.0:
                    break
                    
                u = current_p[i]
                
                start_j = random.randint(0, n - 1)
                for m in range(n):
                    j = (start_j + m) % n
                    if i == j:
                        continue
                    if time.time() > deadline - 5.0:
                        break
                    
                    delta = swap_delta(current_p, i, j)
                    
                    if delta < 0:
                        # Perform swap
                        v = current_p[j]
                        current_p[i], current_p[j] = v, u
                        current_cost += delta
                        
                        # Update best
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = current_p.copy()
                            
                        improved = True
                        break # First improvement found, restart pass
                        
                if improved:
                    break
                    
            if not improved:
                # Local optimum reached
                break
        
    # Final check
    if sorted(best_p.tolist()) != list(range(n)):
        return list(range(n))
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
