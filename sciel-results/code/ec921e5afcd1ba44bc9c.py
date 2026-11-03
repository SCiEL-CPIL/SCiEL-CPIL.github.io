import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Check if B is symmetric
    is_sym = np.allclose(b, b.T)
    
    # Initial assignment: Rank matching
    # Heavy flows (high row sum in A) to central locations (low row sum in B)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by decreasing flow sum
    heavy_order = np.argsort(-a_sums, kind='stable')
    # Sort locations by increasing distance sum (central)
    central_order = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    # Calculate initial cost
    def calc_cost(p):
        # b_p[i, j] = b[p[i], p[j]]
        # cost = sum(a[i, j] * b[p[i], p[j]])
        b_perm = b[p][:, p]
        return np.sum(a * b_perm)
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Hungarian-based Local Search
    # We iterate:
    # 1. Construct linear cost matrix C based on current p
    # 2. Solve assignment problem for C
    # 3. Update p
    # 4. Repeat until no improvement or time limit
    
    for _ in range(200):  # Max iterations, will break on time
        if time.time() > deadline:
            break
            
        # Construct C
        # C[i, k] = sum_j a[i, j] b[k, p[j]] + sum_j a[j, i] b[p[j], k]
        # If B is symmetric: C = (A + A.T) @ B_p where B_p[i, :] = B[p[i], :]
        # If B is not symmetric: C = A @ B_p.T + A.T @ B_p
        
        B_p = b[p]  # Rows of B permuted by p. B_p[i, k] = B[p[i], k]
        
        if is_sym:
            A_sym = a + a.T
            C = A_sym @ B_p
        else:
            C = a @ B_p.T + a.T @ B_p
            
        # Solve linear assignment
        row_ind, col_ind = linear_sum_assignment(C)
        
        # New permutation
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        # Evaluate new cost
        new_cost = calc_cost(p_new)
        
        if new_cost < current_cost:
            current_cost = new_cost
            p = p_new
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # No improvement, we might try a small perturbation or break
            # To escape local optima of the linearization, we can break
            # or try a random restart of the linearization weights?
            # For now, break to save time for local search or return
            # Since linearization is convex, if it doesn't improve, we are likely stuck
            # in a local optimum of the QAP relative to this linearization.
            # We can try a few random 2-swaps to perturb p before next linearization
            improved = False
            for _ in range(10):
                i, j = np.random.choice(n, 2, replace=False)
                p_swap = p.copy()
                p_swap[i], p_swap[j] = p_swap[j], p_swap[i]
                swap_cost = calc_cost(p_swap)
                if swap_cost < current_cost:
                    p = p_swap
                    current_cost = swap_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break
            if not improved:
                break
                
    # Final local search polish (2-opt) if time remains
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Vectorized 2-opt evaluation is complex, so we do a simple loop
            # but limit the number of passes
            # n=256, n^2/2 = 32640 pairs.
            # We can't do all in Python loop fast enough if not optimized.
            # Let's do a random subset of swaps or a single pass of best-improvement
            # To keep it simple and safe, we'll do a limited number of random swap attempts
            # or a greedy pass.
            
            # Greedy 2-opt pass
            # This might be slow in pure Python. 
            # Let's stick to the Hungarian result which is usually good.
            # If we have time, a few random restarts of the Hungarian process might be better.
            
            # Let's try a simple random restart of the whole process if we have lots of time?
            # No, the deadline is strict.
            pass

    return best_p.tolist()
