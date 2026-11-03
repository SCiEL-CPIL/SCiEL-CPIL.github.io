import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    # Leave a safety margin to ensure the function returns before the hard kill
    deadline = start_time + 145.0 

    def compute_cost(p):
        # p is a permutation array where p[i] is the location of facility i
        # Cost = sum a[i,j] * b[p[i], p[j]]
        b_sub = b[np.ix_(p, p)]
        return int(np.sum(a * b_sub))

    def get_delta(p, i, j):
        """
        Compute the change in cost if we swap the locations of facilities i and j.
        Let li = p[i] and lj = p[j].
        The cost change is the sum of changes in terms involving i or j.
        """
        li = p[i]
        lj = p[j]
        
        # We want to calculate:
        # C_new - C_old
        # Terms involving i or j in the sum sum_{k,l} a[k,l] b[p[k], p[l]]
        
        # Vectorized approach:
        # Delta = sum_k (a[i,k] - a[j,k]) * (b[lj, p[k]] - b[li, p[k]])
        #       + sum_k (a[k,i] - a[k,j]) * (b[p[k], lj] - b[p[k], li])
        
        # Note: The term k=i and k=j are handled correctly by the differences if we are careful,
        # but standard derivation:
        # Let S = sum_{u,v} a[u,v] b[p[u], p[v]]
        # Swap p[i] <-> p[j].
        # Only terms where u in {i,j} or v in {i,j} change.
        
        # 1. Terms where u=i or u=j (rows of a)
        # Old: sum_k a[i,k] b[li, p[k]] + sum_k a[j,k] b[lj, p[k]]
        # New: sum_k a[i,k] b[lj, p[k]] + sum_k a[j,k] b[li, p[k]]
        # Diff1 = sum_k a[i,k] (b[lj, p[k]] - b[li, p[k]]) + sum_k a[j,k] (b[li, p[k]] - b[lj, p[k]])
        # Diff1 = sum_k (a[i,k] - a[j,k]) * (b[lj, p[k]] - b[li, p[k]])
        
        # 2. Terms where v=i or v=j (cols of a)
        # Old: sum_k a[k,i] b[p[k], li] + sum_k a[k,j] b[p[k], lj]
        # New: sum_k a[k,i] b[p[k], lj] + sum_k a[k,j] b[p[k], li]
        # Diff2 = sum_k a[k,i] (b[p[k], lj] - b[p[k], li]) + sum_k a[k,j] (b[p[k], li] - b[p[k], lj])
        # Diff2 = sum_k (a[k,i] - a[k,j]) * (b[p[k], lj] - b[p[k], li])
        
        # Total Delta = Diff1 + Diff2
        
        diff_a_row = a[i] - a[j]
        diff_b_col = b[lj, p] - b[li, p]
        
        diff_a_col = a[:, i] - a[:, j]
        diff_b_row = b[p, lj] - b[p, li]
        
        return np.dot(diff_a_row, diff_b_col) + np.dot(diff_a_col, diff_b_row)

    def first_fit_local_search(p, current_cost, time_limit):
        """
        Perform First-Fit Local Search.
        Iterates through pairs (i, j) in a random or sequential order.
        If a swap improves the cost, apply it immediately and restart the scan (or continue).
        Restarting is often better for First-Fit to ensure no improving moves are missed due to order dependency.
        """
        improved = True
        # To avoid infinite loops or excessive time in a single LS, we can limit passes or just rely on time.
        # Since we restart on every move, we need to be careful.
        
        # Randomizing the order of checks can help avoid bias
        # However, sequential is faster. Let's try sequential first, but restart index 0 on move.
        
        while improved and time.time() < time_limit:
            improved = False
            # We scan all pairs. If we make a move, we break and restart from 0.
            # This is O(n^2) per move.
            # To speed up, we can just scan and take the first one, then restart.
            
            for i in range(n):
                if time.time() >= time_limit:
                    return p, current_cost
                for j in range(i + 1, n):
                    if time.time() >= time_limit:
                        return p, current_cost
                    delta = get_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += int(delta)
                        improved = True
                        break # Break inner loop
                if improved:
                    break # Break outer loop, restart scan from 0
        return p, current_cost

    def perturb(p, num_swaps):
        """Apply random swaps to p."""
        p_perturbed = p.copy()
        for _ in range(num_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
        return p_perturbed

    # 1. Initial Solution
    # Rank matching heuristic is a good start
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by flow (descending)
    heavy_order = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by distance (ascending) - "central" locations
    central_order = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    current_cost = compute_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Initial Local Search
    # Run a few passes of First-Fit LS to polish the initial solution
    # We allocate a small fraction of time for this
    initial_ls_limit = start_time + 5.0
    p, current_cost = first_fit_local_search(p, current_cost, initial_ls_limit)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
        
    # 3. Iterated Local Search (ILS)
    # Loop until time is up
    # Each iteration: Perturb best solution, run First-Fit LS, update best if improved
    
    # We need to estimate how long a First-Fit LS takes.
    # It depends on how many moves are made.
    # We will set a dynamic time limit for each LS call based on remaining time.
    
    while time.time() < deadline:
        # Check if we have enough time for at least one LS iteration
        # A conservative estimate: if less than 10s left, we might finish one, but risky.
        # Let's say we need at least 2 seconds for a perturbation + LS attempt to be worth it.
        if time.time() > deadline - 2.0:
            break
            
        # Perturb the best solution
        # Number of swaps: usually 2-10 is a good range for QAP
        # Let's vary it slightly
        num_swaps = random.randint(2, 10)
        p_curr = perturb(best_p, num_swaps)
        
        # Calculate cost of perturbed solution
        curr_cost = compute_cost(p_curr)
        
        # Time limit for this specific LS run
        # We want to leave some buffer for the next iteration's overhead
        ls_time_limit = min(deadline - 1.0, time.time() + 10.0) 
        # Actually, just use the global deadline minus a small margin for the loop check
        ls_time_limit = deadline - 1.0
        
        # Run First-Fit Local Search
        p_ls, ls_cost = first_fit_local_search(p_curr, curr_cost, ls_time_limit)
        
        if ls_cost < best_cost:
            best_cost = ls_cost
            best_p = p_ls.copy()
            
        # If we are getting close to the deadline, stop
        if time.time() > deadline - 5.0:
             break

    return best_p.tolist()
