"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Ensure int64 for exact arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    # 1. Initialization: Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy_order = np.argsort(-a_row_sums, kind="stable")
    central_order = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    def calc_cost(p):
        b_perm = b[:, p][:, p]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Tabu parameters
    tabu_tenure = 10
    tabu_list = {} # (i, j) -> iteration when it becomes non-tabu
    iteration = 0
    no_improve_count = 0
    max_no_improve = 50
    max_iterations = 50000 # Hard cap if time is not the bottleneck (unlikely)
    
    # Check symmetry for potential speedups (though we use general formula for safety)
    is_symmetric_a = np.allclose(a, a.T)
    is_symmetric_b = np.allclose(b, b.T)
    
    while time.time() < deadline:
        best_delta = np.inf
        best_i = -1
        best_j = -1
        is_aspiration = False
        
        # We search for the best move
        # To speed up, we can iterate i and vectorize j? 
        # For n=150, O(n^2) loop in Python is slow. 
        # Let's try to vectorize the delta calculation for all j for a given i, or just use a fast inner loop.
        
        # Actually, for n=150, a pure Python loop over 150*150/2 = 11250 pairs per iteration is ~11k iterations.
        # If we do 5000 iterations, that's 50M operations. Might be tight in 115s.
        # Let's optimize the delta calculation.
        
        # Precompute B_p = b[:, p]
        B_p = b[:, p]
        
        # We will iterate i from 0 to n-1
        for i in range(n):
            u = p[i]
            
            # If (i, j) is tabu, we skip unless aspiration
            # We need to check tabu status for each (i, j)
            
            # Vectorized calculation for all j != i is complex due to the specific terms involving i and j.
            # However, we can compute the "bulk" part vectorially.
            
            # Let's stick to a slightly optimized loop.
            # diff_a_row_i = a[i, :]
            # diff_b_row_u = B_p[u, :] # b[u, p[k]]
            
            for j in range(n):
                if i == j:
                    continue
                
                v = p[j]
                
                # Check Tabu
                # We store the move as a sorted tuple (min(i,j), max(i,j))
                move_key = (min(i, j), max(i, j))
                if move_key in tabu_list and tabu_list[move_key] > iteration:
                    # It is tabu
                    # Check aspiration: is this move better than best_cost?
                    # We need delta to check.
                    pass # We calculate delta first to check aspiration
                    
                # Calculate Delta
                if is_symmetric_a and is_symmetric_b:
                    diff_a = a[i] - a[j]
                    diff_b = B_p[v] - B_p[u]
                    full_dot = np.dot(diff_a, diff_b)
                    
                    term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    
                    delta = 2 * (full_dot - term_i - term_j)
                else:
                    diff_a_row = a[i] - a[j]
                    diff_b_row = B_p[v] - B_p[u]
                    
                    sum1_full = np.dot(diff_a_row, diff_b_row)
                    term1_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    term1_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    sum1 = sum1_full - term1_i - term1_j
                    
                    diff_a_col = a[:, i] - a[:, j]
                    diff_b_col = B_p[:, v] - B_p[:, u]
                    
                    sum2_full = np.dot(diff_a_col, diff_b_col)
                    term2_i = (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
                    term2_j = (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
                    sum2 = sum2_full - term2_i - term2_j
                    
                    direct = (a[i, j] * (b[v, u] - b[u, v]) + 
                              a[j, i] * (b[u, v] - b[v, u]) + 
                              a[i, i] * (b[v, v] - b[u, u]) + 
                              a[j, j] * (b[u, u] - b[v, v]))
                    
                    delta = sum1 + sum2 + direct

                # Check Aspiration
                if move_key in tabu_list and tabu_list[move_key] > iteration:
                    # It is tabu, so only allow if new cost < best_cost
                    new_cost = current_cost + delta
                    if new_cost < best_cost:
                        is_aspiration = True
                    else:
                        continue # Skip tabu move
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
                    is_aspiration = False # Reset aspiration flag for this best candidate
        
        # If no valid move found (all tabu and not aspiration), pick a random move
        if best_i == -1:
            # Pick random non-tabu move if possible, else any
            valid_moves = []
            for i in range(n):
                for j in range(i+1, n):
                    move_key = (i, j)
                    if move_key not in tabu_list or tabu_list[move_key] <= iteration:
                        valid_moves.append((i, j))
            if valid_moves:
                best_i, best_j = random.choice(valid_moves)
                # Calculate delta for this random move
                u = p[best_i]
                v = p[best_j]
                if is_symmetric_a and is_symmetric_b:
                    diff_a = a[best_i] - a[best_j]
                    diff_b = B_p[v] - B_p[u]
                    full_dot = np.dot(diff_a, diff_b)
                    term_i = (a[best_i, best_i] - a[best_j, best_i]) * (b[v, u] - b[u, u])
                    term_j = (a[best_i, best_j] - a[best_j, best_j]) * (b[v, v] - b[u, v])
                    best_delta = 2 * (full_dot - term_i - term_j)
                else:
                    diff_a_row = a[best_i] - a[best_j]
                    diff_b_row = B_p[v] - B_p[u]
                    sum1_full = np.dot(diff_a_row, diff_b_row)
                    term1_i = (a[best_i, best_i] - a[best_j, best_i]) * (b[v, u] - b[u, u])
                    term1_j = (a[best_i, best_j] - a[best_j, best_j]) * (b[v, v] - b[u, v])
                    sum1 = sum1_full - term1_i - term1_j
                    diff_a_col = a[:, best_i] - a[:, best_j]
                    diff_b_col = B_p[:, v] - B_p[:, u]
                    sum2_full = np.dot(diff_a_col, diff_b_col)
                    term2_i = (a[best_i, best_i] - a[best_i, best_j]) * (b[u, v] - b[u, u])
                    term2_j = (a[best_j, best_i] - a[best_j, best_j]) * (b[v, v] - b[v, u])
                    sum2 = sum2_full - term2_i - term2_j
                    direct = (a[best_i, best_j] * (b[v, u] - b[u, v]) + 
                              a[best_j, best_i] * (b[u, v] - b[v, u]) + 
                              a[best_i, best_i] * (b[v, v] - b[u, u]) + 
                              a[best_j, best_j] * (b[u, u] - b[v, v]))
                    best_delta = sum1 + sum2 + direct
            else:
                # All moves tabu? Just pick random
                best_i = random.randint(0, n-1)
                best_j = random.randint(0, n-1)
                while best_j == best_i:
                    best_j = random.randint(0, n-1)
                # Recalc delta
                u = p[best_i]
                v = p[best_j]
                # ... (same delta calc as above, omitted for brevity in thought, will be in code)
                # For safety, just accept the move and recalc cost fully if needed, or approximate
                # Let's just recalc cost fully for this rare case
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost = calc_cost(p)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                iteration += 1
                continue
        
        # Perform the best move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += int(best_delta)
        
        # Update Tabu List
        move_key = (min(best_i, best_j), max(best_i, best_j))
        tabu_list[move_key] = iteration + tabu_tenure
        
        # Update Best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        iteration += 1
        
        # Stagnation handling
        if no_improve_count > max_no_improve:
            # Perturb
            for _ in range(5):
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = calc_cost(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            no_improve_count = 0
            # Clear tabu list to allow exploration
            tabu_list.clear()
            
        if iteration > max_iterations:
            break

    return best_p.tolist()
# EVOLVE-BLOCK-END
