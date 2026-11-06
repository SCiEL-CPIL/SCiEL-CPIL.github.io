"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan_vectorized(processing_times, perm):
    """
    Calculate makespan for a given permutation.
    """
    if len(perm) == 0:
        return 0.0
    
    p = processing_times[perm]
    n_jobs, n_machines = p.shape
    
    if n_machines == 1:
        return np.sum(p)
    
    # C[i, j] = max(C[i-1, j], C[i, j-1]) + P[i, j]
    # We compute column by column, using a Python loop for the sequential dependency along rows.
    C = np.empty_like(p, dtype=np.float64)
    C[:, 0] = np.cumsum(p[:, 0])
    
    for j in range(1, n_machines):
        c_prev_row = 0.0
        c_col_j_minus_1 = C[:, j-1]
        c_col_j = C[:, j]
        p_col_j = p[:, j]
        
        for i in range(n_jobs):
            if c_prev_row > c_col_j_minus_1[i]:
                val = c_prev_row + p_col_j[i]
            else:
                val = c_col_j_minus_1[i] + p_col_j[i]
            c_col_j[i] = val
            c_prev_row = val

    return C[-1, -1]

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Initial Construction: NEH Heuristic
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    current_perm = [sorted_jobs[0]]
    
    for i in range(1, n_jobs):
        job = sorted_jobs[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_perm) + 1):
            candidate_perm = current_perm[:pos] + [job] + current_perm[pos:]
            ms = _calc_makespan_vectorized(processing_times, candidate_perm)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
                
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    best_perm = current_perm
    best_ms = _calc_makespan_vectorized(processing_times, best_perm)
    
    # 2. Iterated Local Search (ILS) with 2-opt neighborhood
    # Strategy:
    # - Perform a full 2-opt local search (swap any two jobs) to reach a local optimum.
    # - Perturb the local optimum with a random block move (Or-opt style, size 1 or 2).
    # - Repeat until time runs out.
    # This avoids the temperature scheduling issues of SA and focuses on deep local search.
    
    current_perm = best_perm
    current_ms = best_ms
    
    def _local_search_2opt(perm, current_ms):
        """
        Perform 2-opt local search (swap any two jobs).
        Returns best_perm, best_ms found.
        """
        perm = list(perm)
        best_ms_ls = current_ms
        best_perm_ls = list(perm)
        
        improved = True
        while improved:
            improved = False
            # To speed up, we can iterate over pairs. For N=100, N^2/2 = 5000 pairs.
            # Each evaluation is O(N*M).
            # If N=100, M=10, eval is ~1000 ops. 5000 * 1000 = 5M ops per pass.
            # Python might handle ~10M simple ops/sec, so 1 pass ~ 0.5s.
            # We need to be careful with time.
            
            # Optimization: Use a "first improvement" or "best improvement" strategy?
            # Best improvement requires evaluating all pairs, then picking the best.
            # First improvement stops at the first improvement.
            # Best improvement is usually better for convergence quality but slower per pass.
            # Given the time budget, let's try best improvement but with a time check inside.
            
            best_delta = float('inf')
            best_i = -1
            best_j = -1
            
            for i in range(n_jobs - 1):
                if time.time() > deadline - 5.0:
                    return best_perm_ls, best_ms_ls
                for j in range(i + 1, n_jobs):
                    # Swap i and j
                    # Instead of creating a new list, we can compute makespan on a modified view?
                    # No, _calc_makespan_vectorized takes a list.
                    # Creating a new list is O(N).
                    # Let's try to optimize the swap evaluation.
                    
                    # Simple swap
                    perm[i], perm[j] = perm[j], perm[i]
                    ms = _calc_makespan_vectorized(processing_times, perm)
                    perm[i], perm[j] = perm[j], perm[i] # Swap back
                    
                    delta = ms - current_ms
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                        if delta < 0:
                            improved = True
            
            if best_i != -1 and best_delta < 0:
                perm[best_i], perm[best_j] = perm[best_j], perm[best_i]
                current_ms = best_ms_ls + best_delta
                best_ms_ls = current_ms
                best_perm_ls = list(perm)
            # If no improvement, the while loop will terminate.
            
        return best_perm_ls, best_ms_ls

    while time.time() < deadline - 5.0:
        # Local Search
        current_perm, current_ms = _local_search_2opt(current_perm, current_ms)
        
        if current_ms < best_ms:
            best_ms = current_ms
            best_perm = list(current_perm)
        
        if time.time() > deadline - 5.0:
            break
            
        # Perturbation: Move a block of size 1 or 2 to a random position
        block_size = random.choice([1, 2])
        
        if block_size == 1:
            i = random.randint(0, n_jobs - 1)
            job = current_perm[i]
            temp_perm = current_perm[:i] + current_perm[i+1:]
            if len(temp_perm) > 0:
                k_ins = random.randint(0, len(temp_perm) - 1)
                new_perm = temp_perm[:k_ins] + [job] + temp_perm[k_ins:]
            else:
                new_perm = [job]
        else:
            if n_jobs < 2:
                continue
            i = random.randint(0, n_jobs - 2)
            block = current_perm[i:i+2]
            temp_perm = current_perm[:i] + current_perm[i+2:]
            if len(temp_perm) > 0:
                k_ins = random.randint(0, len(temp_perm) - 1)
                new_perm = temp_perm[:k_ins] + block + temp_perm[k_ins:]
            else:
                new_perm = block
                
        current_perm = new_perm
        current_ms = _calc_makespan_vectorized(processing_times, current_perm)

    return best_perm
# EVOLVE-BLOCK-END
