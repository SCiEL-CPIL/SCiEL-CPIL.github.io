"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a given permutation efficiently."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0 or m == 0:
        return 0.0
    prev = np.zeros(m)
    for j in perm:
        cur = np.empty(m)
        cur[0] = pt[j, 0] + prev[0]
        for k in range(1, m):
            cur[k] = max(cur[k-1], prev[k]) + pt[j, k]
        prev = cur
    return prev[m-1]


def _neh_order(pt):
    """NEH heuristic: sort by decreasing total processing time, then insert each job at best position."""
    n, m = pt.shape
    total_pt = pt.sum(axis=1)
    order = list(np.argsort(-total_pt))
    
    perm = [order[0]]
    for i in range(1, n):
        job = order[i]
        best_pos = 0
        best_ms = _makespan(perm + [job], pt)
        for pos in range(1, len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial solution via NEH
    global_best_perm = _neh_order(pt)
    global_best_ms = _makespan(global_best_perm, pt)
    
    # Tabu search parameters
    tabu_tenure = max(10, min(40, n // 2))
    tabu_list = {}  # (i, block_len, j) -> iteration
    iteration = 0
    no_improve = 0
    max_no_improve = 80
    
    current_perm = list(global_best_perm)
    current_ms = global_best_ms
    
    while time.time() < deadline:
        iteration += 1
        
        # Clean up expired tabu entries
        if iteration % 10 == 0:
            for key in list(tabu_list.keys()):
                if iteration - tabu_list[key] > tabu_tenure:
                    del tabu_list[key]
        
        # Find best non-tabu move
        best_move = None
        best_move_ms = current_ms
        best_new_perm = None
        
        for i in range(n):
            for block_len in [1, 2, 3]:
                if i + block_len > n:
                    continue
                block = current_perm[i:i+block_len]
                remaining = current_perm[:i] + current_perm[i+block_len:]
                
                for j in range(len(remaining) + 1):
                    if j == i:
                        continue
                    
                    move_key = (i, block_len, j)
                    new_perm = remaining[:j] + block + remaining[j:]
                    ms = _makespan(new_perm, pt)
                    
                    if ms < best_move_ms:
                        is_tabu = move_key in tabu_list
                        if not is_tabu or ms < global_best_ms:  # aspiration
                            best_move_ms = ms
                            best_move = move_key
                            best_new_perm = new_perm
        
        if best_move is None:
            # All moves tabu or no improvement: random perturbation
            idx = np.random.randint(n)
            new_perm = list(current_perm)
            job = new_perm.pop(idx)
            new_idx = np.random.randint(n)
            new_perm.insert(new_idx, job)
            best_new_perm = new_perm
            best_move_ms = _makespan(best_new_perm, pt)
            best_move = (idx, 1, new_idx)
        
        # Apply move
        current_perm = best_new_perm
        current_ms = best_move_ms
        
        # Update global best
        if current_ms < global_best_ms:
            global_best_ms = current_ms
            global_best_perm = list(current_perm)
            no_improve = 0
        else:
            no_improve += 1
        
        # Add to tabu
        tabu_list[best_move] = iteration
        
        # Restart if stuck too long
        if no_improve >= max_no_improve:
            no_improve = 0
            tabu_list = {}
            # Perturb current solution
            for _ in range(3):
                idx = np.random.randint(n)
                job = current_perm.pop(idx)
                new_idx = np.random.randint(n)
                current_perm.insert(new_idx, job)
            current_ms = _makespan(current_perm, pt)
    
    return list(global_best_perm)
# EVOLVE-BLOCK-END
