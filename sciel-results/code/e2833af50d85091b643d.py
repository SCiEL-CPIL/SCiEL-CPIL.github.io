"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_and_flowtime(processing_times, perm):
    """
    Compute makespan and total flow time for a given permutation.
    Returns (makespan, total_flow_time).
    """
    n = len(perm)
    if n == 0:
        return 0.0, 0.0
    m = processing_times.shape[1]
    
    # C[j][k] = completion time of job perm[j] on machine k
    # C[j][k] = max(C[j][k-1], C[j-1][k]) + p[perm[j]][k]
    # Base cases: C[-1][k] = 0, C[j][-1] = 0
    
    # We need the full matrix to compute flow time (sum of completion times on last machine)
    # Actually, flow time is sum of completion times of each job on the last machine.
    # C[j][m-1] is the completion time of job perm[j] on the last machine.
    
    # To be efficient, we can just track the last column if we only needed makespan,
    # but for flow time we need all C[j][m-1].
    # Let's compute the full matrix or at least the last column.
    # Since m is usually small, full matrix is fine.
    
    completion = np.zeros((n, m))
    for j in range(n):
        job = perm[j]
        for k in range(m):
            if j > 0 and k > 0:
                completion[j, k] = max(completion[j, k-1], completion[j-1, k])
            elif j > 0:
                completion[j, k] = completion[j-1, k]
            elif k > 0:
                completion[j, k] = completion[j, k-1]
            else:
                completion[j, k] = 0.0
            completion[j, k] += processing_times[job, k]
    
    makespan = completion[-1, -1]
    total_flow_time = np.sum(completion[:, -1])
    
    return makespan, total_flow_time

def _makespan(processing_times, perm):
    """Compute only makespan for speed."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = processing_times.shape[1]
    
    # Optimization: only track the last column if we don't need flow time?
    # No, C[j][k] depends on C[j-1][k] and C[j][k-1].
    # We need the previous row's last element? No, C[j][k] needs C[j][k-1] (current row, prev col)
    # and C[j-1][k] (prev row, same col).
    # So we need the entire previous row and the current row's previous element.
    # We can use a 1D array for the previous row and a scalar for current row's prev element.
    
    prev_row = np.zeros(m)
    for j in range(n):
        job = perm[j]
        curr_row_prev = 0.0
        for k in range(m):
            if j > 0 and k > 0:
                curr_val = max(curr_row_prev, prev_row[k])
            elif j > 0:
                curr_val = prev_row[k]
            elif k > 0:
                curr_val = curr_row_prev
            else:
                curr_val = 0.0
            curr_val += processing_times[job, k]
            curr_row_prev = curr_val
            prev_row[k] = curr_val
            
    return prev_row[-1]

def _neh(processing_times, random_state=None):
    """NEH heuristic for flow shop scheduling."""
    n, m = processing_times.shape
    if n == 0:
        return []
    
    totals = processing_times.sum(axis=1)
    
    if random_state is not None:
        # Add small random noise to break ties randomly
        noise = random_state.uniform(0, 1e-6, size=n)
        sort_keys = -totals + noise
    else:
        sort_keys = -totals
    
    sorted_jobs = np.argsort(sort_keys)
    
    perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(processing_times, new_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm

def _local_search(processing_times, perm, deadline):
    """
    Perform local search using swap and or-opt moves.
    Uses makespan as the objective for local search to keep it fast.
    """
    n = len(perm)
    if n < 2:
        return perm
    
    current_perm = list(perm)
    current_ms = _makespan(processing_times, current_perm)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Swap moves
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                new_perm = current_perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_ms = _makespan(processing_times, new_perm)
                if new_ms < current_ms:
                    current_perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
        
        if time.time() > deadline:
            break
        
        # Or-opt moves (block size 1-3)
        for i in range(n):
            if time.time() > deadline:
                break
            for block_len in [1, 2, 3]:
                if i + block_len > n:
                    continue
                block = current_perm[i:i+block_len]
                remaining = current_perm[:i] + current_perm[i+block_len:]
                
                for j in range(len(remaining) + 1):
                    if time.time() > deadline:
                        break
                    # Skip original position
                    if j == i or (j == i + block_len):
                        continue
                    new_perm = remaining[:j] + block + remaining[j:]
                    new_ms = _makespan(processing_times, new_perm)
                    if new_ms < current_ms:
                        current_perm = new_perm
                        current_ms = new_ms
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return current_perm

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses a hybrid approach: NEH initialization, multi-objective-inspired diversity
    via randomized restarts, and local search.
    """
    start_time = time.time()
    # Safety margin: 10 seconds before the 150s limit
    deadline = start_time + 140
    
    n, m = processing_times.shape
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # 1. Initial NEH solution
    best_perm = _neh(processing_times)
    best_ms = _makespan(processing_times, best_perm)
    best_flow = None # We don't strictly need flow time for final selection if we focus on makespan,
                     # but the prompt suggests a MOEA approach.
                     # However, the task is to minimize makespan.
                     # The "MOEA" inspiration might be overkill or a distraction if the objective is single.
                     # Let's stick to a robust single-objective search with diverse restarts,
                     # which is structurally similar to exploring a Pareto front by
                     # keeping diverse high-quality solutions.
    
    # 2. Generate diverse initial solutions using randomized NEH
    # We keep a "population" of the best few solutions found so far to maintain diversity
    # and avoid getting stuck in the same local optimum.
    population = [(best_ms, best_perm)]
    
    rng = np.random.default_rng(42)
    
    # Run several randomized NEH restarts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 10: # Leave time for local search
            break
            
        alt_perm = _neh(processing_times, random_state=rng)
        alt_ms = _makespan(processing_times, alt_perm)
        
        # Add to population if it's distinct enough or better
        # Simple diversity check: if it's not too similar to existing population members
        is_diverse = True
        for _, pop_perm in population:
            # Simple similarity metric: number of positions where they differ
            if sum(1 for a, b in zip(alt_perm, pop_perm) if a == b) > 0.8 * n:
                is_diverse = False
                break
        
        if alt_ms < best_ms:
            best_ms = alt_ms
            best_perm = alt_perm
            population.insert(0, (alt_ms, alt_perm))
        elif is_diverse and alt_ms < best_ms * 1.1: # Keep slightly worse but diverse solutions
            population.append((alt_ms, alt_perm))
            population.sort(key=lambda x: x[0])
            if len(population) > 5:
                population.pop()
        
        # Local search on this new solution
        if time.time() < deadline - 5:
            ls_perm = _local_search(processing_times, alt_perm, deadline - 5)
            ls_ms = _makespan(processing_times, ls_perm)
            
            if ls_ms < best_ms:
                best_ms = ls_ms
                best_perm = ls_perm
                population.insert(0, (ls_ms, ls_perm))
            
            # Update population with local search result if diverse
            is_diverse = True
            for _, pop_perm in population:
                if sum(1 for a, b in zip(ls_perm, pop_perm) if a == b) > 0.8 * n:
                    is_diverse = False
                    break
            
            if is_diverse and ls_ms < best_ms * 1.1:
                population.append((ls_ms, ls_perm))
                population.sort(key=lambda x: x[0])
                if len(population) > 5:
                    population.pop()
    
    # 3. Final local search on the best solution found
    if time.time() < deadline:
        final_perm = _local_search(processing_times, best_perm, deadline)
        final_ms = _makespan(processing_times, final_perm)
        
        if final_ms < best_ms:
            best_perm = final_perm
            best_ms = final_ms
    
    return list(best_perm)

# EVOLVE-BLOCK-END
