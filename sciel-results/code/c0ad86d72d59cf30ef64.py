"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(order, processing_times):
    """Compute makespan for a given job permutation."""
    n = len(order)
    m = processing_times.shape[1]
    # completion_time[j] tracks the completion time of job j on the last machine
    # We simulate the flow shop
    machine_finish = np.zeros(m)
    for job in order:
        for k in range(m):
            if k == 0:
                machine_finish[k] = machine_finish[k] + processing_times[job, k]
            else:
                machine_finish[k] = max(machine_finish[k], machine_finish[k - 1]) + processing_times[job, k]
    return machine_finish[-1]


def _makespan_fast(order, pt):
    """Vectorized-ish makespan using numpy for speed on small instances."""
    # For very small instances, the loop above is fine.
    # For larger, we can try to be smarter, but let's stick to correctness first.
    n = len(order)
    m = pt.shape[1]
    # completion[j, k] = completion time of job j on machine k
    # completion[j, 0] = completion[j-1, 0] + pt[order[j], 0]
    # completion[j, k] = max(completion[j-1, k], completion[j, k-1]) + pt[order[j], k]
    
    # This is O(N*M). For N=20, M=5, this is 100 ops. Very fast.
    # The previous _makespan is also O(N*M).
    # Let's just use a single efficient function.
    
    m_finish = np.zeros(m)
    for job in order:
        for k in range(m):
            if k == 0:
                m_finish[k] += pt[job, k]
            else:
                m_finish[k] = max(m_finish[k], m_finish[k-1]) + pt[job, k]
    return m_finish[-1]


def _neh_initialization(pt):
    """NEH heuristic for initial solution."""
    n = pt.shape[0]
    m = pt.shape[1]
    totals = pt.sum(axis=1)
    # Sort jobs by decreasing total processing time
    order = list(np.argsort(-totals))
    
    best_order = order[:]
    best_ms = _makespan_fast(best_order, pt)
    
    for i in range(1, n):
        job = order[i]
        # Try inserting job at every position 0..i
        for pos in range(i + 1):
            trial_order = order[:pos] + [job] + order[pos:i] + order[i+1:]
            ms = _makespan_fast(trial_order, pt)
            if ms < best_ms:
                best_ms = ms
                best_order = trial_order
        order = best_order[:]
        # Note: standard NEH inserts one by one. The above is a simplified version.
        # Actually, standard NEH:
        # 1. Sort jobs by decreasing total time.
        # 2. Start with first two jobs, try both orders, pick best.
        # 3. For each subsequent job, insert it into the best position of the current sequence.
        
        # Let's re-implement NEH properly.
    # Re-doing NEH properly
    order = list(np.argsort(-totals))
    best_order = order[:2]
    # Try both orders for the first two
    ms1 = _makespan_fast(order[:2], pt)
    ms2 = _makespan_fast(order[1:0:-1], pt) # [order[1], order[0]]
    if ms2 < ms1:
        best_order = [order[1], order[0]]
    
    for i in range(2, n):
        job = order[i]
        best_pos = 0
        best_ms = _makespan_fast(best_order + [job], pt)
        for pos in range(1, len(best_order) + 1):
            trial = best_order[:pos] + [job] + best_order[pos:]
            ms = _makespan_fast(trial, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
                # We don't update best_order yet, we find the best pos first
        best_order = best_order[:best_pos] + [job] + best_order[best_pos:]
        
    return best_order


def _local_search(order, pt, max_iterations=1000):
    """Adjacent swap local search."""
    n = len(order)
    current_ms = _makespan_fast(order, pt)
    best_order = order[:]
    best_ms = current_ms
    
    for _ in range(max_iterations):
        improved = False
        for i in range(n - 1):
            # Try swapping i and i+1
            trial_order = best_order[:]
            trial_order[i], trial_order[i+1] = trial_order[i+1], trial_order[i]
            ms = _makespan_fast(trial_order, pt)
            if ms < best_ms:
                best_ms = ms
                best_order = trial_order
                improved = True
        if not improved:
            break
            
    return best_order, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    Uses NEH initialization + Local Search + Simulated Annealing.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    pt = np.asarray(processing_times)
    n, m = pt.shape
    
    # 1. NEH Initialization
    best_order = _neh_initialization(pt)
    best_ms = _makespan_fast(best_order, pt)
    
    # 2. Local Search Polish
    best_order, best_ms = _local_search(best_order, pt, max_iterations=500)
    
    # 3. Simulated Annealing / Multi-start
    # If we have time, do more.
    # Given N is likely small (< 50), we can do many restarts.
    
    # Simple SA on the best solution found so far
    current_order = best_order[:]
    current_ms = best_ms
    temp = 100.0
    cooling = 0.99
    
    # Determine step size based on N
    # For small N, we can afford more iterations
    
    while time.time() < deadline:
        # Pick a random adjacent swap
        i = random.randint(0, n - 2)
        trial_order = current_order[:]
        trial_order[i], trial_order[i+1] = trial_order[i+1], trial_order[i]
        trial_ms = _makespan_fast(trial_order, pt)
        
        delta = trial_ms - current_ms
        if delta < 0 or random.random() < np.exp(-delta / temp):
            current_order = trial_order
            current_ms = trial_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = current_order[:]
        
        temp *= cooling
        if temp < 0.01:
            break
            
    # If we still have time, do multi-start with random initializations
    while time.time() < deadline - 5.0:
        # Random initialization
        random_order = list(range(n))
        random.shuffle(random_order)
        
        # Local search from random start
        ls_order, ls_ms = _local_search(random_order, pt, max_iterations=100)
        
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_order = ls_order[:]
            
    return best_order

# EVOLVE-BLOCK-END
