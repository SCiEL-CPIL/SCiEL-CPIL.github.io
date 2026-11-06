"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan_from_order(order, processing_times):
    n_machines = processing_times.shape[1]
    if n_machines == 0 or len(order) == 0:
        return 0.0
    completion = np.zeros(n_machines)
    for job in order:
        p = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            prev = max(prev, completion[m]) + p[m]
            completion[m] = prev
    return float(completion[-1])


def _neh_initialization(processing_times, rng=None):
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    if rng is None:
        rng = np.random.default_rng()

    totals = processing_times.sum(axis=1)
    # Add noise for diversity
    noise = rng.random(n_jobs) * 0.1 * (np.max(totals) - np.min(totals) + 1e-9)
    order_keys = - (totals + noise)
    initial_order = list(np.argsort(order_keys))

    seq = []
    for job in initial_order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [job] + seq[pos:]
            ms = _makespan_from_order(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    return seq


def order_jobs(processing_times):
    start_time = time.time()
    time_budget = 110.0
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    rng = np.random.default_rng()
    
    best_order = None
    best_ms = float('inf')
    
    # Phase 1: Diverse NEH
    init_time_limit = time_budget * 0.2
    n_initial_max = 50
    for i in range(n_initial_max):
        if time.time() - start_time > init_time_limit:
            break
        order = _neh_initialization(processing_times, rng)
        ms = _makespan_from_order(order, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_order = list(order)
            
    if best_order is None:
        return list(np.argsort(-processing_times.sum(axis=1)))
        
    # Phase 2: Local Search (Adjacent Swaps)
    current_order = best_order
    current_ms = best_ms
    
    # We will perform hill climbing on adjacent swaps.
    # We will run multiple passes until no improvement is found or time runs out.
    
    while True:
        if time.time() - start_time > time_budget - 5:
            break
            
        improved = False
        # Try all adjacent swaps in a single pass
        for i in range(n_jobs - 1):
            if time.time() - start_time > time_budget - 5:
                break
                
            # Swap i and i+1
            current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
            ms = _makespan_from_order(current_order, processing_times)
            
            if ms < current_ms - 1e-9:
                current_ms = ms
                improved = True
                if current_ms < best_ms - 1e-9:
                    best_ms = current_ms
                    best_order = list(current_order)
            else:
                # Swap back
                current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                
        if not improved:
            break
            
    return best_order
# EVOLVE-BLOCK-END
