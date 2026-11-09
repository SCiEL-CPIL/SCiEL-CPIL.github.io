"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order on the flow shop."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    # completion_times[m] tracks the completion time on machine m
    completion_times = np.zeros(n_machines)
    for job_idx in order:
        for m in range(n_machines):
            completion_times[m] = max(completion_times[m], completion_times[m-1] if m > 0 else 0.0) + processing_times[job_idx, m]
    return completion_times[-1]


def _neh_order(processing_times):
    """NEH heuristic: order jobs by decreasing total processing time."""
    totals = processing_times.sum(axis=1)
    return list(np.argsort(-totals))


def _construct_insertion_order(processing_times, order):
    """Given an initial order, greedily insert jobs one by one to minimize makespan."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return []
    
    # Start with the first job
    current_order = [order[0]]
    
    for job in order[1:]:
        # Try inserting job at each possible position in current_order
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(current_order) + 1):
            candidate = current_order[:pos] + [job] + current_order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        current_order = current_order[:best_pos] + [job] + current_order[best_pos:]
    
    return current_order


def _local_search(processing_times, order, time_limit=5.0):
    """Simple local search using adjacent swaps and 2-opt style moves."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n_jobs - 1):
            if (time.time() - start_time) >= time_limit:
                break
            # Try swapping adjacent jobs i and i+1
            candidate = current_order.copy()
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            ms = _makespan(processing_times, candidate)
            if ms < current_ms:
                current_order = candidate
                current_ms = ms
                improved = True
        # Also try inserting job i at position j (2-opt style)
        for i in range(n_jobs):
            if (time.time() - start_time) >= time_limit:
                break
            job = current_order[i]
            remaining = [j for j in current_order if j != job]
            for j in range(len(remaining) + 1):
                if (time.time() - start_time) >= time_limit:
                    break
                candidate = remaining[:j] + [job] + remaining[j:]
                ms = _makespan(processing_times, candidate)
                if ms < current_ms:
                    current_order = candidate
                    current_ms = ms
                    improved = True
                    break
    
    return current_order


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Initial heuristic: NEH
    neh_order = _neh_order(processing_times)
    
    # Improve with insertion construction
    inserted_order = _construct_insertion_order(processing_times, neh_order)
    best_order = inserted_order
    best_ms = _makespan(processing_times, best_order)
    
    # Local search
    ls_order = _local_search(processing_times, best_order, time_limit=5.0)
    ls_ms = _makespan(processing_times, ls_order)
    if ls_ms < best_ms:
        best_order = ls_order
        best_ms = ls_ms
    
    # If time remains, try a few more local search restarts from perturbed orders
    elapsed = time.time() - start_time
    if elapsed < deadline - 2.0:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() >= deadline - 2.0:
                break
            # Perturb: random swap
            perturbed = best_order.copy()
            i, j = rng.integers(0, n_jobs, size=2)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            
            ls2 = _local_search(processing_times, perturbed, time_limit=3.0)
            ms2 = _makespan(processing_times, ls2)
            if ms2 < best_ms:
                best_order = ls2
                best_ms = ms2
    
    return list(best_order)
# EVOLVE-BLOCK-END
