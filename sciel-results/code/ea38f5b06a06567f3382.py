"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    # c[j] is completion time on machine j for the last processed job
    c = [0.0] * n_machines
    
    for job_idx in order:
        prev_c = 0.0
        row = processing_times[job_idx]
        for k in range(n_machines):
            val = c[k]
            if val < prev_c:
                val = prev_c
            val += row[k]
            c[k] = val
            prev_c = val
    return c[-1]


def _neh_rule(processing_times):
    """NEH heuristic for flow-shop problem."""
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    job_order = np.argsort(-totals)
    
    # Build sequence iteratively
    seq = []
    remaining = list(job_order)
    
    for _ in range(n_jobs):
        best_pos = 0
        best_ms = float('inf')
        
        # Try inserting the next job at each possible position
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [remaining[0]] + seq[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        seq = seq[:best_pos] + [remaining[0]] + seq[best_pos:]
        remaining.pop(0)
    
    return seq


def _insertion_local_search(processing_times, order, time_limit, start_time=None):
    """Insertion-based local search (remove and re-insert each job)."""
    if start_time is None:
        start_time = time.time()
    
    n_jobs = len(order)
    best_order = list(order)
    best_ms = _makespan(processing_times, best_order)
    
    improved = True
    while improved:
        improved = False
        
        if time.time() - start_time > time_limit:
            break
        
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                return best_order, best_ms
            
            # Remove job at position i
            job = best_order[i]
            remaining = best_order[:i] + best_order[i+1:]
            
            # Try inserting at each position
            best_pos = i
            best_candidate_ms = best_ms
            
            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan(processing_times, candidate)
                if ms < best_candidate_ms - 1e-9:
                    best_candidate_ms = ms
                    best_pos = pos
            
            if best_pos != i and best_candidate_ms < best_ms - 1e-9:
                best_order = remaining[:best_pos] + [job] + remaining[best_pos:]
                best_ms = best_candidate_ms
                improved = True
    
    return best_order, best_ms


def _simulated_annealing(processing_times, initial_order, time_limit, start_time=None):
    """Simulated annealing with insertion moves."""
    if start_time is None:
        start_time = time.time()
    
    n_jobs = len(initial_order)
    current_order = list(initial_order)
    current_ms = _makespan(processing_times, current_order)
    
    best_order = list(current_order)
    best_ms = current_ms
    
    # Temperature schedule
    t0 = max(1.0, current_ms * 0.1)
    t_end = 1e-6
    n_iterations = 10000
    
    # Precompute for speed
    rng = random.Random(42)
    
    for it in range(n_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Temperature
        alpha = it / max(1, n_iterations - 1)
        T = t0 * (t_end / t0) ** alpha
        
        # Generate neighbor: remove a job and insert at random position
        i = rng.randint(0, n_jobs - 1)
        job = current_order[i]
        remaining = current_order[:i] + current_order[i+1:]
        j = rng.randint(0, len(remaining))
        if j == i:
            continue
        neighbor = remaining[:j] + [job] + remaining[j:]
        
        neighbor_ms = _makespan(processing_times, neighbor)
        delta = neighbor_ms - current_ms
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_order = neighbor
            current_ms = neighbor_ms
            
            if current_ms < best_ms - 1e-9:
                best_order = list(current_order)
                best_ms = current_ms
    
    return best_order, best_ms


def _tabu_search(processing_times, initial_order, time_limit, start_time=None):
    """Tabu search with insertion moves."""
    if start_time is None:
        start_time = time.time()
    
    n_jobs = len(initial_order)
    current_order = list(initial_order)
    current_ms = _makespan(processing_times, current_order)
    
    best_order = list(current_order)
    best_ms = current_ms
    
    tabu_tenure = max(5, n_jobs // 3)
    tabu_list = {}  # (i, j) -> iteration when tabu expires
    iteration = 0
    max_iterations = 5000
    
    rng = random.Random(123)
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        iteration += 1
        
        # Generate all insertion neighbors
        best_neighbor = None
        best_neighbor_ms = float('inf')
        best_move = None
        
        for i in range(n_jobs):
            job = current_order[i]
            remaining = current_order[:i] + current_order[i+1:]
            
            for j in range(len(remaining) + 1):
                if j == i:
                    continue
                
                # Check tabu
                if (i, j) in tabu_list and tabu_list[(i, j)] > iteration:
                    continue
                
                candidate = remaining[:j] + [job] + remaining[j:]
                ms = _makespan(processing_times, candidate)
                
                # Aspiration: allow tabu move if it improves best
                if ms < best_ms - 1e-9:
                    if ms < best_neighbor_ms:
                        best_neighbor_ms = ms
                        best_neighbor = candidate
                        best_move = (i, j)
                elif ms < best_neighbor_ms:
                    best_neighbor_ms = ms
                    best_neighbor = candidate
                    best_move = (i, j)
        
        if best_neighbor is None:
            break
        
        current_order = best_neighbor
        current_ms = best_neighbor_ms
        
        # Add to tabu list
        if best_move:
            tabu_list[best_move] = iteration + tabu_tenure
        
        if current_ms < best_ms - 1e-9:
            best_order = list(current_order)
            best_ms = current_ms
    
    return best_order, best_ms


def order_jobs(processing_times):
    """Main entry point: construct a job permutation minimizing makespan."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    
    # Generate initial solution using NEH
    best_order = _neh_rule(processing_times)
    best_ms = _makespan(processing_times, best_order)
    
    # Phase 1: Insertion local search
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        best_order, best_ms = _insertion_local_search(
            processing_times, best_order, remaining_time - 2.0, start_time)
    
    # Phase 2: Simulated Annealing for diversification
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        sa_order, sa_ms = _simulated_annealing(
            processing_times, best_order, remaining_time - 5.0, start_time)
        if sa_ms < best_ms - 1e-9:
            best_order = sa_order
            best_ms = sa_ms
    
    # Phase 3: Another round of insertion local search
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        best_order, best_ms = _insertion_local_search(
            processing_times, best_order, remaining_time - 2.0, start_time)
    
    # Phase 4: Tabu search for further improvement
    remaining_time = deadline - time.time()
    if remaining_time > 3.0:
        ts_order, ts_ms = _tabu_search(
            processing_times, best_order, remaining_time - 3.0, start_time)
        if ts_ms < best_ms - 1e-9:
            best_order = ts_order
            best_ms = ts_ms
    
    # Phase 5: Final local search polish
    remaining_time = deadline - time.time()
    if remaining_time > 1.0:
        best_order, best_ms = _insertion_local_search(
            processing_times, best_order, remaining_time - 1.0, start_time)
    
    # Final check: ensure we return within time limit
    if time.time() > deadline - 0.5:
        return list(best_order)
    
    return list(best_order)
# EVOLVE-BLOCK-END
