"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order in O(n_jobs * n_machines)."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # C[i][j] = completion time of job i on machine j
    # We can compute this efficiently
    prev_machine_time = np.zeros(n_machines)  # completion time of previous job on each machine
    current_job_time = 0.0
    
    for i in range(n_jobs):
        job_idx = order[i]
        for j in range(n_machines):
            if j == 0:
                current_job_time = current_job_time + processing_times[job_idx, j]
            else:
                current_job_time = max(current_job_time, prev_machine_time[j-1]) + processing_times[job_idx, j]
            prev_machine_time[j] = current_job_time
    
    return prev_machine_time[-1]


def _neh_rule(processing_times):
    """NEH heuristic for initial solution."""
    n_jobs, n_machines = processing_times.shape
    sums = processing_times.sum(axis=1)
    order = list(np.argsort(-sums))
    
    # Insert each job in best position
    result = [order[0]]
    for k in range(1, n_jobs):
        job = order[k]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(result) + 1):
            candidate = result[:pos] + [job] + result[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        result = result[:best_pos] + [job] + result[best_pos:]
    
    return result


def _local_search(processing_times, order, max_iters=5000, time_limit=10.0):
    """Local search with swap and insertion moves."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 100
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if no_improve_count > max_no_improve:
            break
        
        improved = False
        
        # Try swap moves
        for i in range(n_jobs):
            for j in range(i + 1, n_jobs):
                candidate = list(current_order)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan(processing_times, candidate)
                if ms < current_ms:
                    current_order = candidate
                    current_ms = ms
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try insertion moves
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i == j:
                    continue
                candidate = list(current_order)
                job = candidate.pop(i)
                candidate.insert(j, job)
                ms = _makespan(processing_times, candidate)
                if ms < current_ms:
                    current_order = candidate
                    current_ms = ms
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
    
    return current_order, current_ms


def _tabu_search(processing_times, order, tabu_tenure=10, max_iters=2000, time_limit=15.0):
    """Tabu search with aspiration criterion."""
    n_jobs = len(order)
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_ms = current_ms
    
    # Tabu list: (position_i, position_j, iteration)
    tabu_list = {}
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        best_move = None
        best_move_ms = current_ms
        best_move_order = None
        
        # Try all swap moves
        for i in range(n_jobs):
            for j in range(i + 1, n_jobs):
                candidate = list(current_order)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan(processing_times, candidate)
                
                # Check tabu
                key = (min(i, j), max(i, j))
                is_tabu = key in tabu_list and (it - tabu_list[key]) < tabu_tenure
                
                # Aspiration: accept if better than best found
                if is_tabu and ms >= best_ms:
                    continue
                
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move_order = candidate
                    best_move = (i, j)
        
        if best_move is None:
            break
        
        # Update tabu list
        i, j = best_move
        tabu_list[(min(i, j), max(i, j))] = it
        # Clean old tabu entries
        old_keys = [k for k, v in tabu_list.items() if it - v >= tabu_tenure]
        for k in old_keys:
            del tabu_list[k]
        
        current_order = best_move_order
        current_ms = best_move_ms
        
        if current_ms < best_ms:
            best_ms = current_ms
            best_order = list(current_order)
    
    return best_order, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan."""
    start_time = time.time()
    deadline = start_time + 140.0  # 20s safety margin before 160s kill
    
    n_jobs, n_machines = processing_times.shape
    
    # Generate initial solutions using different heuristics
    initial_solutions = []
    
    # NEH rule
    neh_sol = _neh_rule(processing_times)
    initial_solutions.append(neh_sol)
    
    # Longest processing time first
    lpt_sol = list(np.argsort(-processing_times.sum(axis=1)))
    initial_solutions.append(lpt_sol)
    
    # Shortest processing time first
    spt_sol = list(np.argsort(processing_times.sum(axis=1)))
    initial_solutions.append(spt_sol)
    
    # Random permutations
    rng = np.random.RandomState(42)
    for _ in range(5):
        rand_sol = list(rng.permutation(n_jobs))
        initial_solutions.append(rand_sol)
    
    # Local search on each initial solution
    best_order = None
    best_ms = float('inf')
    
    for sol in initial_solutions:
        if time.time() > deadline:
            break
        ls_order, ls_ms = _local_search(processing_times, sol, max_iters=2000, time_limit=5.0)
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_order = ls_order
    
    # Tabu search from the best local search result
    if best_order is not None and time.time() < deadline:
        ts_order, ts_ms = _tabu_search(processing_times, best_order, tabu_tenure=10, 
                                        max_iters=3000, time_limit=15.0)
        if ts_ms < best_ms:
            best_ms = ts_ms
            best_order = ts_order
    
    # Final local search polish
    if best_order is not None and time.time() < deadline:
        final_order, final_ms = _local_search(processing_times, best_order, max_iters=5000, 
                                               time_limit=5.0)
        if final_ms < best_ms:
            best_ms = final_ms
            best_order = final_order
    
    return best_order
# EVOLVE-BLOCK-END
