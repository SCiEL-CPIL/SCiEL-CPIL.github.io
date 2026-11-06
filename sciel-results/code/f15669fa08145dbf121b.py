"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan_fast(order, processing_times):
    """
    Fast makespan calculation using pre-allocated arrays.
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    if len(order) == 0:
        return 0.0
    
    prev = np.zeros(n_machines)
    
    for j in order:
        if n_machines == 1:
            prev[0] += processing_times[j, 0]
        else:
            curr = np.empty(n_machines)
            curr[0] = prev[0] + processing_times[j, 0]
            for m in range(1, n_machines):
                curr[m] = max(curr[m-1], prev[m]) + processing_times[j, m]
            prev = curr
    
    return prev[-1]


def _makespan(order, processing_times):
    """
    Calculate the makespan for a given job order in a flow shop.
    """
    return _makespan_fast(order, processing_times)


def _makespan_array(order, processing_times):
    """
    Vectorized makespan calculation returning completion times per machine.
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return np.array([0.0])
    if len(order) == 0:
        return np.zeros(n_machines)
    
    prev = np.zeros(n_machines)
    
    for j in order:
        if n_machines == 1:
            prev[0] += processing_times[j, 0]
        else:
            curr = np.empty(n_machines)
            curr[0] = prev[0] + processing_times[j, 0]
            for m in range(1, n_machines):
                curr[m] = max(curr[m-1], prev[m]) + processing_times[j, m]
            prev = curr
    
    return prev


def _neh_heuristic(processing_times):
    """
    Generate an initial permutation using the NEH (Nawaz-Enscore-Ham) heuristic.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Calculate sum of processing times for each job
    job_sums = processing_times.sum(axis=1)
    
    # Sort jobs in descending order of their sums
    sorted_jobs = np.argsort(-job_sums)
    
    current_sequence = []
    
    for job in sorted_jobs:
        # Find the best position to insert the current job
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(current_sequence) + 1):
            new_seq = current_sequence[:pos] + [job] + current_sequence[pos:]
            ms = _makespan_fast(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        current_sequence.insert(best_pos, job)
        
    return current_sequence


def _compute_prefix_completions(order, processing_times):
    """
    Precompute prefix completion times for efficient incremental evaluation.
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n_machines == 0:
        return [np.array([0.0])] * (n + 1)
    
    prefix = [np.zeros(n_machines)]
    current = np.zeros(n_machines)
    
    for i in range(n):
        j = order[i]
        if n_machines == 1:
            current[0] += processing_times[j, 0]
        else:
            curr = np.empty(n_machines)
            curr[0] = current[0] + processing_times[j, 0]
            for m in range(1, n_machines):
                curr[m] = max(curr[m-1], current[m]) + processing_times[j, m]
            current = curr
        
        prefix.append(current.copy())
    
    return prefix


def _eval_insertion_fast(order, processing_times, prefix, i, j):
    """
    Fast evaluation of insertion move from position i to position j (i < j).
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n_machines == 0:
        return 0.0
    
    current = prefix[i].copy()
    
    # Process jobs from i+1 to j
    for k in range(i + 1, j + 1):
        job = order[k]
        if n_machines == 1:
            current[0] += processing_times[job, 0]
        else:
            curr = np.empty(n_machines)
            curr[0] = current[0] + processing_times[job, 0]
            for m in range(1, n_machines):
                curr[m] = max(curr[m-1], current[m]) + processing_times[job, m]
            current = curr
    
    # Process the moved job
    job = order[i]
    if n_machines == 1:
        current[0] += processing_times[job, 0]
    else:
        curr = np.empty(n_machines)
        curr[0] = current[0] + processing_times[job, 0]
        for m in range(1, n_machines):
            curr[m] = max(curr[m-1], current[m]) + processing_times[job, m]
        current = curr
    
    # Process remaining jobs from j+1 to end
    for k in range(j + 1, n):
        job = order[k]
        if n_machines == 1:
            current[0] += processing_times[job, 0]
        else:
            curr = np.empty(n_machines)
            curr[0] = current[0] + processing_times[job, 0]
            for m in range(1, n_machines):
                curr[m] = max(curr[m-1], current[m]) + processing_times[job, m]
            current = curr
    
    return current[-1]


def _eval_insertion_reverse_fast(order, processing_times, prefix, i, j):
    """
    Fast evaluation of insertion move from position i to position j (i > j).
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n_machines == 0:
        return 0.0
    
    current = prefix[j].copy()
    
    # Process the moved job
    job = order[i]
    if n_machines == 1:
        current[0] += processing_times[job, 0]
    else:
        curr = np.empty(n_machines)
        curr[0] = current[0] + processing_times[job, 0]
        for m in range(1, n_machines):
            curr[m] = max(curr[m-1], current[m]) + processing_times[job, m]
        current = curr
    
    # Process jobs from j to i-1
    for k in range(j, i):
        job = order[k]
        if n_machines == 1:
            current[0] += processing_times[job, 0]
        else:
            curr = np.empty(n_machines)
            curr[0] = current[0] + processing_times[job, 0]
            for m in range(1, n_machines):
                curr[m] = max(curr[m-1], current[m]) + processing_times[job, m]
            current = curr
    
    # Process remaining jobs from i+1 to end
    for k in range(i + 1, n):
        job = order[k]
        if n_machines == 1:
            current[0] += processing_times[job, 0]
        else:
            curr = np.empty(n_machines)
            curr[0] = current[0] + processing_times[job, 0]
            for m in range(1, n_machines):
                curr[m] = max(curr[m-1], current[m]) + processing_times[job, m]
            current = curr
    
    return current[-1]


def _eval_2opt(order, processing_times, prefix, i, j):
    """
    Evaluate makespan for reversing segment [i, j] (i < j).
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n_machines == 0:
        return 0.0
    
    current = prefix[i].copy()
    
    # Process reversed segment
    for k in range(j, i - 1, -1):
        job = order[k]
        if n_machines == 1:
            current[0] += processing_times[job, 0]
        else:
            curr = np.empty(n_machines)
            curr[0] = current[0] + processing_times[job, 0]
            for m in range(1, n_machines):
                curr[m] = max(curr[m-1], current[m]) + processing_times[job, m]
            current = curr
    
    # Process remaining jobs
    for k in range(j + 1, n):
        job = order[k]
        if n_machines == 1:
            current[0] += processing_times[job, 0]
        else:
            curr = np.empty(n_machines)
            curr[0] = current[0] + processing_times[job, 0]
            for m in range(1, n_machines):
                curr[m] = max(curr[m-1], current[m]) + processing_times[job, m]
            current = curr
    
    return current[-1]


def _local_search_insertion(order, processing_times, deadline, best_order, best_makespan):
    """
    Local search using insertion moves with prefix precomputation.
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n <= 1 or n_machines == 0:
        return order[:], best_makespan
    
    current_order = order[:]
    current_ms = best_makespan
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        prefix = _compute_prefix_completions(current_order, processing_times)
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            best_pos = i
            best_eval_ms = current_ms
            
            for j in range(n):
                if j == i:
                    continue
                
                if i < j:
                    ms = _eval_insertion_fast(current_order, processing_times, prefix, i, j)
                else:
                    ms = _eval_insertion_reverse_fast(current_order, processing_times, prefix, i, j)
                
                if ms < best_eval_ms:
                    best_eval_ms = ms
                    best_pos = j
            
            if best_pos != i:
                job = current_order.pop(i)
                current_order.insert(best_pos, job)
                current_ms = best_eval_ms
                
                if current_ms < best_makespan:
                    best_makespan = current_ms
                    best_order = current_order[:]
                
                improved = True
                break
    
    return best_order, best_makespan


def _local_search_2opt(order, processing_times, deadline, best_order, best_makespan):
    """
    Local search using 2-opt (segment reversal) moves.
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n <= 2 or n_machines == 0:
        return order[:], best_makespan
    
    current_order = order[:]
    current_ms = best_makespan
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        prefix = _compute_prefix_completions(current_order, processing_times)
        
        for i in range(n - 1):
            if time.time() > deadline:
                break
            
            best_j = -1
            best_eval_ms = current_ms
            
            for j in range(i + 1, n):
                ms = _eval_2opt(current_order, processing_times, prefix, i, j)
                if ms < best_eval_ms:
                    best_eval_ms = ms
                    best_j = j
            
            if best_j != -1:
                # Reverse segment [i, best_j]
                current_order[i:best_j+1] = current_order[i:best_j+1][::-1]
                current_ms = best_eval_ms
                
                if current_ms < best_makespan:
                    best_makespan = current_ms
                    best_order = current_order[:]
                
                improved = True
                break
    
    return best_order, best_makespan


def _simulated_annealing(order, processing_times, deadline, best_order, best_makespan):
    """
    Simulated annealing with insertion and 2-opt moves.
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n <= 1 or n_machines == 0:
        return best_order, best_makespan
    
    current_order = order[:]
    current_ms = _makespan_fast(current_order, processing_times)
    
    T = max(1.0, current_ms * 0.1)
    T_min = max(0.01, current_ms * 0.001)
    cooling_rate = 0.97
    
    while T > T_min and time.time() < deadline:
        # Randomly choose between insertion and 2-opt
        if random.random() < 0.7:
            # Insertion move
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            
            if i == j:
                continue
            
            job = current_order[i]
            remaining = current_order[:i] + current_order[i+1:]
            
            if j > i:
                j -= 1
            j = max(0, min(j, len(remaining)))
            
            new_order = remaining[:j] + [job] + remaining[j:]
        else:
            # 2-opt move
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            
            new_order = current_order[:]
            new_order[i:j+1] = new_order[i:j+1][::-1]
        
        new_ms = _makespan_fast(new_order, processing_times)
        delta = new_ms - current_ms
        
        if delta < 0:
            current_order = new_order
            current_ms = new_ms
            
            if current_ms < best_makespan:
                best_makespan = current_ms
                best_order = current_order[:]
        else:
            if random.random() < np.exp(-delta / T):
                current_order = new_order
                current_ms = new_ms
        
        T *= cooling_rate
    
    return best_order, best_makespan


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # 1. Generate initial solution using NEH heuristic
    current_order = _neh_heuristic(processing_times)
    current_makespan = _makespan_fast(current_order, processing_times)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # 2. Local search with insertion moves
    best_order, best_makespan = _local_search_insertion(
        best_order, processing_times, deadline, best_order, best_makespan
    )
    
    # 3. Local search with 2-opt moves
    if time.time() < deadline - 10.0:
        best_order, best_makespan = _local_search_2opt(
            best_order, processing_times, deadline, best_order, best_makespan
        )
    
    # 4. Simulated annealing to escape local minima
    if time.time() < deadline - 10.0:
        best_order, best_makespan = _simulated_annealing(
            best_order, processing_times, deadline, best_order, best_makespan
        )
    
    # 5. Final local search polish (insertion)
    if time.time() < deadline - 3.0:
        best_order, best_makespan = _local_search_insertion(
            best_order, processing_times, deadline, best_order, best_makespan
        )
    
    # 6. Final 2-opt polish
    if time.time() < deadline - 1.0:
        best_order, best_makespan = _local_search_2opt(
            best_order, processing_times, deadline, best_order, best_makespan
        )
    
    return best_order

# EVOLVE-BLOCK-END
