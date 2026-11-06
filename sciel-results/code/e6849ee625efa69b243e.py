"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _calc_makespan(processing_times, perm):
    """
    Calculate makespan for a given permutation.
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    
    n_machines = processing_times.shape[1]
    if n_machines == 1:
        return float(processing_times[perm, 0].sum())
    
    pt = processing_times[perm]
    
    # prev_C[j] is the completion time of the previous job on machine j
    # curr_C[j] is the completion time of the current job on machine j
    
    # Initialize with the first job
    curr_C = np.empty(n_machines, dtype=float)
    curr_C[0] = pt[0, 0]
    for j in range(1, n_machines):
        curr_C[j] = curr_C[j-1] + pt[0, j]
        
    prev_C = curr_C.copy()
    
    for i in range(1, n_jobs):
        # Compute completion times for job i
        curr_C[0] = prev_C[0] + pt[i, 0]
        for j in range(1, n_machines):
            curr_C[j] = (prev_C[j] if prev_C[j] > curr_C[j-1] else curr_C[j-1]) + pt[i, j]
        
        # Swap buffers for next iteration
        prev_C, curr_C = curr_C, prev_C

    return float(prev_C[-1])


def _calc_makespan_vectorized(processing_times, perm):
    """
    Vectorized makespan calculation using NumPy.
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    
    n_machines = processing_times.shape[1]
    if n_machines == 1:
        return float(processing_times[perm, 0].sum())
    
    pt = processing_times[perm]
    
    # C[i, j] = completion time of job i on machine j
    C = np.zeros((n_jobs, n_machines), dtype=float)
    C[0, 0] = pt[0, 0]
    for j in range(1, n_machines):
        C[0, j] = C[0, j-1] + pt[0, j]
    
    for i in range(1, n_jobs):
        C[i, 0] = C[i-1, 0] + pt[i, 0]
        for j in range(1, n_machines):
            C[i, j] = np.maximum(C[i-1, j], C[i, j-1]) + pt[i, j]
    
    return float(C[-1, -1])


def _neh_heuristic(processing_times):
    """
    NEH heuristic: Sort jobs by total processing time descending,
    then insert each job into the position that minimizes makespan.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence


def _random_permutation(n_jobs):
    """Generate a random permutation of job indices."""
    perm = list(range(n_jobs))
    random.shuffle(perm)
    return perm


def _insertion_move(sequence, i, j):
    """
    Remove element at position i and insert it at position j.
    If j > i, adjust for the shift caused by removal.
    """
    new_seq = list(sequence)
    job = new_seq.pop(i)
    if j > i:
        insert_idx = j - 1
    else:
        insert_idx = j
    new_seq.insert(insert_idx, job)
    return new_seq


def _block_move(sequence, i, j, block_size):
    """
    Move a block of block_size jobs starting at position i to position j.
    """
    if i + block_size > len(sequence):
        return list(sequence)
    
    new_seq = list(sequence)
    block = new_seq[i:i+block_size]
    rest = new_seq[:i] + new_seq[i+block_size:]
    
    # Determine insertion position
    if j > i:
        insert_idx = j - block_size
    else:
        insert_idx = j
    
    if insert_idx < 0:
        insert_idx = 0
    if insert_idx > len(rest):
        insert_idx = len(rest)
    
    new_seq = rest[:insert_idx] + block + rest[insert_idx:]
    return new_seq


def _local_search(processing_times, sequence, deadline, max_iterations=1000):
    """
    First-improvement local search with insertion and block moves.
    Uses a time budget to limit the number of iterations.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    iterations = 0
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        improved = False
        
        # Try insertion moves (single job)
        for i in range(n_jobs):
            if time.time() >= deadline:
                return current_seq, current_makespan
            job = current_seq[i]
            for j in range(n_jobs):
                if j == i:
                    continue
                
                new_seq = _insertion_move(current_seq, i, j)
                new_makespan = _calc_makespan(processing_times, new_seq)
                
                if new_makespan < current_makespan:
                    current_seq = new_seq
                    current_makespan = new_makespan
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # Try block moves (2 or 3 jobs) if time permits
        for block_size in [2, 3]:
            if time.time() >= deadline:
                return current_seq, current_makespan
            for i in range(n_jobs - block_size + 1):
                if time.time() >= deadline:
                    return current_seq, current_makespan
                for j in range(n_jobs - block_size + 1):
                    if j == i:
                        continue
                    
                    new_seq = _block_move(current_seq, i, j, block_size)
                    if new_seq == current_seq:
                        continue
                    
                    new_makespan = _calc_makespan(processing_times, new_seq)
                    if new_makespan < current_makespan:
                        current_seq = new_seq
                        current_makespan = new_makespan
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return current_seq, current_makespan


def _simulated_annealing(processing_times, sequence, deadline):
    """
    Simulated annealing with insertion perturbations.
    """
    n_jobs = len(sequence)
    if n_jobs < 2:
        return list(sequence), _calc_makespan(processing_times, sequence)
    
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    T = current_makespan * 0.1
    cooling_rate = 0.9995
    min_T = 1e-6
    
    while time.time() < deadline and T > min_T:
        # Use insertion move for perturbation
        i, j = random.sample(range(n_jobs), 2)
        if i == j:
            continue
        new_seq = _insertion_move(current_seq, i, j)
        new_makespan = _calc_makespan(processing_times, new_seq)
        
        delta = new_makespan - current_makespan
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_seq = new_seq
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = list(current_seq)
        
        T *= cooling_rate
    
    return best_seq, best_makespan


def _generate_candidates(processing_times, n_jobs, n_candidates=10):
    """
    Generate diverse initial candidates.
    """
    candidates = []
    
    # NEH heuristic
    candidates.append(_neh_heuristic(processing_times))
    
    # Random permutations
    for _ in range(5):
        candidates.append(_random_permutation(n_jobs))
    
    # Reversed NEH
    candidates.append(list(reversed(_neh_heuristic(processing_times))))
    
    # Shifted NEH
    base = _neh_heuristic(processing_times)
    for shift in [1, 2, 3]:
        if shift < n_jobs:
            candidates.append(list(base[shift:]) + list(base[:shift]))
    
    # Random swaps on NEH
    for _ in range(3):
        base = _neh_heuristic(processing_times)
        if n_jobs >= 2:
            i, j = random.sample(range(n_jobs), 2)
            new_seq = list(base)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            candidates.append(new_seq)
    
    return candidates


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Phase 1: Generate diverse candidates and find the best
    candidates = _generate_candidates(processing_times, n_jobs)
    
    best_seq = None
    best_makespan = float('inf')
    
    for cand in candidates:
        if time.time() >= deadline:
            break
        ms = _calc_makespan(processing_times, cand)
        if ms < best_makespan:
            best_makespan = ms
            best_seq = list(cand)
    
    if best_seq is None:
        best_seq = list(range(n_jobs))
        best_makespan = _calc_makespan(processing_times, best_seq)
    
    # Phase 2: Local search on the best candidate
    # Allocate 40s to initial LS
    if time.time() < deadline - 80.0:
        ls_deadline = min(deadline - 80.0, time.time() + 40.0)
        ls_seq, ls_makespan = _local_search(processing_times, best_seq, ls_deadline)
        if ls_makespan < best_makespan:
            best_seq = ls_seq
            best_makespan = ls_makespan
    
    # Phase 3: Simulated annealing
    # Allocate 50s to SA
    if time.time() < deadline - 20.0:
        sa_deadline = min(deadline - 20.0, time.time() + 50.0)
        sa_seq, sa_makespan = _simulated_annealing(processing_times, best_seq, sa_deadline)
        if sa_makespan < best_makespan:
            best_seq = sa_seq
            best_makespan = sa_makespan
    
    # Phase 4: Final local search polish
    # Use remaining time for final LS
    if time.time() < deadline - 1.0:
        final_seq, final_makespan = _local_search(processing_times, best_seq, deadline)
        if final_makespan < best_makespan:
            best_seq = final_seq
            best_makespan = final_makespan
    
    return best_seq


# EVOLVE-BLOCK-END
