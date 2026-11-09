"""
Flow-shop permutation construction using Iterated Local Search (ILS) 
with 1-insertion local search and Or-opt perturbations.
Strategy:
1. Initialize with NEH heuristic.
2. Perform deterministic 1-insertion local search to reach a local optimum.
3. Perturb the solution using Or-opt moves (block size 1-3) to escape local optima.
4. Repeat until time limit.
This differs from Simulated Annealing by using a deterministic descent phase 
followed by a structured jump, rather than probabilistic acceptance of worse moves.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    Optimized for speed using local variables and minimal overhead.
    """
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    
    n_machines = P.shape[1]
    # Pre-allocate completion time array for current machine
    # C[m] stores the completion time of the current job on machine m
    # Actually, we need the completion time of the previous job on machine m to compute the current.
    # Let's use the standard recurrence:
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    # We can maintain an array 'prev_machine_time' of size n_machines, 
    # where prev_machine_time[m] is C[j-1, m].
    # And a variable 'curr_machine_time' for C[j, m-1] as we iterate m.
    
    prev_machine_time = np.zeros(n_machines)
    last_completion = 0.0
    
    for j in range(n_jobs):
        job_idx = order[j]
        p_row = P[job_idx]
        curr_machine_time = 0.0
        for m in range(n_machines):
            # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
            # prev_machine_time[m] is C[j-1, m]
            # curr_machine_time is C[j, m-1]
            if prev_machine_time[m] > curr_machine_time:
                curr_machine_time = prev_machine_time[m]
            curr_machine_time += p_row[m]
            prev_machine_time[m] = curr_machine_time
        last_completion = curr_machine_time
        
    return last_completion

def _neh_heuristic(P):
    """
    NEH heuristic for initial solution.
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def _local_search_1_insertion(seq, P):
    """
    Perform 1-insertion local search.
    Returns improved sequence and makespan.
    """
    n = len(seq)
    if n < 2:
        return seq, _makespan(seq, P)
    
    current_makespan = _makespan(seq, P)
    improved = True
    seq = list(seq)
    
    while improved:
        improved = False
        for i in range(n):
            # Remove job at i
            job = seq[i]
            new_seq = seq[:i] + seq[i+1:]
            
            # Try inserting job at all positions in new_seq
            # We can optimize by only checking positions that might improve, 
            # but for simplicity and correctness, we check all.
            # However, we can break early if we find a better makespan? 
            # No, we need the best.
            
            best_pos = -1
            best_ms = current_makespan
            
            for j in range(len(new_seq) + 1):
                candidate_seq = new_seq[:j] + [job] + new_seq[j:]
                ms = _makespan(candidate_seq, P)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = j
            
            if best_pos != -1:
                # Apply the best move
                new_seq = new_seq[:best_pos] + [job] + new_seq[best_pos:]
                seq = new_seq
                current_makespan = best_ms
                improved = True
                # Restart the loop because the sequence changed
                break
                
    return seq, current_makespan

def _perturb(seq, P, strength=1):
    """
    Perturb the sequence using Or-opt moves.
    strength determines the number of moves or block size range.
    """
    n = len(seq)
    if n < 4:
        return seq
    
    seq = list(seq)
    num_moves = strength
    for _ in range(num_moves):
        # Choose block size 1, 2, or 3
        max_block = min(3, n - 1)
        block_size = random.randint(1, max_block)
        
        # Choose start position
        start = random.randint(0, n - block_size)
        
        # Extract block
        block = seq[start:start + block_size]
        
        # Remove block
        remaining = seq[:start] + seq[start + block_size:]
        
        # Choose insertion position
        # Avoid inserting back at the same place
        valid_positions = [p for p in range(len(remaining) + 1) if p != start and p != start + 1]
        if not valid_positions:
            continue
            
        ins_pos = random.choice(valid_positions)
        
        seq = remaining[:ins_pos] + block + remaining[ins_pos:]
        
    return seq

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using Iterated Local Search.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    # 1. Initialize with NEH
    current_seq = _neh_heuristic(P)
    current_makespan = _makespan(current_seq, P)
    
    best_seq = current_seq.copy()
    best_makespan = current_makespan
    
    # 2. Iterated Local Search
    no_improvement_count = 0
    perturbation_strength = 1
    
    while (time.time() - start_time) < time_limit:
        # Check time more frequently
        if time.time() - start_time > time_limit:
            break
            
        # Local Search
        current_seq, current_makespan = _local_search_1_insertion(current_seq, P)
        
        # Update best
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_seq = current_seq.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # If stuck, increase perturbation strength
        if no_improvement_count > 5:
            perturbation_strength = min(3, perturbation_strength + 1)
            no_improvement_count = 0
        elif no_improvement_count == 0:
            perturbation_strength = 1
            
        # Perturb
        if (time.time() - start_time) > time_limit - 5:
            # If running out of time, do a quick local search on best_seq and return
            # to ensure we don't return a perturbed worse solution
            break
            
        current_seq = _perturb(current_seq, P, strength=perturbation_strength)
        
    # Final polish: run local search on the best sequence found
    best_seq, best_makespan = _local_search_1_insertion(best_seq, P)
    
    return best_seq
# EVOLVE-BLOCK-END
