"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def _makespan(pt, perm):
    """Compute makespan for a given permutation on the flow-shop."""
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    # C[i][j] = completion time of job i on machine j
    # We only need the last column, but we need all columns for recurrence.
    # C[i][j] = max(C[i-1][j], C[i][j-1]) + pt[perm[i], j]
    # Base: C[-1][j] = 0, C[i][-1] = 0
    
    # Optimization: we can compute this in O(n_jobs * n_machines)
    # But for large n_jobs, this might be slow if called many times.
    # Let's use a 1D array for the previous row (machine times)
    # Actually, we need C[i][j] which depends on C[i-1][j] (previous job, same machine)
    # and C[i][j-1] (same job, previous machine).
    
    # Let's keep an array `prev_machine_time` of length n_machines.
    # prev_machine_time[j] = C[i-1][j]
    # curr_machine_time = C[i][j-1] as we iterate j
    
    prev_machine_time = np.zeros(n_machines)
    makespan = 0.0
    
    for i in range(n_jobs):
        job_idx = perm[i]
        curr_machine_time = 0.0
        for j in range(n_machines):
            # C[i][j] = max(prev_machine_time[j], curr_machine_time) + pt[job_idx, j]
            curr_machine_time = max(prev_machine_time[j], curr_machine_time) + pt[job_idx, j]
            prev_machine_time[j] = curr_machine_time
        makespan = max(makespan, prev_machine_time[-1])
    
    return makespan


def _neh_order(pt):
    """
    NEH heuristic for flow-shop.
    1. Sort jobs by total processing time in descending order.
    2. Insert each job into the best position in the current sequence.
    """
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    totals = pt.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    seq = []
    for job in sorted_jobs:
        # Try inserting job at each position in seq
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(pt, new_seq)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq.insert(best_pos, job)
    
    return seq


def _two_opt(pt, initial_seq, deadline):
    """
    Best-Improvement 2-Opt local search.
    """
    n_jobs = len(initial_seq)
    if n_jobs <= 2:
        return initial_seq, _makespan(pt, initial_seq)
    
    best_seq = initial_seq[:]
    best_ms = _makespan(pt, best_seq)
    
    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_swap = None
        
        # Check deadline
        if time.time() > deadline:
            break
        
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_jobs):
                # Swap i and j
                seq = best_seq[:]
                seq[i], seq[j] = seq[j], seq[i]
                ms = _makespan(pt, seq)
                delta = ms - best_ms
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (i, j)
        
        if best_swap is not None:
            i, j = best_swap
            best_seq[i], best_seq[j] = best_seq[j], best_seq[i]
            best_ms += best_delta
            improved = True
    
    return best_seq, best_ms


def _insertion_search(pt, initial_seq, deadline):
    """
    Best-Improvement Insertion local search (remove and reinsert).
    """
    n_jobs = len(initial_seq)
    if n_jobs <= 2:
        return initial_seq, _makespan(pt, initial_seq)
    
    best_seq = initial_seq[:]
    best_ms = _makespan(pt, best_seq)
    
    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_move = None  # (remove_idx, insert_idx)
        
        if time.time() > deadline:
            break
        
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            job = best_seq[i]
            # Remove job i
            seq_without = best_seq[:i] + best_seq[i+1:]
            
            for j in range(len(seq_without) + 1):
                # Insert job at position j in seq_without
                seq = seq_without[:j] + [job] + seq_without[j:]
                ms = _makespan(pt, seq)
                delta = ms - best_ms
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)
        
        if best_move is not None:
            i, j = best_move
            job = best_seq[i]
            seq_without = best_seq[:i] + best_seq[i+1:]
            best_seq = seq_without[:j] + [job] + seq_without[j:]
            best_ms += best_delta
            improved = True
    
    return best_seq, best_ms


def order_jobs(processing_times):
    start_time = time.time()
    # Safety margin: stop at 140s to allow for return overhead
    deadline = start_time + 140.0
    
    pt = np.asarray(processing_times)
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Generate initial solution using NEH
    seq = _neh_order(pt)
    current_ms = _makespan(pt, seq)
    
    # 2. Apply local search
    # For small instances, we can do more thorough search.
    # For large instances, we need to be careful with time.
    
    # Let's alternate between 2-Opt and Insertion, or just use Insertion which is often stronger.
    # Insertion search is O(n^3) per pass, 2-Opt is O(n^3) per pass.
    # Let's try Insertion first, then 2-Opt if time permits.
    
    # Check if we have enough time for local search
    if time.time() < deadline - 5.0:
        seq, current_ms = _insertion_search(pt, seq, deadline)
    
    if time.time() < deadline - 5.0:
        seq, current_ms = _two_opt(pt, seq, deadline)
    
    # If we have even more time, we could try restarting with different initial heuristics
    # but NEH is quite strong. Let's stick with one good pass to ensure we return in time.
    
    return seq

# EVOLVE-BLOCK-END
