"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _calc_makespan_list(processing_times_list, perm):
    """
    Calculate makespan using pure Python lists for speed in the inner loop.
    processing_times_list: list of lists [job][machine]
    perm: list of job indices
    """
    n_machines = len(processing_times_list[0]) if processing_times_list else 0
    if not perm:
        return 0.0
        
    # completion[m] stores the completion time of the current job on machine m
    completion = [0.0] * n_machines
    
    for job in perm:
        p = processing_times_list[job]
        for m in range(n_machines):
            if m == 0:
                completion[0] += p[0]
            else:
                # completion[m] = max(completion[m], completion[m-1]) + p[m]
                # Using if/else is faster than max() in tight loops sometimes, 
                # or just inline logic.
                if completion[m] < completion[m-1]:
                    completion[m] = completion[m-1] + p[m]
                else:
                    completion[m] = completion[m] + p[m]
    
    return completion[-1]

def _neh_heuristic_list(processing_times_list, n_jobs, n_machines):
    """
    NEH heuristic using list-based makespan calculation.
    """
    # Sort by total processing time descending
    totals = [sum(processing_times_list[j]) for j in range(n_jobs)]
    order = list(range(n_jobs))
    order.sort(key=lambda x: totals[x], reverse=True)
    
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting at each position
        for pos in range(len(sequence) + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = _calc_makespan_list(processing_times_list, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence

def _local_search_list(processing_times_list, sequence, deadline, n_jobs):
    """
    2-opt local search using list-based makespan.
    Tries all swaps (i, j).
    """
    current_seq = list(sequence)
    current_makespan = _calc_makespan_list(processing_times_list, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_jobs - 1):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n_jobs):
                if time.time() >= deadline:
                    break
                # Swap i and j
                current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
                new_makespan = _calc_makespan_list(processing_times_list, current_seq)
                if new_makespan < current_makespan:
                    current_makespan = new_makespan
                    improved = True
                else:
                    # Revert
                    current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
    
    return current_seq, current_makespan

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # Convert to list of lists for faster access in inner loops
    p_list = processing_times.tolist()
    
    # Strategy: Multi-Start with different NEH variations if time permits
    best_seq = []
    best_makespan = float('inf')
    
    # Start 1: Standard NEH (Total Processing Time)
    seq1 = _neh_heuristic_list(p_list, n_jobs, n_machines)
    seq1, ms1 = _local_search_list(p_list, seq1, deadline, n_jobs)
    if ms1 < best_makespan:
        best_makespan = ms1
        best_seq = seq1
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < 10.0:
        # Start 2: NEH based on Last Machine Processing Time (often effective for flow shop)
        # Sort by processing time on the last machine
        last_machine_times = [p_list[j][-1] for j in range(n_jobs)]
        order2 = list(range(n_jobs))
        order2.sort(key=lambda x: last_machine_times[x], reverse=True)
        
        # Modify NEH to use this order
        sequence = [order2[0]]
        for i in range(1, n_jobs):
            job = order2[i]
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(sequence) + 1):
                new_seq = sequence[:pos] + [job] + sequence[pos:]
                ms = _calc_makespan_list(p_list, new_seq)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            sequence.insert(best_pos, job)
        
        seq2, ms2 = _local_search_list(p_list, sequence, deadline, n_jobs)
        if ms2 < best_makespan:
            best_makespan = ms2
            best_seq = seq2
            
        # If still lots of time, we could do a third start or more aggressive local search
        # But 2 starts is a safe bet for robustness.
        
    return best_seq

# EVOLVE-BLOCK-END
