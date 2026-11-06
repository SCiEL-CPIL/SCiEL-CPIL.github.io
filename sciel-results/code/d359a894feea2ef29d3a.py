"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(seq, p):
    """
    Calculate the makespan of a given job sequence.
    seq: list of job indices
    p: numpy array of shape (n_jobs, n_machines)
    """
    if not seq:
        return 0.0
    
    n_machines = p.shape[1]
    C = np.zeros(n_machines)
    
    for j in seq:
        prev_c = 0.0
        for m in range(n_machines):
            start = max(prev_c, C[m])
            end = start + p[j, m]
            C[m] = end
            prev_c = end
    return C[-1]

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using NEH heuristic 
    followed by local search with insertion moves.
    """
    start_time = time.time()
    deadline = 110.0 # Safety margin before 120s limit
    
    p = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = p.shape
    
    # 1. NEH Heuristic
    # Sort jobs by decreasing total processing time
    total_times = p.sum(axis=1)
    order = np.argsort(-total_times)
    
    seq = []
    for j in order:
        best_pos = 0
        best_ms = float('inf')
        
        # Try inserting j at every position in current seq
        for pos in range(len(seq) + 1):
            cand = seq[:pos] + [j] + seq[pos:]
            ms = _makespan(cand, p)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        # Insert j at best_pos
        seq.insert(best_pos, j)
        
        # Check time
        if time.time() > deadline:
            break

    best_seq = seq
    best_ms = _makespan(best_seq, p)
    
    # 2. Local Search (Insertion Moves)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_seq)):
            j = best_seq[i]
            for k in range(len(best_seq)):
                if k == i:
                    continue
                
                cand = best_seq[:i] + best_seq[i+1:]
                insert_idx = k if k < i else k - 1
                cand.insert(insert_idx, j)
                
                ms = _makespan(cand, p)
                if ms < best_ms:
                    best_ms = ms
                    best_seq = cand
                    improved = True
                    break
            if improved:
                break
            
    return list(best_seq)
# EVOLVE-BLOCK-END
