"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(seq, pt):
    """Compute makespan for a given sequence."""
    n_jobs, n_machines = pt.shape
    # Use a vectorized approach for speed
    # For each machine, the completion time is the max of (previous machine completion + processing time)
    # We can compute this iteratively but vectorized per machine
    
    # Actually, for small n, a simple loop is fine. Let's use a numpy-friendly approach.
    # completion[j, m] = max(completion[j, m-1], completion[j-1, m]) + pt[j, m]
    # But this is sequential in both dimensions.
    
    # A faster approach: for each machine m, the completion time of job j on machine m is:
    # C(j, m) = max(C(j-1, m), C(j, m-1)) + p(j, m)
    # This is the standard DP for flow shop.
    
    # Let's implement it with a 1D array for current machine and previous machine.
    # C_prev[j] = completion time of job j on machine m-1
    # C_curr[j] = completion time of job j on machine m
    
    n = len(seq)
    if n == 0:
        return 0.0
    
    # C[j] for current machine, initialized to 0
    # We process jobs in the order of seq
    # For machine 0: C[j] = C[j-1] + pt[seq[j], 0]
    # For machine m>0: C[j] = max(C[j-1], C_prev[j]) + pt[seq[j], m]
    
    C_prev = np.zeros(n, dtype=np.float64)
    C_curr = np.zeros(n, dtype=np.float64)
    
    # Machine 0
    C_curr[0] = pt[seq[0], 0]
    for j in range(1, n):
        C_curr[j] = C_curr[j-1] + pt[seq[j], 0]
    C_prev = C_curr.copy()
    
    for m in range(1, n_machines):
        C_curr[0] = C_prev[0] + pt[seq[0], m]
        for j in range(1, n):
            C_curr[j] = max(C_curr[j-1], C_prev[j]) + pt[seq[j], m]
        C_prev = C_curr.copy()
    
    return C_prev[n-1]


def _neh(pt):
    """NEH heuristic: sort by decreasing total processing time, then insert each job in best position."""
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = _makespan(new_seq, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    
    return seq


def _local_search_insertion(seq, pt, max_iters=100, time_limit=None, start_time=None):
    """Local search with insertion moves."""
    if start_time is None:
        start_time = time.time()
    
    n = len(seq)
    best_seq = list(seq)
    best_ms = _makespan(best_seq, pt)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Move job from position i to position j
                new_seq = list(best_seq)
                job = new_seq.pop(i)
                new_seq.insert(j, job)
                
                ms = _makespan(new_seq, pt)
                if ms < best_ms - 1e-10:
                    best_ms = ms
                    best_seq = new_seq
                    improved = True
    
    return best_seq, best_ms


def _perturb(seq, pt, strength=2):
    """Perturb the sequence by random 2-opt or insertion moves."""
    n = len(seq)
    new_seq = list(seq)
    
    # Randomly select a segment and reverse it
    if n >= 2:
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        new_seq[i:j+1] = new_seq[i:j+1][::-1]
    
    return new_seq


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using NEH + local search + perturbations.
    """
    n_jobs, n_machines = processing_times.shape
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Stage 1: NEH initial solution
    seq = _neh(processing_times)
    best_seq = list(seq)
    best_ms = _makespan(best_seq, processing_times)
    
    # Stage 2: Local search refinement
    if time.time() < deadline:
        best_seq, best_ms = _local_search_insertion(
            best_seq, processing_times, 
            max_iters=50, 
            time_limit=deadline - time.time()
        )
    
    # Stage 3: Perturbation-based restarts (multi-start)
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        perturbed = _perturb(best_seq, processing_times)
        refined_seq, refined_ms = _local_search_insertion(
            perturbed, processing_times,
            max_iters=30,
            time_limit=deadline - time.time()
        )
        
        if refined_ms < best_ms - 1e-10:
            best_ms = refined_ms
            best_seq = list(refined_seq)
        
        # If we've done many restarts without improvement, stop
        if n_restarts > 50:
            break
    
    return best_seq


# EVOLVE-BLOCK-END
