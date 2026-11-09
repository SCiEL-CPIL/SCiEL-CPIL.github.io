"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def _makespan(perm, p):
    """Compute makespan for a given permutation. p shape: (n_jobs, n_machines)"""
    n_jobs, n_machines = p.shape
    if n_jobs == 0:
        return 0.0
    # completion[job, machine] = time job finishes on machine
    # completion[j][m] = max(completion[j-1][m] if j>0 else 0, completion[j][m-1] if m>0 else 0) + p[j][m]
    # We process jobs in order of perm, machines 0..n_machines-1
    # Let's use a vectorized approach for speed
    # machine_times[m] = completion time of last job on machine m
    # For each job in perm order:
    #   for m in range(n_machines):
    #       machine_times[m] = max(machine_times[m], prev_machine_end) + p[job, m]
    #       prev_machine_end = machine_times[m]
    machine_times = np.zeros(n_machines)
    total = 0.0
    for j in perm:
        prev = 0.0
        for m in range(n_machines):
            if machine_times[m] < prev:
                machine_times[m] = prev
            machine_times[m] += p[j, m]
            prev = machine_times[m]
        total = prev
    return total


def _neh_order(p):
    """NEH heuristic: sort by total processing time descending, then insert each job at best position."""
    n_jobs, n_machines = p.shape
    totals = p.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Build sequence incrementally
    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, p)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    return seq


def _local_search(perm, p, max_iters=200, time_limit=10.0):
    """2-opt / adjacent swap local search."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, p)
    start = time.time()
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() - start > time_limit:
            break
        for i in range(n_jobs - 1):
            # Try swapping adjacent jobs i and i+1
            new_perm = list(best_perm)
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            ms = _makespan(new_perm, p)
            if ms < best_ms:
                best_perm = new_perm
                best_ms = ms
                improved = True
                # Restart scan from beginning after improvement
                break
    return best_perm


def _simulated_annealing(perm, p, time_limit=8.0, T0=1.0, T_end=0.01, alpha=0.999):
    """Simulated annealing with 2-opt moves (insertion)."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, p)
    curr_perm = list(best_perm)
    curr_ms = best_ms
    
    start = time.time()
    T = T0
    
    while time.time() - start < time_limit and T > T_end:
        # Generate neighbor: remove a job and insert at random position
        i = np.random.randint(n_jobs)
        job = curr_perm[i]
        new_perm = curr_perm[:i] + curr_perm[i+1:]
        j = np.random.randint(len(new_perm) + 1)
        new_perm = new_perm[:j] + [job] + new_perm[j:]
        
        new_ms = _makespan(new_perm, p)
        delta = new_ms - curr_ms
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            curr_perm = new_perm
            curr_ms = new_ms
            if curr_ms < best_ms:
                best_perm = list(curr_perm)
                best_ms = curr_ms
        
        T *= alpha
    
    return best_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Parameters:
        processing_times: np.ndarray of shape (n_jobs, n_machines)
    
    Returns:
        list of job indices (permutation)
    """
    p = np.asarray(processing_times)
    n_jobs, n_machines = p.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    global_start = time.time()
    total_time_limit = 140.0  # Safety margin under 150s wall-clock
    
    # Phase 1: NEH heuristic (fast, good starting point)
    neh_perm = _neh_order(p)
    best_perm = list(neh_perm)
    best_ms = _makespan(best_perm, p)
    
    # Phase 2: Local search from NEH
    ls_perm = _local_search(best_perm, p, max_iters=300, time_limit=5.0)
    ls_ms = _makespan(ls_perm, p)
    if ls_ms < best_ms:
        best_perm = ls_perm
        best_ms = ls_ms
    
    # Phase 3: Simulated annealing with multiple restarts
    elapsed = time.time() - global_start
    remaining = total_time_limit - elapsed
    
    if remaining > 5.0:
        # Run SA from best solution
        sa_time = min(remaining * 0.4, 20.0)
        if sa_time > 1.0:
            sa_perm = _simulated_annealing(best_perm, p, time_limit=sa_time)
            sa_ms = _makespan(sa_perm, p)
            if sa_ms < best_ms:
                best_perm = sa_perm
                best_ms = sa_ms
        
        # Restart SA from NEH with different random seed
        elapsed = time.time() - global_start
        remaining = total_time_limit - elapsed
        if remaining > 5.0:
            sa_time = min(remaining * 0.3, 15.0)
            if sa_time > 1.0:
                np.random.seed(42)
                sa_perm2 = _simulated_annealing(neh_perm, p, time_limit=sa_time)
                sa_ms2 = _makespan(sa_perm2, p)
                if sa_ms2 < best_ms:
                    best_perm = sa_perm2
                    best_ms = sa_ms2
        
        # One more local search from best
        elapsed = time.time() - global_start
        remaining = total_time_limit - elapsed
        if remaining > 3.0:
            ls_time = min(remaining * 0.3, 10.0)
            if ls_time > 1.0:
                ls_perm2 = _local_search(best_perm, p, max_iters=200, time_limit=ls_time)
                ls_ms2 = _makespan(ls_perm2, p)
                if ls_ms2 < best_ms:
                    best_perm = ls_perm2
                    best_ms = ls_ms2
    
    # Final safety check
    if time.time() - global_start > total_time_limit - 2.0:
        # Return best found so far
        pass
    
    return list(best_perm)
# EVOLVE-BLOCK-END
