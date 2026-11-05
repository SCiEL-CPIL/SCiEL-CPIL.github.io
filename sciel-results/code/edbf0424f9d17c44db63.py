"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a given permutation of job indices."""
    n_jobs, n_machines = pt.shape
    # completion times per machine
    # pt[perm] gives processing times in the order of the permutation
    # We simulate the flow shop
    # c[j][m] = completion time of job j on machine m
    # c[j][0] = pt[j][0]
    # c[j][m] = max(c[j][m-1], c[j-1][m]) + pt[j][m]
    # We only need the last job's completion time on the last machine for makespan
    # But we need intermediate values. Let's use a vectorized approach where possible.
    # For efficiency, we simulate machine by machine or job by job.
    # Job by job:
    # Let C be a 1D array of size n_machines, where C[m] is the completion time of the current job on machine m
    # Actually, it's easier to keep track of the completion time of the previous job on each machine.
    # Let's use a 1D array 'machine_time' of size n_machines, initialized to 0.
    # For each job j in perm:
    #   job_start = 0
    #   for m in range(n_machines):
    #       job_start = max(job_start, machine_time[m])
    #       job_end = job_start + pt[j, m]
    #       machine_time[m] = job_end
    #   # After the loop, machine_time[m] is the completion time of job j on machine m
    # Makespan is machine_time[-1] after processing all jobs.
    
    machine_time = np.zeros(n_machines, dtype=np.float64)
    for j in perm:
        job_start = 0.0
        for m in range(n_machines):
            if machine_time[m] > job_start:
                job_start = machine_time[m]
            job_end = job_start + pt[j, m]
            machine_time[m] = job_end
    return machine_time[-1]


def _makespan_fast(perm, pt):
    """Faster makespan computation using numpy operations where possible."""
    n_jobs, n_machines = pt.shape
    if n_jobs == 0:
        return 0.0
    if n_machines == 1:
        return pt[perm].sum()
    
    # We can use a more vectorized approach for the first and last machines, but the middle ones require sequential dependency.
    # Let's stick with the loop but optimize it slightly.
    machine_time = np.zeros(n_machines, dtype=np.float64)
    for j in perm:
        # job_start is the max of the previous job's completion on this machine (machine_time[m]) and the current job's completion on the previous machine
        # For m=0, job_start = machine_time[0]
        # For m>0, job_start = max(machine_time[m], job_end_prev_machine)
        job_start = machine_time[0]
        job_end = job_start + pt[j, 0]
        machine_time[0] = job_end
        for m in range(1, n_machines):
            if machine_time[m] > job_end:
                job_start = machine_time[m]
            else:
                job_start = job_end
            job_end = job_start + pt[j, m]
            machine_time[m] = job_end
    return machine_time[-1]


def _neh_order(pt):
    """NEH heuristic: insert jobs one by one in order of decreasing total processing time."""
    n_jobs, n_machines = pt.shape
    order = np.argsort(-pt.sum(axis=1))
    perm = []
    for j in order:
        # Find best position to insert j
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [j] + perm[pos:]
            ms = _makespan_fast(new_perm, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, j)
    return perm


def _two_opt(perm, pt, max_iter=1000, deadline=None):
    """2-opt local search: reverse segments of the permutation."""
    n_jobs = len(perm)
    best_ms = _makespan_fast(perm, pt)
    best_perm = list(perm)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        if deadline is not None and time.time() > deadline:
            break
        for i in range(n_jobs - 1):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i + 1, n_jobs):
                new_perm = list(perm[:i]) + list(perm[i:j+1][::-1]) + list(perm[j+1:])
                ms = _makespan_fast(new_perm, pt)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_perm = new_perm
                    perm = new_perm
                    improved = True
                    break
            if improved:
                break
    return best_perm, best_ms


def _or_opt(perm, pt, block_sizes=(1, 2, 3), max_iter=1000, deadline=None):
    """Or-Opt local search: move blocks of size 1-3 to other positions."""
    n_jobs = len(perm)
    best_ms = _makespan_fast(perm, pt)
    best_perm = list(perm)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        if deadline is not None and time.time() > deadline:
            break
        for bsize in block_sizes:
            if deadline is not None and time.time() > deadline:
                break
            for i in range(n_jobs - bsize):
                if deadline is not None and time.time() > deadline:
                    break
                block = perm[i:i+bsize]
                # Remove block
                remaining = perm[:i] + perm[i+bsize:]
                # Try inserting block at each position in remaining
                for j in range(len(remaining) + 1):
                    if j == i:  # Skip original position
                        continue
                    new_perm = remaining[:j] + block + remaining[j:]
                    ms = _makespan_fast(new_perm, pt)
                    if ms < best_ms - 1e-9:
                        best_ms = ms
                        best_perm = new_perm
                        perm = new_perm
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return best_perm, best_ms


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial solution using NEH
    best_perm = _neh_order(pt)
    best_ms = _makespan_fast(best_perm, pt)
    
    # Local search: Or-Opt followed by 2-opt
    if time.time() < deadline:
        best_perm, best_ms = _or_opt(best_perm, pt, deadline=deadline)
    
    if time.time() < deadline:
        best_perm, best_ms = _two_opt(best_perm, pt, deadline=deadline)
    
    # Multi-start: try a few random perturbations if time allows
    rng = np.random.default_rng(42)
    n_starts = 0
    max_starts = 20
    
    while n_starts < max_starts and time.time() < deadline - 5.0:
        n_starts += 1
        # Perturb the best solution: random swap or reverse
        perm = list(best_perm)
        if len(perm) > 1:
            if rng.random() < 0.5:
                # Random swap
                i, j = rng.choice(len(perm), size=2, replace=False)
                perm[i], perm[j] = perm[j], perm[i]
            else:
                # Random reverse
                i, j = sorted(rng.choice(len(perm), size=2, replace=False))
                perm[i:j+1] = perm[i:j+1][::-1]
        
        ms = _makespan_fast(perm, pt)
        
        # Local search from perturbed solution
        if time.time() < deadline - 5.0:
            perm, ms = _or_opt(perm, pt, deadline=deadline - 5.0)
        
        if time.time() < deadline - 5.0:
            perm, ms = _two_opt(perm, pt, deadline=deadline - 5.0)
        
        if ms < best_ms - 1e-9:
            best_ms = ms
            best_perm = perm
    
    return list(best_perm)
# EVOLVE-BLOCK-END
