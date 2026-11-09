"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def _makespan(pt, perm):
    """Compute makespan for a given permutation."""
    n_jobs, n_machines = pt.shape
    if len(perm) == 0:
        return 0
    # Build completion time matrix
    # C[j][m] = completion time of job j on machine m
    # We process in order of perm
    # For each machine, completion time accumulates
    # C[j][0] = sum of processing times of all jobs before j in perm on machine 0 + pt[j][0]
    # C[j][m] = max(C[j][m-1], C[j_prev_on_m][m]) + pt[j][m]
    # Simpler: simulate
    
    # comp[j] = completion time of job j on last machine
    # We need to track when each machine becomes free
    machine_free = np.zeros(n_machines)
    for j in perm:
        for m in range(n_machines):
            machine_free[m] = max(machine_free[m], pt[j][m])
            # Wait, this is wrong. In flow shop, job j on machine m must wait until:
            # 1. Machine m is free (i.e., previous job on machine m is done)
            # 2. Job j is done on machine m-1
            # So we need to track completion of job j on each machine
    
    # Let's redo properly
    machine_free = np.zeros(n_machines)
    for j in perm:
        for m in range(n_machines):
            if m == 0:
                # Job j on machine 0: must wait until machine 0 is free
                machine_free[m] += pt[j][m]
            else:
                # Job j on machine m: must wait until both machine m is free AND job j finished on m-1
                # The time job j finishes on m-1 is the completion time we just computed for m-1
                # But we need to be careful: machine_free[m-1] at this point is the completion of job j on m-1
                # Actually no. Let me think again.
                # 
                # Let's use a different approach:
                # C[j][m] = max(C[j][m-1], C[prev_job_on_m][m]) + pt[j][m]
                # where prev_job_on_m is the job immediately before j in the permutation (since it's a permutation, all jobs go through all machines in the same order)
                # Wait, that's not right either. The previous job on machine m is the job that came before j in the permutation.
                # 
                # Actually for flow shop with permutation schedule:
                # C[j][0] = sum_{k in perm, k before j} pt[k][0] + pt[j][0]
                # C[j][m] = max(C[j][m-1], C[j_prev][m]) + pt[j][m]
                # where j_prev is the job immediately preceding j in perm.
                
                pass
    
    # Let me just do it the straightforward way
    # completion[j_index_in_perm][m] 
    n = len(perm)
    if n == 0:
        return 0
    comp = np.zeros((n, n_machines))
    for idx, j in enumerate(perm):
        for m in range(n_machines):
            if m == 0:
                comp[idx][m] = comp[idx-1][m] + pt[j][m] if idx > 0 else pt[j][m]
            else:
                # Job j on machine m: max of (job j done on m-1, previous job in perm done on m)
                comp[idx][m] = max(comp[idx][m-1], comp[idx-1][m]) + pt[j][m]
    return comp[-1][-1]


def _neh_ordering(pt, rng):
    """NEH heuristic: insert jobs one by one in order of decreasing total processing time."""
    n_jobs, n_machines = pt.shape
    # Sort by total processing time descending
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    # Insert one job at a time
    current_perm = []
    for j in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current_perm) + 1):
            trial = current_perm[:pos] + [j] + current_perm[pos:]
            ms = _makespan(pt, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current_perm = current_perm[:best_pos] + [j] + current_perm[best_pos:]
    return current_perm


def _swap_and_insert_neighborhood(pt, perm, rng):
    """Generate neighbors via swaps and insertions."""
    n = len(perm)
    if n <= 1:
        return [perm]
    
    neighbors = []
    
    # Swaps
    for i in range(n):
        for j in range(i+1, n):
            new_perm = perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            neighbors.append(new_perm)
    
    # Insertions (Or-opt style: remove job and insert elsewhere)
    for i in range(n):
        j = perm[i]
        remaining = [x for x in perm if x != j]
        for pos in range(len(remaining) + 1):
            if pos == i or (pos > 0 and remaining[pos-1] == j):
                continue
            new_perm = remaining[:pos] + [j] + remaining[pos:]
            if new_perm != perm:
                neighbors.append(new_perm)
    
    return neighbors


def _or_opt_moves(pt, perm, rng):
    """Or-opt: move blocks of size 1-3."""
    n = len(perm)
    neighbors = []
    for block_size in [1, 2, 3]:
        for start in range(n - block_size + 1):
            block = perm[start:start+block_size]
            rest = perm[:start] + perm[start+block_size:]
            for pos in range(len(rest) + 1):
                new_perm = rest[:pos] + block + rest[pos:]
                if new_perm != perm:
                    neighbors.append(new_perm)
    return neighbors


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses NEH initialization with local search (swap, insert, Or-opt moves).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    # Generate multiple NEH starts with random tie-breaking
    best_perm = None
    best_ms = float('inf')
    
    n_starts = min(20, max(4, n_jobs // 2))
    
    for s in range(n_starts):
        if time.time() > deadline:
            break
        
        # NEH with random perturbation in ordering
        totals = pt.sum(axis=1)
        # Add small random noise to break ties
        if s == 0:
            order = np.argsort(-totals)
        else:
            # Random perturbation
            noise = rng.random() * 0.1 * totals.max() if totals.max() > 0 else 0
            noisy_totals = totals + rng.random(n_jobs) * noise
            order = np.argsort(-noisy_totals)
        
        # NEH insertion
        current_perm = []
        for j in order:
            best_pos = 0
            best_local_ms = float('inf')
            for pos in range(len(current_perm) + 1):
                trial = current_perm[:pos] + [j] + current_perm[pos:]
                ms = _makespan(pt, trial)
                if ms < best_local_ms:
                    best_local_ms = ms
                    best_pos = pos
            current_perm = current_perm[:best_pos] + [j] + current_perm[best_pos:]
        
        ms = _makespan(pt, current_perm)
        if ms < best_ms:
            best_ms = ms
            best_perm = current_perm[:]
    
    if best_perm is None:
        return list(range(n_jobs))
    
    # Local search phase
    current_perm = best_perm
    current_ms = best_ms
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Generate neighborhood
        neighbors = _swap_and_insert_neighborhood(pt, current_perm, rng)
        if time.time() < deadline - 2:
            neighbors += _or_opt_moves(pt, current_perm, rng)
        
        if not neighbors:
            break
        
        # Shuffle for diversity
        rng.shuffle(neighbors)
        
        improved = False
        for new_perm in neighbors:
            if time.time() > deadline:
                break
            ms = _makespan(pt, new_perm)
            if ms < current_ms:
                current_perm = new_perm
                current_ms = ms
                improved = True
                no_improve_count = 0
                break
        if not improved:
            no_improve_count += 1
    
    if current_ms < best_ms:
        best_perm = current_perm
        best_ms = current_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
