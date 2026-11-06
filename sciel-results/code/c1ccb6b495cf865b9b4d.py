"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    # C[j, k] = completion time of job j on machine k
    # We compute iteratively
    c = np.zeros((n, m), dtype=np.float64)
    for j_idx, job in enumerate(perm):
        for k in range(m):
            if j_idx == 0:
                c[j_idx, k] = c[j_idx, k-1] + pt[job, k] if k > 0 else pt[job, k]
            else:
                c[j_idx, k] = max(c[j_idx, k-1] if k > 0 else 0.0, c[j_idx-1, k]) + pt[job, k]
    return c[-1, -1]


def _makespan_fast(perm, pt):
    """Vectorized makespan computation."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    c = np.zeros((n, m), dtype=np.float64)
    for j_idx in range(n):
        job = perm[j_idx]
        row = pt[job]
        if j_idx == 0:
            c[j_idx, 0] = row[0]
            if m > 1:
                c[j_idx, 1:] = np.cumsum(row[1:]) + c[j_idx, 0]
        else:
            c[j_idx, 0] = c[j_idx-1, 0] + row[0]
            if m > 1:
                for k in range(1, m):
                    c[j_idx, k] = max(c[j_idx, k-1], c[j_idx-1, k]) + row[k]
    return c[-1, -1]


def _makespan_vec(perm, pt):
    """Another vectorized approach."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    # Build the matrix of processing times in order
    ordered = pt[perm]  # shape (n, m)
    # C[0, :] = cumsum of first row
    # C[j, k] = max(C[j, k-1], C[j-1, k]) + ordered[j, k]
    c = np.zeros((n, m), dtype=np.float64)
    c[0, 0] = ordered[0, 0]
    if m > 1:
        c[0, 1:] = np.cumsum(ordered[0, 1:]) + ordered[0, 0]
    for j in range(1, n):
        c[j, 0] = c[j-1, 0] + ordered[j, 0]
        for k in range(1, m):
            c[j, k] = max(c[j, k-1], c[j-1, k]) + ordered[j, k]
    return c[-1, -1]


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using:
    1. NEH heuristic for initial construction
    2. Best-insertion local search with move and swap neighborhoods
    3. Deadline guard to ensure we finish within time limits
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # --- Stage 1: NEH Heuristic ---
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    # Build sequence using NEH rule: insert each job at position minimizing makespan
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [j] + seq[pos:]
            ms = _makespan_vec(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    
    current_best = list(seq)
    current_best_ms = _makespan_vec(current_best, processing_times)
    
    # --- Stage 2: Local Search ---
    # Try multiple restarts with perturbation if time allows
    best_perm = list(current_best)
    best_ms = current_best_ms
    
    while time.time() < deadline - 5.0:  # Keep 5s safety margin
        improved = False
        
        # Phase A: Best insertion moves
        n = len(current_best)
        for i in range(n):
            if time.time() > deadline - 2.0:
                break
            job = current_best[i]
            remaining = [j for j in range(n) if j != i]
            # Try inserting job at each position in the remaining sequence
            for pos in range(len(remaining) + 1):
                candidate = remaining[:pos] + [job] + remaining[pos:]
                # Map back to job indices
                cand_jobs = [current_best[j] if j < i else current_best[j+1] for j in range(len(remaining))]
                # Actually let's just build the permutation directly
                cand_perm = current_best[:i] + current_best[i+1:]
                cand_perm = cand_perm[:pos] + [job] + cand_perm[pos:]
                ms = _makespan_vec(cand_perm, processing_times)
                if ms < current_best_ms - 1e-10:
                    current_best = cand_perm
                    current_best_ms = ms
                    improved = True
                    break  # Restart from new position
        
        # Phase B: Best swap moves
        for i in range(n - 1):
            if time.time() > deadline - 2.0:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 1.0:
                    break
                cand_perm = list(current_best)
                cand_perm[i], cand_perm[j] = cand_perm[j], cand_perm[i]
                ms = _makespan_vec(cand_perm, processing_times)
                if ms < current_best_ms - 1e-10:
                    current_best = cand_perm
                    current_best_ms = ms
                    improved = True
                    break
            if improved:
                break
        
        if current_best_ms < best_ms - 1e-10:
            best_perm = list(current_best)
            best_ms = current_best_ms
        
        if not improved:
            # Local optimum reached; try perturbation
            if n_jobs <= 20:
                # 2-opt style perturbation: reverse a segment
                if n >= 2:
                    # Pick two random positions and reverse
                    p1 = np.random.randint(0, n - 1)
                    p2 = np.random.randint(p1 + 1, n)
                    perturbed = current_best[:p1] + current_best[p1:p2+1][::-1] + current_best[p2+1:]
                    pm = _makespan_vec(perturbed, processing_times)
                    # Accept if not too bad, or use worst-case acceptance
                    if pm < current_best_ms * 1.5:
                        current_best = perturbed
                        current_best_ms = pm
                    else:
                        # Try a random swap
                        i, j = np.random.choice(n, 2, replace=False)
                        current_best = list(current_best)
                        current_best[i], current_best[j] = current_best[j], current_best[i]
                        current_best_ms = _makespan_vec(current_best, processing_times)
                break  # If perturbation doesn't help, we're likely at a good local opt
            else:
                break
    
    # Final check: if we improved during local search, use best found
    if best_ms < current_best_ms:
        return best_perm
    return current_best
# EVOLVE-BLOCK-END
