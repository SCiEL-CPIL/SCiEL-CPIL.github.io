"""Stronger flow-shop makespan solver: NEH + iterated local search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """Compute makespan for a given job permutation (list or array of job indices)."""
    n = len(perm)
    if n == 0:
        return 0.0
    # completion[machine, job_position]
    # We process machine by machine; completion on machine m for position i
    # = max(completion on machine m-1 at position i, completion on machine m at position i-1) + p[perm[i], m]
    m = processing_times.shape[1]
    # completion[i] for current machine across positions
    comp = np.zeros(n, dtype=np.float64)
    for mm in range(m):
        p = processing_times[perm, mm]  # processing times for this machine in perm order
        new_comp = np.empty(n, dtype=np.float64)
        new_comp[0] = comp[0] + p[0]
        if n > 1:
            # new_comp[i] = max(comp[i], new_comp[i-1]) + p[i]
            np.maximum(comp, np.concatenate(([0.0], new_comp[:-1])), out=new_comp[1:])
            new_comp[1:] += p[1:]
        comp = new_comp
    return float(comp[-1])


def _neh_order(processing_times, rng):
    """Nawaz-Enscore-Ham heuristic: sort by descending total processing time,
    insert each job at the position minimizing makespan."""
    n, m = processing_times.shape
    totals = processing_times.sum(axis=1)
    # Descending order of totals; ties broken randomly for diversity
    order = list(rng.permutation(n))
    order.sort(key=lambda j: -totals[j])
    
    seq = []
    for j in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            cand = seq[:pos] + [j] + seq[pos:]
            ms = _makespan(cand, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    return seq


def _insertion_local_search(perm, processing_times, rng, time_limit):
    """Iterated local search using insertion moves. Returns best permutation found."""
    n = len(perm)
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    start = time.time()
    
    improved = True
    while improved:
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(n):
            if time.time() - start > time_limit:
                break
            job = best_perm[i]
            for j in range(n):
                if j == i:
                    continue
                # Move job from i to j
                if j > i:
                    new_perm = best_perm[:i] + best_perm[i+1:j+1] + [job] + best_perm[j+1:]
                else:
                    new_perm = best_perm[:j] + [job] + best_perm[j:i] + best_perm[i+1:]
                ms = _makespan(new_perm, processing_times)
                if ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
    return best_perm, best_ms


def _perturb(perm, rng, strength=2):
    """Apply random insertion perturbations to escape local optima."""
    perm = list(perm)
    n = len(perm)
    for _ in range(strength):
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
        job = perm[i]
        perm.pop(i)
        perm.insert(j, job)
    return perm


def order_jobs(processing_times):
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n, m = processing_times.shape
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Phase 1: NEH heuristic to get a strong initial solution
    best_perm = _neh_order(processing_times, rng)
    best_ms = _makespan(best_perm, processing_times)
    
    # Phase 2: Local search to polish
    if time.time() < deadline - 5.0:
        best_perm, best_ms = _insertion_local_search(
            best_perm, processing_times, rng,
            time_limit=max(1.0, deadline - time.time() - 5.0)
        )
    
    # Phase 3: Iterated local search with perturbations
    if time.time() < deadline - 5.0:
        il_start = time.time()
        while time.time() < deadline - 5.0:
            # Perturb the best solution
            perturbed = _perturb(best_perm, rng, strength=2)
            # Local search from perturbed
            cand_perm, cand_ms = _insertion_local_search(
                perturbed, processing_times, rng,
                time_limit=max(0.5, deadline - time.time() - 5.0)
            )
            if cand_ms < best_ms - 1e-9:
                best_ms = cand_ms
                best_perm = cand_perm
            # Occasionally try a fresh NEH with different tie-breaking
            if time.time() < deadline - 10.0 and rng.random() < 0.1:
                fresh = _neh_order(processing_times, rng)
                fresh_perm, fresh_ms = _insertion_local_search(
                    fresh, processing_times, rng,
                    time_limit=max(0.5, deadline - time.time() - 5.0)
                )
                if fresh_ms < best_ms - 1e-9:
                    best_ms = fresh_ms
                    best_perm = fresh_perm
    
    return list(best_perm)
# EVOLVE-BLOCK-END
