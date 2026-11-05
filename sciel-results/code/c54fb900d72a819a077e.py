"""Flow-shop permutation via direct heuristics + fast insertion local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan_fast(pt_t, perm):
    """
    Fast makespan calculation using pure Python lists for the state vector.
    pt_t: (n_machines, n_jobs) numpy array.
    perm: list of job indices.
    """
    n_machines = pt_t.shape[0]
    n = len(perm)
    if n == 0:
        return 0.0
    
    # Unpack pt_t rows into Python lists for faster indexing if needed, 
    # but pt_t[m][job] is usually fast enough.
    # To maximize speed, we can convert pt_t to a list of lists once.
    # However, pt_t is passed in. Let's assume pt_t is numpy.
    
    prev = [0.0] * n_machines
    # Optimization: local variable references
    pt_rows = pt_t
    for job in perm:
        curr = [0.0] * n_machines
        curr[0] = prev[0] + pt_rows[0][job]
        for m in range(1, n_machines):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + pt_rows[m][job]
        prev = curr
    return prev[-1]


def _neh_initial(pt):
    """NEH heuristic."""
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    perm = [order[0]]
    pt_t = pt.T
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            test = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(pt_t, test)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm


def _palmer_initial(pt):
    """Palmer's dispatching rule."""
    n_jobs, n_machines = pt.shape
    n = n_machines
    scores = np.zeros(n_jobs)
    for j in range(n_jobs):
        s = 0.0
        for m in range(n):
            if m < n // 2:
                s += (n - 1 - m) * pt[j, m]
            else:
                s -= m * pt[j, m]
        scores[j] = s
    return list(np.argsort(-scores))


def _lpt_initial(pt):
    totals = pt.sum(axis=1)
    return list(np.argsort(-totals))


def _bottleneck_initial(pt):
    """Sort by total processing on the machine with largest total load, descending."""
    n_jobs, n_machines = pt.shape
    machine_loads = pt.sum(axis=0)
    bottleneck = int(np.argmax(machine_loads))
    return list(np.argsort(-pt[:, bottleneck]))


def _mean_initial(pt):
    return list(np.argsort(-pt.mean(axis=1)))


def _first_machine_initial(pt):
    return list(np.argsort(-pt[:, 0]))


def _last_machine_initial(pt):
    return list(np.argsort(-pt[:, -1]))


def _random_initial(n_jobs, seed):
    rng = random.Random(seed)
    perm = list(range(n_jobs))
    rng.shuffle(perm)
    return perm


def _local_search_insertion(pt_t, perm, deadline, max_rounds=2000):
    """
    Local search using insertion neighborhood: remove job at position i, insert at position j.
    This is stronger than adjacent swap for flow-shop.
    """
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_t, perm)

    current = list(perm)
    current_ms = _makespan_fast(pt_t, current)
    best = list(current)
    best_ms = current_ms

    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            job = current[i]
            # remove job at i
            rest = current[:i] + current[i+1:]
            for j in range(n):
                if j == i or j == i + 1:
                    continue
                if time.time() > deadline:
                    break
                test = rest[:j] + [job] + rest[j:]
                ms = _makespan_fast(pt_t, test)
                if ms < current_ms - 1e-9:
                    current_ms = ms
                    current = test
                    improved = True
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best = list(current)
                    break  # restart i loop with new current
            if improved:
                # restart to keep neighborhood consistent
                break
        if not improved:
            break
    return best, best_ms


def _local_search_swap(pt_t, perm, deadline, max_rounds=5000):
    """Local search using adjacent swap."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_t, perm)
    current = list(perm)
    current_ms = _makespan_fast(pt_t, current)
    best = list(current)
    best_ms = current_ms
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            current[i], current[i+1] = current[i+1], current[i]
            ms = _makespan_fast(pt_t, current)
            if ms < current_ms - 1e-9:
                current_ms = ms
                improved = True
                if current_ms < best_ms - 1e-9:
                    best_ms = current_ms
                    best = list(current)
            else:
                current[i], current[i+1] = current[i+1], current[i]
        if not improved:
            break
    return best, best_ms


def _simulated_annealing(pt_t, perm, deadline, max_iter=20000, seed=0):
    """Simulated annealing with insertion moves."""
    rng = random.Random(seed)
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_t, perm)
    current = list(perm)
    current_ms = _makespan_fast(pt_t, current)
    best = list(current)
    best_ms = current_ms

    # Temperature based on typical delta
    T = max(current_ms * 0.05, 1.0)
    T_min = max(current_ms * 0.0005, 0.01)
    cooling = 0.995

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        # Insertion move
        i = rng.randint(0, n - 1)
        job = current[i]
        rest = current[:i] + current[i+1:]
        j = rng.randint(0, n - 1)
        if j == i or j == i + 1:
            j = (j + 1) % n
        test = rest[:j] + [job] + rest[j:]
        new_ms = _makespan_fast(pt_t, test)
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = test
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best = list(current)
        T *= cooling
        if T < T_min:
            T = T_min
    return best, best_ms


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    start = time.time()
    deadline = start + 148.0  # safety margin before 150s kill

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))

    pt_t = pt.T  # (n_machines, n_jobs)

    # Build diverse initial solutions (direct heuristics)
    candidates = [
        _neh_initial(pt),
        _palmer_initial(pt),
        _lpt_initial(pt),
        _bottleneck_initial(pt),
        _mean_initial(pt),
        _first_machine_initial(pt),
        _last_machine_initial(pt),
    ]
    # Add a few random starts
    for s in range(4):
        candidates.append(_random_initial(n_jobs, 100 + s))

    # Evaluate all, pick best
    best_ms = float('inf')
    best_perm = candidates[0]
    for cand in candidates:
        ms = _makespan_fast(pt_t, cand)
        if ms < best_ms:
            best_ms = ms
            best_perm = cand

    # Local search on best candidate first (insertion is strongest)
    perm, ms = _local_search_insertion(pt_t, best_perm, deadline - 20.0)
    if ms < best_ms:
        best_ms = ms
        best_perm = perm

    # Local search on other good candidates if time permits
    for cand in candidates[1:]:
        if time.time() > deadline - 15.0:
            break
        # Skip if too far from best
        ms0 = _makespan_fast(pt_t, cand)
        if ms0 > best_ms * 1.15:
            continue
        perm, ms = _local_search_insertion(pt_t, cand, deadline - 15.0)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Simulated annealing from best to escape local optima
    if time.time() < deadline - 5.0:
        perm, ms = _simulated_annealing(pt_t, best_perm, deadline - 5.0, max_iter=30000, seed=7)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Polish with a final insertion local search if time remains
    if time.time() < deadline - 2.0:
        perm, ms = _local_search_insertion(pt_t, best_perm, deadline - 1.0)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    return best_perm
# EVOLVE-BLOCK-END
