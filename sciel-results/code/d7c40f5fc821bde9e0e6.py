"""Flow-shop permutation via fast vectorized makespan + 2-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan_vec(pt_t, perm):
    """
    Vectorized makespan.
    pt_t: (n_machines, n_jobs) numpy array.
    perm: array-like of job indices.
    Returns the makespan (float).
    """
    n = len(perm)
    if n == 0:
        return 0.0
    n_machines = pt_t.shape[0]
    jobs = np.asarray(perm, dtype=np.intp)
    # (n_machines, n)
    P = pt_t[:, jobs]
    if n_machines == 1:
        return float(P[0].sum())
    # C[m, k] = max(C[m, k-1], C[m-1, k]) + P[m, k]
    C = P[0].copy()
    C = np.maximum.accumulate(C, axis=1)
    for m in range(1, n_machines):
        C = np.maximum(C, P[m]) + P[m]
        C = np.maximum.accumulate(C, axis=1)
    return float(C[-1])


def _neh_initial(pt):
    """NEH heuristic with full re-insertion."""
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    pt_t = pt.T
    perm = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            test = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_vec(pt_t, test)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm


def _palmer_initial(pt):
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


def _local_search_2opt(pt_t, perm, deadline, max_rounds=200000):
    """
    2-opt local search: reverse a contiguous segment.
    Stronger than insertion for flow-shop; O(1) per move via vectorized makespan.
    """
    n = len(perm)
    if n <= 2:
        return list(perm), _makespan_vec(pt_t, perm)

    current = list(perm)
    current_ms = _makespan_vec(pt_t, current)
    best = list(current)
    best_ms = current_ms

    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                # reverse segment [i, j]
                test = current[:i] + current[i:j+1][::-1] + current[j+1:]
                ms = _makespan_vec(pt_t, test)
                if ms < current_ms - 1e-9:
                    current = test
                    current_ms = ms
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best = list(current)
                    improved = True
                    break  # restart j loop with new current
            if improved:
                break
        if not improved:
            break
    return best, best_ms


def _local_search_insertion(pt_t, perm, deadline, max_rounds=200000):
    """
    Insertion local search: remove job at i, insert at j.
    """
    n = len(perm)
    if n <= 1:
        return list(perm), _makespan_vec(pt_t, perm)

    current = list(perm)
    current_ms = _makespan_vec(pt_t, current)
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
            rest = current[:i] + current[i+1:]
            for j in range(n):
                if j == i or j == i + 1:
                    continue
                if time.time() > deadline:
                    break
                test = rest[:j] + [job] + rest[j:]
                ms = _makespan_vec(pt_t, test)
                if ms < current_ms - 1e-9:
                    current = test
                    current_ms = ms
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best = list(current)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_ms


def _simulated_annealing(pt_t, perm, deadline, max_iter=200000, seed=0):
    """Simulated annealing with 2-opt moves."""
    rng = random.Random(seed)
    n = len(perm)
    if n <= 2:
        return list(perm), _makespan_vec(pt_t, perm)

    current = list(perm)
    current_ms = _makespan_vec(pt_t, current)
    best = list(current)
    best_ms = current_ms

    T = max(current_ms * 0.05, 1.0)
    T_min = max(current_ms * 0.0005, 0.01)
    cooling = 0.998

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        test = current[:i] + current[i:j+1][::-1] + current[j+1:]
        new_ms = _makespan_vec(pt_t, test)
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
    deadline = start + 145.0  # safety margin before 150s kill

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))

    pt_t = pt.T  # (n_machines, n_jobs)

    # Diverse initial solutions
    candidates = [
        _neh_initial(pt),
        _palmer_initial(pt),
        _lpt_initial(pt),
        _bottleneck_initial(pt),
        _mean_initial(pt),
        _first_machine_initial(pt),
        _last_machine_initial(pt),
    ]
    for s in range(4):
        candidates.append(_random_initial(n_jobs, 100 + s))

    best_ms = float('inf')
    best_perm = candidates[0]
    for cand in candidates:
        ms = _makespan_vec(pt_t, cand)
        if ms < best_ms:
            best_ms = ms
            best_perm = cand

    # 2-opt on best candidate (strongest move)
    perm, ms = _local_search_2opt(pt_t, best_perm, deadline - 25.0)
    if ms < best_ms:
        best_ms = ms
        best_perm = perm

    # 2-opt on other good candidates if time permits
    for cand in candidates[1:]:
        if time.time() > deadline - 20.0:
            break
        ms0 = _makespan_vec(pt_t, cand)
        if ms0 > best_ms * 1.15:
            continue
        perm, ms = _local_search_2opt(pt_t, cand, deadline - 20.0)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Insertion polish on best
    if time.time() < deadline - 10.0:
        perm, ms = _local_search_insertion(pt_t, best_perm, deadline - 10.0)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # SA to escape local optima
    if time.time() < deadline - 5.0:
        perm, ms = _simulated_annealing(pt_t, best_perm, deadline - 5.0, max_iter=200000, seed=7)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final 2-opt polish
    if time.time() < deadline - 2.0:
        perm, ms = _local_search_2opt(pt_t, best_perm, deadline - 1.0)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    return best_perm
# EVOLVE-BLOCK-END
