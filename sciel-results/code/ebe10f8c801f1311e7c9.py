# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(pt, seq):
    """Makespan for a sequence of job indices. pt: (n_jobs, n_machines) array."""
    n = len(seq)
    if n == 0:
        return 0.0
    M = pt.shape[1]
    # C[j, m] = completion time of job j on machine m
    C = np.empty((n, M), dtype=np.float64)
    row = C[0]
    row[0] = pt[seq[0], 0]
    if M > 1:
        prev = row[0]
        for m in range(1, M):
            prev = max(prev, 0.0) + pt[seq[0], m]
            row[m] = prev
    for i in range(1, n):
        j = seq[i]
        prev = C[i - 1, M - 1]
        row = C[i]
        row[0] = prev + pt[j, 0]
        if M > 1:
            for m in range(1, M):
                prev = max(prev, C[i - 1, m]) + pt[j, m]
                row[m] = prev
    return C[n - 1, M - 1]

def _cds(pt):
    """CDS (Campbell, Dudek, Smith) heuristic: build by inserting in descending total time."""
    n = pt.shape[0]
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(pt, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    return seq

def _neh(pt):
    """NEH heuristic (identical to CDS here)."""
    return _cds(pt)

def _palmer(pt):
    n = pt.shape[0]
    p_first = pt[:, 0]
    p_last = pt[:, -1]
    keys = n * p_first - p_last
    return list(np.argsort(-keys))

def _lpt(pt):
    return list(np.argsort(-pt.sum(axis=1)))

def _spt(pt):
    return list(np.argsort(pt.sum(axis=1)))

def _random_perm(n):
    seq = list(range(n))
    random.shuffle(seq)
    return seq

def _local_search(pt, seq, time_budget):
    """
    First-improvement insertion + interchange local search with a wall-clock guard.
    Returns (seq, best_ms).
    """
    n = len(seq)
    M = pt.shape[1]
    seq = list(seq)
    best_ms = _makespan(pt, seq)
    start = time.time()
    deadline = start + time_budget

    # Precompute per-job processing vectors for speed
    # We recompute makespan fully each time (simple, robust).

    improved = True
    while improved and (time.time() < deadline):
        improved = False
        # --- Interchange (swap adjacent and nearby) ---
        for i in range(n):
            if time.time() >= deadline:
                return seq, best_ms
            for j in range(i + 1, min(n, i + 4)):
                seq[j - 1] = seq[j - 1]
                # swap i and j
                seq[i], seq[j] = seq[j], seq[i]
                ms = _makespan(pt, seq)
                if ms < best_ms - 1e-12:
                    best_ms = ms
                    improved = True
                else:
                    seq[i], seq[j] = seq[j], seq[i]
        # --- Insertion (move each job to a better position) ---
        for i in range(n):
            if time.time() >= deadline:
                return seq, best_ms
            job = seq[i]
            # remove job
            rest = seq[:i] + seq[i + 1:]
            best_pos = i
            best_ms2 = best_ms
            for pos in range(len(rest) + 1):
                trial = rest[:pos] + [job] + rest[pos:]
                ms = _makespan(pt, trial)
                if ms < best_ms2 - 1e-12:
                    best_ms2 = ms
                    best_pos = pos
            if best_pos != i:
                new_seq = rest[:best_pos] + [job] + rest[best_pos:]
                seq = new_seq
                best_ms = best_ms2
                improved = True
    return seq, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape

    if n_jobs <= 1:
        return list(range(n_jobs))

    if n_jobs * n_machines > 500000:
        return list(np.argsort(-pt.sum(axis=1)))

    # --- Initial candidates ---
    candidates = []
    candidates.append(_cds(pt))
    candidates.append(_palmer(pt))
    candidates.append(_lpt(pt))
    candidates.append(_spt(pt))

    # A few random restarts
    n_random = min(4, max(2, n_jobs // 8))
    for _ in range(n_random):
        candidates.append(_random_perm(n_jobs))

    best_seq = None
    best_ms = float('inf')

    # Local search each candidate with a decaying time budget
    for i, seq in enumerate(candidates):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        # spend a fraction of remaining on this candidate
        budget = min(remaining * 0.35, 25.0)
        if budget < 0.05:
            break
        seq_ls, ms = _local_search(pt, seq, budget)
        if ms < best_ms:
            best_ms = ms
            best_seq = seq_ls

    # If we have leftover time, do more random restarts with local search
    if best_seq is not None and time.time() < deadline:
        attempts = 0
        while time.time() < deadline and attempts < 40:
            attempts += 1
            seq = _random_perm(n_jobs)
            remaining = deadline - time.time()
            budget = min(remaining * 0.25, 15.0)
            if budget < 0.05:
                break
            seq_ls, ms = _local_search(pt, seq, budget)
            if ms < best_ms:
                best_ms = ms
                best_seq = seq_ls

    if best_seq is None:
        best_seq = _lpt(pt)

    return list(best_seq)
# EVOLVE-BLOCK-END
