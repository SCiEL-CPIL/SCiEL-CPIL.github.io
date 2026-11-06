"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses NEH heuristic for initial construction, then local search with
    insertion moves, plus a deadline guard to ensure termination.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n_jobs, n_machines = processing_times.shape

    if n_jobs <= 1:
        return list(range(n_jobs))

    # Precompute makespan function for speed
    def makespan(seq):
        """Compute makespan for a given job sequence."""
        completion = np.zeros(n_machines)
        for job in seq:
            for m in range(n_machines):
                completion[m] = completion[m] + processing_times[job, m]
                if m > 0:
                    completion[m] = max(completion[m], completion[m-1])
        return completion[-1]

    # --- Stage 1: NEH Heuristic ---
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))

    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [j] + seq[pos:]
            ms = makespan(candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]

    best_seq = list(seq)
    best_ms = makespan(best_seq)

    # --- Stage 2: Local Search with Insertion Moves ---
    improved = True
    while improved:
        improved = False
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            job = best_seq[i]
            for j in range(n_jobs):
                if i == j or time.time() > deadline:
                    continue
                # Try moving job from position i to position j
                new_seq = best_seq[:i] + best_seq[i+1:j+1] + [job] + best_seq[j+1:]
                if i > j:
                    new_seq = best_seq[:j] + [job] + best_seq[j:i] + best_seq[i+1:]
                ms = makespan(new_seq)
                if ms < best_ms:
                    best_ms = ms
                    best_seq = new_seq
                    improved = True
                    break  # Restart scan from beginning after improvement
            if time.time() > deadline:
                break

    return list(best_seq)
# EVOLVE-BLOCK-END
