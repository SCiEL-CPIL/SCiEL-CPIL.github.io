"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, processing_times):
    """Compute makespan for a given job order on the flow shop."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    # completion_times[m] = completion time of the last job on machine m
    completion_times = np.zeros(n_machines)
    for job_idx in order:
        for m in range(n_machines):
            if m == 0:
                completion_times[m] += processing_times[job_idx, m]
            else:
                completion_times[m] = max(completion_times[m], completion_times[m - 1]) + processing_times[job_idx, m]
    return completion_times[-1]


def _makespan_from_array(order, processing_times):
    """Vectorized makespan computation using cumulative sums."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    # Extract processing times in the given order
    pt_ordered = processing_times[order]  # shape (n_jobs, n_machines)
    # Compute completion times using the flow shop recurrence
    # C[j, m] = C[j-1, m] + p[j, m] if m == 0
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m] if m > 0
    # We can compute this iteratively over jobs
    completion_times = np.zeros(n_machines)
    for j in range(n_jobs):
        for m in range(n_machines):
            if m == 0:
                completion_times[m] += pt_ordered[j, m]
            else:
                completion_times[m] = max(completion_times[m], completion_times[m - 1]) + pt_ordered[j, m]
    return completion_times[-1]


def _neh_initialization(processing_times):
    """
    NEH heuristic initialization.
    Sort jobs by total processing time descending, then insert each job
    into the best position in the current sequence.
    """
    n_jobs = processing_times.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    sequence = [sorted_jobs[0]]
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(sequence) + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        sequence = sequence[:best_pos] + [job] + sequence[best_pos:]
    
    return sequence


def _adjacent_swap_improve(order, processing_times, max_iterations=100):
    """
    Local search using adjacent swaps with first improvement.
    Returns improved order and makespan.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return list(order), _makespan(order, processing_times)
    
    current_order = list(order)
    current_makespan = _makespan(current_order, processing_times)
    
    for _ in range(max_iterations):
        improved = False
        for i in range(n_jobs - 1):
            # Try swapping positions i and i+1
            new_order = current_order[:]
            new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
            new_makespan = _makespan(new_order, processing_times)
            if new_makespan < current_makespan:
                current_order = new_order
                current_makespan = new_makespan
                improved = True
                # Restart the scan after improvement
                break
        if not improved:
            break
    
    return current_order, current_makespan


def _insertion_improve(order, processing_times, max_iterations=50):
    """
    Local search using insertions (remove job from position i, insert at position j).
    Uses first improvement strategy.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return list(order), _makespan(order, processing_times)
    
    current_order = list(order)
    current_makespan = _makespan(current_order, processing_times)
    
    for _ in range(max_iterations):
        improved = False
        for i in range(n_jobs):
            job = current_order[i]
            # Remove job from position i
            base_seq = current_order[:i] + current_order[i+1:]
            for j in range(len(base_seq) + 1):
                if j == i or j == i + 1:
                    continue  # Skip no-op and adjacent swap (handled separately)
                new_order = base_seq[:j] + [job] + base_seq[j:]
                new_makespan = _makespan(new_order, processing_times)
                if new_makespan < current_makespan:
                    current_order = new_order
                    current_makespan = new_makespan
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return current_order, current_makespan


def _vns_search(order, processing_times, deadline):
    """
    Variable Neighborhood Search: systematically explore neighborhoods
    of increasing radius using insertion moves with |i-j| <= k.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return list(order), _makespan(order, processing_times)
    
    current_order = list(order)
    current_makespan = _makespan(current_order, processing_times)
    
    max_radius = n_jobs - 1
    k = 1
    
    while k <= max_radius and time.time() < deadline:
        improved = False
        for i in range(n_jobs):
            if time.time() >= deadline:
                return current_order, current_makespan
            job = current_order[i]
            base_seq = current_order[:i] + current_order[i+1:]
            # Consider insertions within radius k
            for offset in range(1, k + 1):
                for direction in [-1, 1]:
                    j = i + direction * offset
                    # Map j to base_seq coordinate
                    # In base_seq, positions after i are shifted by -1
                    if direction == 1:
                        j_base = j - 1
                    else:
                        j_base = j
                    if j_base < 0 or j_base > len(base_seq):
                        continue
                    if j_base == i or j_base == i - 1:
                        continue  # No-op
                    new_order = base_seq[:j_base] + [job] + base_seq[j_base:]
                    new_makespan = _makespan(new_order, processing_times)
                    if new_makespan < current_makespan:
                        current_order = new_order
                        current_makespan = new_makespan
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            k += 1
        else:
            # Reset k to 1 after an improvement (standard VNS)
            k = 1
    
    return current_order, current_makespan


def order_jobs(processing_times):
    """
    Main entry point: construct a job permutation minimizing makespan.
    Uses NEH initialization followed by VNS with deadline guard.
    """
    processing_times = np.asarray(processing_times)
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: 110s out of 120s limit
    
    # Step 1: NEH initialization
    initial_order = _neh_initialization(processing_times)
    current_order, current_makespan = _adjacent_swap_improve(initial_order, processing_times, max_iterations=20)
    
    # Step 2: VNS search
    if time.time() < deadline:
        current_order, current_makespan = _vns_search(current_order, processing_times, deadline)
    
    # Step 3: Final polish with adjacent swaps if time permits
    if time.time() < deadline - 5.0:
        current_order, current_makespan = _adjacent_swap_improve(current_order, processing_times, max_iterations=50)
    
    return current_order
# EVOLVE-BLOCK-END
