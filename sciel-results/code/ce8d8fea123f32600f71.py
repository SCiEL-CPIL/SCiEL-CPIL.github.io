"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(seq, pt):
    """Compute makespan of a permutation on a flow shop.
    seq: sequence of job indices (list or array)
    pt:  np.ndarray [n_jobs, n_machines]
    """
    if len(seq) == 0:
        return 0.0
    n_m = pt.shape[1]
    prev = np.zeros(n_m, dtype=np.float64)
    for job in seq:
        row = pt[job]
        curr = np.empty(n_m, dtype=np.float64)
        curr[0] = prev[0] + row[0]
        for k in range(1, n_m):
            a = prev[k]
            b = curr[k-1]
            curr[k] = (a if a > b else b) + row[k]
        prev = curr
    return float(prev[-1])

def _makespan_from_prev(seq, pt, start_idx, prev):
    """Compute makespan starting from a given prefix completion state.
    seq: full sequence
    pt: processing times
    start_idx: index in seq from which to start computation
    prev: completion times array before start_idx (length n_m)
    Returns makespan.
    """
    n = len(seq)
    n_m = pt.shape[1]
    if start_idx >= n:
        return float(prev[-1]) if n_m > 0 else 0.0
    for idx in range(start_idx, n):
        row = pt[seq[idx]]
        curr = np.empty(n_m, dtype=np.float64)
        curr[0] = prev[0] + row[0]
        for k in range(1, n_m):
            a = prev[k]
            b = curr[k-1]
            curr[k] = (a if a > b else b) + row[k]
        prev = curr
    return float(prev[-1])

def _compute_prefix_completions(seq, pt, up_to):
    """Compute completion times for prefix seq[0:up_to].
    Returns the completion vector after processing up_to jobs.
    """
    n_m = pt.shape[1]
    prev = np.zeros(n_m, dtype=np.float64)
    for idx in range(up_to):
        row = pt[seq[idx]]
        curr = np.empty(n_m, dtype=np.float64)
        curr[0] = prev[0] + row[0]
        for k in range(1, n_m):
            a = prev[k]
            b = curr[k-1]
            curr[k] = (a if a > b else b) + row[k]
        prev = curr
    return prev

def _eval_move(seq, pt, i, j):
    """Evaluate moving job at position i to position j.
    Returns the new makespan.
    """
    n = len(seq)
    if i == j:
        return _makespan(seq, pt)
    n_m = pt.shape[1]
    
    start = min(i, j)
    # Get completion state before start
    if start == 0:
        prev = np.zeros(n_m, dtype=np.float64)
    else:
        prev = _compute_prefix_completions(seq, pt, start)
    
    # Build new sequence from start onward
    moved_job = seq[i]
    new_seq = list(seq)
    new_seq.pop(i)
    target = j - 1 if j > i else j
    new_seq.insert(target, moved_job)
    
    return _makespan_from_prev(new_seq, pt, start, prev)

def _eval_swap(seq, pt, i, j):
    """Evaluate swapping jobs at positions i and j.
    Returns the new makespan.
    """
    n = len(seq)
    n_m = pt.shape[1]
    start = min(i, j)
    if start == 0:
        prev = np.zeros(n_m, dtype=np.float64)
    else:
        prev = _compute_prefix_completions(seq, pt, start)
    
    new_seq = list(seq)
    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
    
    return _makespan_from_prev(new_seq, pt, start, prev)

def _eval_reversal(seq, pt, i, j):
    """Evaluate reversing the segment from position i to j (inclusive).
    Returns the new makespan.
    """
    n = len(seq)
    if i == j:
        return _makespan(seq, pt)
    n_m = pt.shape[1]
    start = min(i, j)
    if start == 0:
        prev = np.zeros(n_m, dtype=np.float64)
    else:
        prev = _compute_prefix_completions(seq, pt, start)
    
    new_seq = list(seq)
    new_seq[start:start+abs(j-i)+1] = new_seq[start:start+abs(j-i)+1][::-1]
    
    return _makespan_from_prev(new_seq, pt, start, prev)

def _neh_heuristic(pt):
    """NEH heuristic: sort by decreasing total processing time, insert each job
    at the best position."""
    n_jobs, n_machines = pt.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    total = pt.sum(axis=1)
    order = list(np.argsort(-total))
    
    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq.insert(best_pos, job)
    return seq

def _local_search(seq, pt, deadline, max_passes=100):
    """Local search with move and swap neighborhoods, first-improvement strategy.
    Returns the best sequence and its makespan found within the time budget."""
    n = len(seq)
    best_seq = list(seq)
    best_ms = _makespan(best_seq, pt)
    
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        
        # Try move neighborhood
        for i in range(n):
            if time.time() > deadline:
                break
            job = best_seq[i]
            for j in range(n):
                if i == j:
                    continue
                new_ms = _eval_move(best_seq, pt, i, j)
                if new_ms < best_ms - 1e-9:
                    best_seq.pop(i)
                    target = j - 1 if j > i else j
                    best_seq.insert(target, job)
                    best_ms = new_ms
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try swap neighborhood
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline:
                        break
                    new_ms = _eval_swap(best_seq, pt, i, j)
                    if new_ms < best_ms - 1e-9:
                        best_seq[i], best_seq[j] = best_seq[j], best_seq[i]
                        best_ms = new_ms
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_seq, best_ms

def _perturb(seq, pt, strength):
    """Perturb a sequence by performing a random segment reversal.
    strength: fraction of sequence length to use for perturbation window."""
    n = len(seq)
    if n <= 1:
        return list(seq)
    
    # Choose a random segment length based on strength
    min_len = max(2, int(n * strength * 0.3))
    max_len = max(min_len + 1, int(n * strength))
    seg_len = random.randint(min_len, min(n - 1, max_len))
    start = random.randint(0, n - seg_len)
    end = start + seg_len - 1
    
    new_seq = list(seq)
    new_seq[start:end+1] = new_seq[start:end+1][::-1]
    return new_seq

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. NEH construction
    seq = _neh_heuristic(processing_times)
    best_ms = _makespan(seq, processing_times)
    best_seq = list(seq)
    
    # 2. Initial local search
    seq, best_ms = _local_search(seq, processing_times, deadline)
    best_seq = list(seq)
    
    # 3. Iterated local search with perturbation
    random.seed(42)
    current_seq = list(seq)
    current_ms = best_ms
    perturbation_strength = 0.15
    
    while time.time() < deadline - 5.0:
        # Perturb current solution
        perturbed = _perturb(current_seq, processing_times, perturbation_strength)
        
        # Local search on perturbed solution
        improved_seq, improved_ms = _local_search(perturbed, processing_times, deadline)
        
        if improved_ms < current_ms - 1e-9:
            current_seq = improved_seq
            current_ms = improved_ms
            if improved_ms < best_ms - 1e-9:
                best_ms = improved_ms
                best_seq = list(improved_seq)
            # Slightly reduce perturbation on improvement
            perturbation_strength = max(0.05, perturbation_strength * 0.95)
        else:
            # Keep current, slightly increase perturbation
            perturbation_strength = min(0.30, perturbation_strength * 1.05)
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # 4. Final polish: one more local search on the best solution
    if time.time() < deadline - 2.0:
        final_seq, final_ms = _local_search(best_seq, processing_times, deadline)
        if final_ms < best_ms - 1e-9:
            best_seq = final_seq
            best_ms = final_ms
    
    return best_seq

# EVOLVE-BLOCK-END
