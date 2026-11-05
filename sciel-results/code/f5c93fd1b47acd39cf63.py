# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using TSP-inspired decomposition
    and Or-opt local search with vectorized suffix updates.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Fast fallback for very large instances
    if n_jobs * n_machines > 500000:
        return list(np.argsort(-processing_times.sum(axis=1)))
    
    def makespan(seq):
        """Compute makespan for a given sequence of job indices."""
        if len(seq) == 0:
            return 0.0
        completion = np.zeros(n_machines)
        for job in seq:
            prev = completion[0]
            curr = prev + processing_times[job, 0]
            completion[0] = curr
            for m in range(1, n_machines):
                prev = curr
                curr = max(prev, completion[m]) + processing_times[job, m]
                completion[m] = curr
        return completion[-1]
    
    def compute_completion(seq):
        """Compute full completion matrix for a sequence."""
        n = len(seq)
        completion = np.zeros((n, n_machines))
        for i, job in enumerate(seq):
            if i == 0:
                completion[i, 0] = processing_times[job, 0]
            else:
                completion[i, 0] = completion[i-1, 0] + processing_times[job, 0]
            for m in range(1, n_machines):
                completion[i, m] = max(completion[i, m-1], completion[i-1, m]) + processing_times[job, m]
        return completion
    
    def makespan_suffix(seq, start_idx, prefix):
        """
        Compute makespan for seq[start_idx:] given completion times of seq[:start_idx].
        prefix: array of shape (m,) containing completion times of the job before start_idx.
        """
        if start_idx >= len(seq):
            return prefix[-1]
        
        completion = prefix.copy()
        for job in seq[start_idx:]:
            prev = completion[0]
            curr = prev + processing_times[job, 0]
            completion[0] = curr
            for m in range(1, n_machines):
                prev = curr
                curr = max(prev, completion[m]) + processing_times[job, m]
                completion[m] = curr
        return completion[-1]
    
    def neh_heuristic():
        """NEH (Nawaz, Enscore, Ham) heuristic."""
        totals = processing_times.sum(axis=1)
        order = list(np.argsort(-totals))
        
        seq = [order[0]]
        for job in order[1:]:
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                trial = seq[:pos] + [job] + seq[pos:]
                ms = makespan(trial)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        
        return seq
    
    def palmer_heuristic():
        """Palmer's dispatching rule."""
        n = n_jobs
        r = np.arange(n)
        totals = processing_times.sum(axis=1)
        s = (n - 1 - r) * totals + np.cumsum(processing_times.sum(axis=0))
        return list(np.argsort(s))
    
    def random_heuristic():
        """Random permutation."""
        seq = list(range(n_jobs))
        random.shuffle(seq)
        return seq
    
    # Initialize with multiple heuristics
    candidates = []
    candidates.append(neh_heuristic())
    candidates.append(palmer_heuristic())
    candidates.append(random_heuristic())
    
    best_seq = None
    best_ms = float('inf')
    
    for seq in candidates:
        ms = makespan(seq)
        if ms < best_ms:
            best_ms = ms
            best_seq = list(seq)
    
    # TSP-inspired decomposition: Fix the job with highest total processing time at position 0
    # This reduces the search space significantly and aligns with flow-shop properties
    totals = processing_times.sum(axis=1)
    fixed_job = np.argmax(totals)
    
    # Ensure the fixed job is at position 0
    if best_seq[0] != fixed_job:
        idx = best_seq.index(fixed_job)
        best_seq = [fixed_job] + best_seq[:idx] + best_seq[idx+1:]
        best_ms = makespan(best_seq)
    
    # Local Search with Or-opt (segment relocation)
    current_seq = list(best_seq)
    current_ms = best_ms
    current_completion = compute_completion(current_seq)
    n = len(current_seq)
    
    # Or-opt moves: Remove a segment of length L (1, 2, or 3) and reinsert it elsewhere
    # This is more powerful than adjacent swaps for flow-shop problems
    
    max_iterations = 10000
    iteration = 0
    no_improvement = 0
    max_no_improvement = 500
    
    while iteration < max_iterations and time.time() < deadline:
        # Choose a random segment length (1, 2, or 3)
        L = random.choice([1, 2, 3])
        
        # Choose a random start position for the segment
        if n <= L:
            break
        i = random.randint(0, n - L)
        
        # Choose a random insertion position (not adjacent to original position)
        j = random.randint(0, n - L)
        if abs(j - i) <= 1:
            continue
        
        # Create new sequence by removing segment [i:i+L] and inserting at j
        # Adjust j if j > i
        if j > i:
            j -= L
        
        new_seq = current_seq[:i] + current_seq[i+L:]
        new_seq = new_seq[:j] + current_seq[i:i+L] + new_seq[j:]
        
        # Compute new makespan using suffix update
        # The change starts at min(i, j)
        start_idx = min(i, j)
        
        if start_idx == 0:
            prefix = np.zeros(n_machines)
        else:
            prefix = current_completion[start_idx - 1]
        
        new_ms = makespan_suffix(new_seq, start_idx, prefix)
        
        if new_ms < current_ms:
            # Accept the move
            current_seq = new_seq
            current_ms = new_ms
            current_completion = compute_completion(current_seq)
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_seq = list(current_seq)
                no_improvement = 0
            else:
                no_improvement += 1
        else:
            no_improvement += 1
        
        iteration += 1
        
        if no_improvement > max_no_improvement:
            # Restart with a perturbation
            no_improvement = 0
            # Perturb the best sequence
            current_seq = list(best_seq)
            # Randomly swap a few elements
            for _ in range(3):
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                current_seq[a], current_seq[b] = current_seq[b], current_seq[a]
            current_ms = makespan(current_seq)
            current_completion = compute_completion(current_seq)
    
    return list(best_seq)
# EVOLVE-BLOCK-END
