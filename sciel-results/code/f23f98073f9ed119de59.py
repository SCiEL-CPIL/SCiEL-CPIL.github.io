import numpy as np
import time
import sys
import random

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Helper to calculate makespan for a given sequence
    def calc_makespan(seq):
        c = [0.0] * n_machines
        for j in seq:
            for k in range(n_machines):
                if k == 0:
                    c[k] += processing_times[j, k]
                else:
                    c[k] = max(c[k], c[k-1]) + processing_times[j, k]
        return c[n_machines - 1]

    # Fast evaluation for insertion move:
    # If we insert job j at position pos in seq, we can compute the new makespan
    # by updating the completion times from pos onwards.
    # However, for simplicity and robustness in SA, we might stick to full recalc
    # if n is small, or optimized recalc if n is large.
    # Given the 120s limit, and typical n, let's try to optimize.
    
    # Precompute job totals for NEH
    total_times = processing_times.sum(axis=1)
    
    # 1. NEH Construction
    order = list(np.argsort(-total_times))
    
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = calc_makespan(new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    
    current_best_seq = seq
    current_best_ms = calc_makespan(seq)
    
    # 2. Local Search (Insertion) - Fast version
    # We iterate until no improvement or time limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_move = None
        best_new_ms = current_best_ms
        
        # Check all insertion moves
        for i in range(n_jobs):
            job = seq[i]
            # Remove job
            seq_without = seq[:i] + seq[i+1:]
            
            # Try inserting at all positions
            for pos in range(n_jobs):
                if pos == i or pos == i + 1:
                    continue # Skip no-op moves
                
                new_seq = seq_without[:pos] + [job] + seq_without[pos:]
                ms = calc_makespan(new_seq)
                
                if ms < best_new_ms:
                    best_new_ms = ms
                    best_move = new_seq
        
        if best_move:
            current_best_seq = best_move
            current_best_ms = best_new_ms
            improved = True
            
    # 3. Simulated Annealing
    # Only run if time remains
    if time.time() < deadline - 5.0:
        current_seq = current_best_seq.copy()
        current_ms = current_best_ms
        best_seq = current_best_seq.copy()
        best_ms = current_best_ms
        
        T = 1.0
        cooling_rate = 0.99
        min_T = 1e-4
        
        # Estimate step size based on variance or just start with 1
        # We will use random swaps and insertions
        
        while time.time() < deadline - 1.0 and T > min_T:
            # Generate neighbor
            r = random.random()
            if r < 0.5:
                # Swap
                i, j = random.sample(range(n_jobs), 2)
                new_seq = current_seq.copy()
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            else:
                # Insertion
                i = random.randint(0, n_jobs - 1)
                j = random.randint(0, n_jobs - 1)
                if i == j:
                    continue
                job = current_seq[i]
                new_seq = current_seq[:i] + current_seq[i+1:]
                # Adjust j if i < j
                if i < j:
                    j -= 1
                new_seq = new_seq[:j] + [job] + new_seq[j:]
            
            new_ms = calc_makespan(new_seq)
            delta = new_ms - current_ms
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_seq = new_seq
                current_ms = new_ms
                
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_seq = current_seq.copy()
            
            T *= cooling_rate
            
        if best_ms < current_best_ms:
            current_best_seq = best_seq
            current_best_ms = best_ms
            
    return current_best_seq
