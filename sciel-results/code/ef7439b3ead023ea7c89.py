import numpy as np
import time
import random

def order_jobs(processing_times):
    start_time = time.time()
    # 120s limit, use 115s for safety
    deadline = start_time + 115.0
    
    n_jobs, n_machines = processing_times.shape
    
    # Pre-process processing times for faster access
    # Convert to list of lists for faster indexing in Python loops than numpy arrays
    pt = processing_times.tolist()
    
    # Helper to calculate makespan for a given permutation (list of job indices)
    def calc_makespan(perm):
        # c[m] is the completion time of the last job on machine m
        c = [0.0] * n_machines
        for j in perm:
            row = pt[j]
            for m in range(n_machines):
                if m == 0:
                    c[m] += row[m]
                else:
                    prev = c[m-1]
                    curr = c[m]
                    if prev > curr:
                        c[m] = prev + row[m]
                    else:
                        c[m] = curr + row[m]
        return c[-1]

    # 1. NEH Construction
    total_times = np.sum(processing_times, axis=1)
    sorted_jobs = np.argsort(-total_times)
    
    seq = [sorted_jobs[0]]
    
    for j in sorted_jobs[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting job j at each position
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = calc_makespan(new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        seq.insert(best_pos, j)
        
        if time.time() > deadline:
            return list(seq)

    best_seq = list(seq)
    best_ms = calc_makespan(best_seq)
    current_seq = list(best_seq)
    current_ms = best_ms

    # 2. Iterated Local Search (ILS)
    # We will use a combination of 2-opt (swap) and Or-opt (move block of 1-3)
    # If stuck, we perturb.
    
    no_improve_count = 0
    max_no_improve = 500  # Perturb after this many failed local search passes
    
    while time.time() < deadline:
        improved = False
        
        # Local Search Phase
        # Alternate between 2-opt and Or-opt to diversify neighborhood
        use_2opt = (no_improve_count % 2 == 0)
        
        n = len(current_seq)
        
        if use_2opt:
            # 2-Opt: Swap any two positions
            for i in range(n - 1):
                if time.time() > deadline:
                    return best_seq
                for j in range(i + 1, n):
                    # Swap
                    current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
                    ms = calc_makespan(current_seq)
                    if ms < current_ms:
                        current_ms = ms
                        if ms < best_ms:
                            best_ms = ms
                            best_seq = list(current_seq)
                        improved = True
                        no_improve_count = 0
                    else:
                        # Revert
                        current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
        else:
            # Or-Opt: Move a block of size 1, 2, or 3
            # We try moving blocks of size 1, 2, 3
            for block_size in [1, 2, 3]:
                if time.time() > deadline:
                    return best_seq
                for i in range(n - block_size):
                    # Extract block
                    block = current_seq[i:i+block_size]
                    # Remove block
                    remaining = current_seq[:i] + current_seq[i+block_size:]
                    # Try inserting block at every position in remaining
                    # Note: inserting at i is the same as original, so skip i
                    for pos in range(len(remaining) + 1):
                        if pos == i:
                            continue
                        # Insert
                        new_seq = remaining[:pos] + block + remaining[pos:]
                        ms = calc_makespan(new_seq)
                        if ms < current_ms:
                            current_seq = new_seq
                            current_ms = ms
                            if ms < best_ms:
                                best_ms = ms
                                best_seq = list(current_seq)
                            improved = True
                            no_improve_count = 0
                            # Restart scan for this block size or break?
                            # Breaking inner loop and restarting outer is safer for consistency
                            break
                    if improved:
                        # If we improved, we should restart the Or-opt scan for this block size
                        # because the sequence changed.
                        # But for speed, we might just continue. However, restarting ensures local optimality.
                        # Let's restart the i-loop for this block_size
                        break
                if improved:
                    # Restart block_size loop? No, just continue to next block size or restart all?
                    # Usually, if we improve, we restart the entire local search phase.
                    break
        
        if improved:
            # If we improved, continue local search on the new sequence
            # The loop above handles the immediate improvement.
            # We need to ensure we don't infinite loop on small improvements.
            # The `no_improve_count` is reset on improvement.
            pass
        else:
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Perturbation: Randomly swap 2 to 5 pairs of jobs
                # or perform a random Or-opt move
                num_perturbations = random.randint(2, 5)
                for _ in range(num_perturbations):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i != j:
                        current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
                
                current_ms = calc_makespan(current_seq)
                no_improve_count = 0
                # Don't update best_seq here, only if we find a better one later

    return best_seq
