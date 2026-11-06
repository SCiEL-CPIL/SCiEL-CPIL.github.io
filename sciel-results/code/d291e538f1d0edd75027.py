import numpy as np
import time
import random

def order_jobs(processing_times):
    start_time = time.time()
    # 120s limit, use 110s for safety margin
    deadline = start_time + 110.0
    
    n_jobs, n_machines = processing_times.shape
    
    # Pre-process processing times for faster access
    pt = processing_times.tolist()
    nm = n_machines
    p = pt
    
    # Helper to calculate makespan for a given permutation (list of job indices)
    def calc_makespan(perm):
        c0 = 0.0
        c = [0.0] * nm
        for j in perm:
            row = p[j]
            # Unroll first machine
            c0 += row[0]
            c[0] = c0
            for m in range(1, nm):
                prev = c[m-1]
                curr = c[m]
                if prev > curr:
                    c[m] = prev + row[m]
                else:
                    c[m] = curr + row[m]
        return c[-1] if nm > 0 else 0.0

    # 1. NEH Construction
    total_times = np.sum(processing_times, axis=1)
    sorted_jobs = np.argsort(-total_times)
    
    seq = [sorted_jobs[0]]
    
    for j in sorted_jobs[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
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
    def local_search(perm, current_ms):
        n = len(perm)
        improved = True
        max_ls_iters = 20 # Reduced to allow more ILS cycles
        ls_iter = 0
        
        while improved and ls_iter < max_ls_iters:
            improved = False
            ls_iter += 1
            if time.time() > deadline:
                break
            
            # 2-Opt
            for i in range(n - 1):
                if time.time() > deadline:
                    return perm, calc_makespan(perm)
                for j in range(i + 1, n):
                    perm[i], perm[j] = perm[j], perm[i]
                    ms = calc_makespan(perm)
                    if ms < current_ms:
                        current_ms = ms
                        improved = True
                    else:
                        perm[i], perm[j] = perm[j], perm[i]
                        
            if not improved:
                # Or-Opt
                for block_size in [1, 2, 3]:
                    if time.time() > deadline:
                        return perm, calc_makespan(perm)
                    for i in range(n - block_size):
                        block = perm[i:i+block_size]
                        remaining = perm[:i] + perm[i+block_size:]
                        for pos in range(len(remaining) + 1):
                            if pos == i:
                                continue
                            new_perm = remaining[:pos] + block + remaining[pos:]
                            ms = calc_makespan(new_perm)
                            if ms < current_ms:
                                perm = new_perm
                                current_ms = ms
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                        
        return perm, current_ms

    while time.time() < deadline:
        current_seq, current_ms = local_search(current_seq, current_ms)
        
        if current_ms < best_ms:
            best_ms = current_ms
            best_seq = list(current_seq)
        
        # Perturbation
        num_perturbations = random.randint(2, 5)
        for _ in range(num_perturbations):
            i = random.randint(0, n_jobs - 1)
            j = random.randint(0, n_jobs - 1)
            if i != j:
                current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
        
        current_ms = calc_makespan(current_seq)
        
        if time.time() > deadline:
            break

    return best_seq
