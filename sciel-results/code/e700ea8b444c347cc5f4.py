import numpy as np
import time

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 114.0  # safety margin before 150s kill
    
    n_jobs, n_machines = processing_times.shape
    
    # Fast makespan calculation using vectorized operations
    def calc_makespan(perm):
        if len(perm) == 0:
            return 0.0
        if n_machines == 1:
            return float(processing_times[list(perm), 0].sum())
        
        pt = processing_times[perm]
        n = len(perm)
        
        # C[i][j] = completion time of job i on machine j
        # Use iterative approach with running max
        completion = np.zeros(n_machines, dtype=float)
        
        for i in range(n):
            row = pt[i]
            # C[i][0] = C[i-1][0] + p[i][0]
            completion[0] += row[0]
            for j in range(1, n_machines):
                # C[i][j] = max(C[i-1][j], C[i][j-1]) + p[i][j]
                completion[j] = max(completion[j], completion[j-1]) + row[j]
        
        return float(completion[-1])
    
    # NEH Construction
    total_times = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_times)
    
    seq = [int(sorted_jobs[0])]
    
    for j in sorted_jobs[1:]:
        j = int(j)
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
    best_ms = calc_makespan(seq)
    
    # Local Search: 2-opt swap neighborhood
    # Use insertion moves for better performance
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(seq)
        
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(n):
                if i == j:
                    continue
                # Try moving job from position i to position j
                if j > i:
                    new_seq = seq[:i] + seq[i+1:j+1] + [seq[i]] + seq[j+1:]
                else:
                    new_seq = seq[:j] + [seq[i]] + seq[j:i] + seq[i+1:]
                
                ms = calc_makespan(new_seq)
                if ms < best_ms:
                    best_ms = ms
                    best_seq = new_seq
                    seq = new_seq
                    improved = True
                    break
            if improved:
                break
    
    # Final polish pass: tighter local search with swap moves
    if time.time() < deadline - 10:
        current_seq = list(best_seq)
        current_ms = best_ms
        polish_improved = True
        while polish_improved and time.time() < deadline - 5:
            polish_improved = False
            n = len(current_seq)
            for i in range(n - 1):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, n):
                    # Swap
                    current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
                    ms = calc_makespan(current_seq)
                    if ms < current_ms:
                        current_ms = ms
                        best_ms = ms
                        best_seq = list(current_seq)
                        polish_improved = True
                    else:
                        # Revert
                        current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
    
    return best_seq
