# EVOLVE-BLOCK-START
import numpy as np
import time

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Strategy:
    1. Fix the first job as the one with the highest total processing time.
    2. Generate initial solutions using:
       a. Nearest-Neighbor TSP heuristic on an N x N incremental makespan matrix.
       b. Greedy Insertion (CDS-style) using exact makespan evaluation.
    3. Apply efficient Local Search (Insertion and Interchange) to the best initial solution.
    """
    p = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = p.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Step 1: Fix the first job (highest total processing time)
    total_pt = p.sum(axis=1)
    first_job = int(np.argmax(total_pt))
    
    remaining_jobs = [j for j in range(n_jobs) if j != first_job]
    m = len(remaining_jobs)
    
    if m == 0:
        return [first_job]
    
    # Helper: Compute Makespan for a sequence
    def compute_makespan(seq):
        if not seq:
            return 0.0
        last_comp = np.zeros(n_machines)
        for j in seq:
            prev = 0.0
            for k in range(n_machines):
                prev = max(last_comp[k], prev) + p[j, k]
                last_comp[k] = prev
        return last_comp[-1]
    
    # Step 2: Build edge weight matrix W[i][j]
    # W[i][j] = makespan([i, j]) - makespan([i])
    # This approximates the cost of appending j after i.
    def makespan_2jobs(i, j):
        C_i = np.cumsum(p[i])
        C_j = np.zeros(n_machines)
        for k in range(n_machines):
            if k == 0:
                C_j[0] = C_i[0] + p[j, 0]
            else:
                C_j[k] = max(C_i[k], C_j[k-1]) + p[j, k]
        return C_j[-1] - np.sum(p[i])

    W = np.zeros((n_jobs, n_jobs))
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i != j:
                W[i][j] = makespan_2jobs(i, j)
            else:
                W[i][j] = float('inf')

    # Step 3: Initial Solutions
    
    # Solution A: Nearest Neighbor on W starting from first_job
    def nn_heuristic(start_job):
        visited = [start_job]
        unvisited = set(remaining_jobs)
        current = start_job
        while unvisited:
            best_j = min(unvisited, key=lambda j: W[current][j])
            visited.append(best_j)
            unvisited.remove(best_j)
            current = best_j
        return visited

    # Solution B: Greedy Insertion (CDS-style)
    def greedy_insertion(start_seq):
        seq = list(start_seq)
        remaining = [j for j in range(n_jobs) if j not in seq]
        # Sort remaining jobs by total PT descending
        rem_sorted = sorted(remaining, key=lambda x: total_pt[x], reverse=True)
        for j in rem_sorted:
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                new_seq = seq[:pos] + [j] + seq[pos:]
                ms = compute_makespan(new_seq)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [j] + seq[best_pos:]
        return seq

    # Generate multiple CDS-style initial solutions
    sorted_jobs = sorted(range(n_jobs), key=lambda x: total_pt[x], reverse=True)
    n_half = n_jobs // 2
    
    # Variant 1: First half
    seq_cds1 = greedy_insertion(sorted_jobs[:n_half])
    # Variant 2: Last half
    seq_cds2 = greedy_insertion(sorted_jobs[-n_half:])
    # Variant 3: Middle half
    start_idx = n_half // 2
    seq_cds3 = greedy_insertion(sorted_jobs[start_idx : start_idx + n_half])
    
    # Also keep NN and original Greedy (starting with highest PT job)
    seq_nn = nn_heuristic(first_job)
    seq_gi = greedy_insertion([first_job])
    
    candidates = [seq_cds1, seq_cds2, seq_cds3, seq_nn, seq_gi]
    
    best_seq = candidates[0]
    best_ms = compute_makespan(best_seq)
    
    for seq in candidates[1:]:
        ms = compute_makespan(seq)
        if ms < best_ms:
            best_seq = seq
            best_ms = ms
        
    # Step 4: Local Search
    # Use a time-limited local search with Insertion, Interchange, and Or-opt moves.
    # Uses First Improvement strategy for speed.
    start_time = time.time()
    time_limit = 105.0 # Leave 15s buffer for overhead
    
    current_seq = list(best_seq)
    current_ms = best_ms
    improved = True
    
    while improved and time.time() - start_time < time_limit:
        improved = False
        n = len(current_seq)
        
        # 1. Insertion Move (First Improvement)
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            job = current_seq[i]
            base_seq = current_seq[:i] + current_seq[i+1:]
            for pos in range(len(base_seq) + 1):
                if pos == i:
                    continue
                new_seq = base_seq[:pos] + [job] + base_seq[pos:]
                new_ms = compute_makespan(new_seq)
                if new_ms < current_ms - 1e-9:
                    current_seq = new_seq
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # 2. Interchange Move (First Improvement)
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                # Swap in place
                current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
                new_ms = compute_makespan(current_seq)
                if new_ms < current_ms - 1e-9:
                    current_ms = new_ms
                    improved = True
                    break
                else:
                    # Swap back
                    current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
            if improved:
                break
        
        if improved:
            continue

        # 3. Or-opt Move (First Improvement) - Move blocks of size 2 and 3
        for block_size in [2, 3]:
            if time.time() - start_time > time_limit:
                break
            for i in range(n - block_size + 1):
                if time.time() - start_time > time_limit:
                    break
                block = current_seq[i:i+block_size]
                base_seq = current_seq[:i] + current_seq[i+block_size:]
                for pos in range(len(base_seq) + 1):
                    if pos == i:
                        continue
                    new_seq = base_seq[:pos] + block + base_seq[pos:]
                    new_ms = compute_makespan(new_seq)
                    if new_ms < current_ms - 1e-9:
                        current_seq = new_seq
                        current_ms = new_ms
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return current_seq
# EVOLVE-BLOCK-END
