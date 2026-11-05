# EVOLVE-BLOCK-START
import numpy as np
import time


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Strategy:
    1. Fix the first job as the one with the highest total processing time.
    2. Build an N x N edge weight matrix approximating incremental makespan
       contribution of placing job j after job i.
    3. Use nearest-neighbor TSP heuristic on this matrix for initial construction.
    4. Apply 2-opt and insertion local search with the edge-weight matrix
       for fast evaluation.
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
    
    # Step 2: Build edge weight matrix for incremental makespan contribution
    # W[i][j] approximates the additional makespan when job j is placed after job i
    # in the sequence (considering the tail of the flow-shop)
    
    # For a flow shop, if we have a sequence, the makespan can be computed incrementally.
    # The key insight: when we place job j after job i, the "cost" depends on
    # the processing times of both jobs across all machines.
    
    # Approximation: W[i][j] = sum over machines of max(p[i][k], p[j][k]) 
    # This captures the interaction between consecutive jobs.
    # A better approximation uses the "tail" of the flow-shop.
    
    # Let's use a more accurate incremental cost:
    # When job j follows job i, the incremental contribution to makespan
    # depends on the cumulative processing times. For a 2-machine approximation:
    # The makespan increase from adding job j after job i is approximately:
    # max(p[i][0] + p[j][0], p[i][1]) + p[j][1] - (previous makespan contribution)
    
    # For general m machines, we'll use:
    # W[i][j] = max over k of (cumulative effect)
    # Simplified: W[i][j] = sum_k max(p[i][k], p[j][k]) + 0.5 * sum_k |p[i][k] - p[j][k]|
    
    # Actually, let's use a proper incremental makespan estimator:
    # For a sequence ...i, j..., the makespan contribution of the pair (i,j)
    # in a flow shop can be approximated by considering the "bottleneck" machines.
    
    # Better approach: precompute for each pair (i,j) the makespan of the 2-job
    # sub-problem [i, j] on all machines, minus the makespan of [i] alone.
    # This gives the true incremental cost of placing j after i in a 2-job context.
    
    def makespan_2jobs(i, j):
        """Makespan of sequence [i, j] on all machines."""
        last_comp = np.zeros(n_machines)
        for job in [i, j]:
            prev = 0.0
            for k in range(n_machines):
                prev = max(last_comp[k], prev) + p[job][k]
                last_comp[k] = prev
        return last_comp[-1]
    
    def makespan_1job(i):
        """Makespan of sequence [i] on all machines."""
        return p[i][-1]  # Last machine completion time for single job
    
    # Build W matrix: W[i][j] = makespan([i,j]) - makespan([i])
    # This is the true incremental cost of placing j after i
    W = np.zeros((n_jobs, n_jobs))
    for i in range(n_jobs):
        ms_i = makespan_1job(i)
        for j in range(n_jobs):
            if i != j:
                W[i][j] = makespan_2jobs(i, j) - ms_i
            else:
                W[i][j] = 0.0
    
    # Step 3: Nearest-neighbor TSP heuristic starting from first_job
    # At each step, pick the unvisited job j that minimizes W[current][j]
    
    visited = [first_job]
    unvisited = set(remaining_jobs)
    current = first_job
    
    while unvisited:
        # Find the job j in unvisited that minimizes W[current][j]
        best_j = min(unvisited, key=lambda j: W[current][j])
        visited.append(best_j)
        unvisited.remove(best_j)
        current = best_j
    
    initial_seq = visited
    
    # Step 4: Local search with best-improvement 2-opt and insertion
    
    def compute_makespan(seq):
        """Compute exact makespan for a job sequence."""
        if not seq:
            return 0.0
        last_comp = np.zeros(n_machines)
        for j in seq:
            prev = 0.0
            for k in range(n_machines):
                prev = max(last_comp[k], prev) + p[j][k]
                last_comp[k] = prev
        return last_comp[-1]
    
    current_makespan = compute_makespan(initial_seq)
    best_seq = initial_seq[:]
    best_makespan = current_makespan
    
    time_limit = 110.0
    start_time = time.time()
    
    # Best-improvement 2-opt: reverse a segment (excluding position 0)
    def best_2opt(seq, current_ms):
        n = len(seq)
        best_new_seq = None
        best_new_ms = current_ms
        for i in range(1, n - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    break
                # Reverse seq[i:j+1]
                new_seq = seq[:i] + seq[i:j+1][::-1] + seq[j+1:]
                new_ms = compute_makespan(new_seq)
                if new_ms < best_new_ms - 1e-9:
                    best_new_ms = new_ms
                    best_new_seq = new_seq
        if best_new_seq is not None:
            return best_new_seq, best_new_ms, True
        return seq, current_ms, False
    
    # Best-improvement insertion: remove a job and reinsert it at the best position
    def best_insertion(seq, current_ms):
        n = len(seq)
        best_new_seq = None
        best_new_ms = current_ms
        for i in range(1, n):  # don't move the first job
            if time.time() - start_time > time_limit:
                break
            job = seq[i]
            # Remove job from position i
            base_seq = seq[:i] + seq[i+1:]
            # Try inserting at each position from 1 to len(base_seq)
            for pos in range(1, len(base_seq) + 1):
                if pos == i:  # same position
                    continue
                new_seq = base_seq[:pos] + [job] + base_seq[pos:]
                new_ms = compute_makespan(new_seq)
                if new_ms < best_new_ms - 1e-9:
                    best_new_ms = new_ms
                    best_new_seq = new_seq
        if best_new_seq is not None:
            return best_new_seq, best_new_ms, True
        return seq, current_ms, False
    
    # Alternate between 2-opt and insertion until no improvement or time runs out
    seq = best_seq[:]
    current_ms = best_makespan
    no_improve_count = 0
    max_no_improve = 6
    
    while no_improve_count < max_no_improve and time.time() - start_time < time_limit:
        # Try 2-opt
        seq, current_ms, imp = best_2opt(seq, current_ms)
        if imp:
            no_improve_count = 0
            if current_ms < best_makespan - 1e-9:
                best_makespan = current_ms
                best_seq = seq[:]
        else:
            no_improve_count += 1
        
        # Try insertion
        if time.time() - start_time >= time_limit:
            break
        seq, current_ms, imp = best_insertion(seq, current_ms)
        if imp:
            no_improve_count = 0
            if current_ms < best_makespan - 1e-9:
                best_makespan = current_ms
                best_seq = seq[:]
        else:
            no_improve_count += 1
    
    # Multi-start: try different initial jobs to escape local optima
    if time.time() - start_time < time_limit - 30:
        # Try a few alternative starting jobs
        top_k = min(3, n_jobs)
        top_jobs = np.argsort(total_pt)[-top_k:]
        tried = set()
        for start_job in top_jobs:
            if start_job in tried:
                continue
            tried.add(start_job)
            if time.time() - start_time > time_limit - 5:
                break
            
            rem = [j for j in range(n_jobs) if j != start_job]
            if not rem:
                continue
            
            # Nearest-neighbor from this start
            visited = [start_job]
            unvisited = set(rem)
            current = start_job
            while unvisited:
                best_j = min(unvisited, key=lambda j: W[current][j])
                visited.append(best_j)
                unvisited.remove(best_j)
                current = best_j
            
            # Quick local search on this start
            ms = compute_makespan(visited)
            if ms < best_makespan - 1e-9:
                # Do a few rounds of local search
                for _ in range(3):
                    if time.time() - start_time > time_limit - 5:
                        break
                    visited, ms, imp = best_2opt(visited, ms)
                    if not imp:
                        break
                for _ in range(3):
                    if time.time() - start_time > time_limit - 5:
                        break
                    visited, ms, imp = best_insertion(visited, ms)
                    if not imp:
                        break
                
                if ms < best_makespan - 1e-9:
                    best_makespan = ms
                    best_seq = visited[:]
    
    return best_seq
# EVOLVE-BLOCK-END
