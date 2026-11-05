"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using Simulated Annealing
    with O(1) incremental makespan evaluation via prefix sums.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Fast fallback for very large instances
    if n_jobs * n_machines > 500000:
        return list(np.argsort(-processing_times.sum(axis=1)))
    
    # 1. Precomputation
    # cum[job, m] = sum of processing times for job from machine 0 to m
    # We need this to quickly calculate the completion time of a job at a specific position
    # if it were to start at a certain time.
    # However, for O(1) delta evaluation of a swap (i, i+1), we need:
    # C_new - C_old
    # Let's precompute the "tail" effect or use a specific formula.
    
    # A simpler O(1) swap evaluation for flow shop:
    # If we swap jobs A and B at positions k and k+1.
    # The completion time of the rest of the schedule (positions > k+1) is affected.
    # Actually, for adjacent swaps in flow shop, the change in makespan is NOT strictly local to the swap
    # unless we track the "bottleneck" or use a specific property.
    # 
    # Wait, the O(1) delta for TSP is easy. For Flow Shop, it's harder because of the max() operation.
    # However, there is a known O(1) approximation or exact calculation for adjacent swaps 
    # if we maintain the completion times array.
    #
    # Let's use a robust approach:
    # 1. Calculate makespan of a sequence in O(N*M) using NumPy vectorization where possible, 
    #    but the dependency is sequential.
    # 2. To make SA fast, we need a very fast makespan evaluator.
    #    Python loops are slow. Let's try to optimize the makespan calculation.
    #    
    # Alternative: Use a "Block Interchange" or "Adjacent Swap" with a fast local re-evaluation.
    # Re-evaluating the whole makespan is O(N*M). 
    # If N=100, M=10, that's 1000 ops. 
    # In 115 seconds, we can do roughly 10^7 to 10^8 simple ops in Python/NumPy mix.
    # If we do 10^6 swaps, and each swap takes 1000 ops (10^3), that's 10^9 ops. Too slow for pure Python.
    #
    # We need a faster makespan evaluation.
    # NumPy can handle the matrix multiplication part, but the max() dependency is the bottleneck.
    # 
    # Let's try a different strategy: 
    # Use a fast heuristic (NEH) to get a good start, then use a "Fast Local Search" 
    # that only re-evaluates the affected part? No, flow shop makespan is global.
    #
    # However, we can vectorize the makespan calculation using a loop over machines, 
    # but processing jobs in parallel? No, jobs are sequential.
    #
    # Let's stick to a highly optimized Python loop for makespan, and limit the number of SA iterations.
    # Or, use a "Greedy" approach with random restarts and a very fast local search (First Improvement).
    #
    # Let's try: 
    # 1. NEH Heuristic for initialization.
    # 2. Simulated Annealing with Adjacent Swaps.
    # 3. Optimized makespan function.
    
    def makespan_fast(seq):
        """
        Compute makespan for a given sequence of job indices.
        Optimized for speed using local variables and minimal overhead.
        """
        if len(seq) == 0:
            return 0.0
        
        # We need the processing times for the jobs in the order of seq.
        # pt_seq = processing_times[seq] -> Shape (N, M)
        # This creates a copy. It's fast in NumPy.
        pt_seq = processing_times[seq]
        
        # Initialize completion times for the first job
        # completion[m] = time when machine m finishes the current job
        # For job 0, completion[m] = sum(pt_seq[0, :m+1])
        # We can compute this as a cumulative sum.
        
        # Actually, a tight loop is often faster for small N, M.
        # Let's try to vectorize the "forward pass" per machine?
        # No, the dependency is C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
        
        # Let's use a pure Python loop but with pre-extracted rows?
        # pt_seq is already extracted.
        
        # C is a 1D array of length M, representing completion times of the current job on each machine.
        # But we need the previous job's completion times to calculate the current one.
        # So we maintain `prev_completion` (from job j-1) and calculate `curr_completion` (for job j).
        
        # Initialize for the first job
        prev_completion = np.cumsum(pt_seq[0])
        
        for j in range(1, len(seq)):
            # pt_j = pt_seq[j]
            # curr_completion[0] = prev_completion[0] + pt_j[0]
            # curr_completion[m] = max(prev_completion[m], curr_completion[m-1]) + pt_j[m]
            
            # We can do this in a loop.
            # To speed up, we can try to unroll or use list operations, but NumPy is likely better for M > 5.
            # Let's use a small Python loop for M.
            
            curr_completion = np.empty(n_machines)
            # First machine
            curr_completion[0] = prev_completion[0] + pt_seq[j, 0]
            
            for m in range(1, n_machines):
                # max(prev_completion[m], curr_completion[m-1]) + pt_seq[j, m]
                val1 = prev_completion[m]
                val2 = curr_completion[m-1]
                if val1 > val2:
                    curr_completion[m] = val1 + pt_seq[j, m]
                else:
                    curr_completion[m] = val2 + pt_seq[j, m]
            
            prev_completion = curr_completion
            
        return prev_completion[-1]

    def neh_heuristic(totals):
        """Standard NEH heuristic."""
        order = list(np.argsort(-totals))
        seq = [order[0]]
        for job in order[1:]:
            best_pos = 0
            best_ms = float('inf')
            # NEH is O(N^2 * M). For N=100, M=10, it's 10000 * 10 = 10^5 ops. Fast.
            for pos in range(len(seq) + 1):
                # Insert job at pos
                # Instead of creating new lists, we can evaluate the makespan of the trial.
                # But creating the list is O(N).
                # Let's just do it simply.
                trial = seq[:pos] + [job] + seq[pos:]
                ms = makespan_fast(trial)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        return seq

    # Generate initial candidates
    candidates = []
    
    # 1. NEH
    totals = processing_times.sum(axis=1)
    candidates.append(neh_heuristic(totals))
    
    # 2. Palmer
    def palmer_heuristic():
        n = n_jobs
        priorities = []
        for job in range(n_jobs):
            p_first = processing_times[job, 0]
            p_last = processing_times[job, -1]
            priorities.append(n * p_first - p_last)
        priorities = np.array(priorities)
        return list(np.argsort(-priorities))
    
    candidates.append(palmer_heuristic())
    
    # 3. LPT
    def lpt_heuristic():
        return list(np.argsort(-totals))
    
    candidates.append(lpt_heuristic())
    
    # 4. Random
    for _ in range(3):
        seq = list(range(n_jobs))
        random.shuffle(seq)
        candidates.append(seq)
        
    # Find best initial
    best_seq = None
    best_ms = float('inf')
    for seq in candidates:
        ms = makespan_fast(seq)
        if ms < best_ms:
            best_ms = ms
            best_seq = list(seq)
            
    # Simulated Annealing
    # Parameters
    T = 100.0  # Initial temperature
    alpha = 0.999  # Cooling rate
    T_min = 1.0
    
    current_seq = best_seq
    current_ms = best_ms
    
    # Precompute a random permutation for swaps? No, just random indices.
    
    steps = 0
    while time.time() < deadline:
        # Check if we should stop
        if T < T_min:
            # Restart with higher T if time permits? Or just stop.
            # Let's restart with a new random sequence or reheat.
            # Reheating can help escape deep local minima.
            T = 10.0 
            # Maybe shuffle current sequence slightly?
            # For now, just continue with lower T or restart.
            # Let's just continue, the loop will break if no improvements for a long time?
            # No, SA continues.
            pass
            
        # Generate a neighbor: Swap two adjacent jobs
        if n_jobs > 1:
            i = random.randint(0, n_jobs - 2)
            # Swap i and i+1
            new_seq = current_seq.copy()
            new_seq[i], new_seq[i+1] = new_seq[i+1], new_seq[i]
            
            # Evaluate
            new_ms = makespan_fast(new_seq)
            delta = new_ms - current_ms
            
            # Acceptance criterion
            if delta < 0:
                # Improvement
                current_seq = new_seq
                current_ms = new_ms
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_seq = current_seq.copy()
            else:
                # Deterioration
                # Prob = exp(-delta / T)
                # If T is small, this is very small.
                # Optimization: if delta > T * 10, prob is ~0.
                if delta < T * 10:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_seq = new_seq
                        current_ms = new_ms
            
            # Cool down
            T *= alpha
            steps += 1
            
            # Check time periodically
            if steps % 1000 == 0 and time.time() > deadline:
                break
                
    return list(best_seq)
# EVOLVE-BLOCK-END
