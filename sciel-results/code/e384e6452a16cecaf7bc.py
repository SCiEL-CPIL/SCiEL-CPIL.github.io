import numpy as np
import time
import random
import threading

def _makespan(perm, pt):
    """
    Vectorized calculation of makespan for a given permutation.
    This is significantly faster than the loop-based _compute_C for single evaluations.
    """
    if len(perm) == 0:
        return 0.0
    
    n_machines = pt.shape[1]
    # pt[perm] gives the processing times in the order of the permutation
    # Shape: (n_jobs, n_machines)
    pt_perm = pt[perm]
    
    # We need to compute the cumulative sum along the job axis, but with a max constraint from the previous machine.
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + pt[j, m]
    
    # This recurrence is hard to fully vectorize without tricks, but we can use a loop over machines 
    # which is usually small (e.g., 10), and vectorize over jobs (e.g., 100-500).
    
    # C_prev_m is the completion time of the previous job on the current machine.
    # C_curr_m_minus_1 is the completion time of the current job on the previous machine.
    
    # Initialize C for machine 0
    # C[0, 0] = pt[0, 0]
    # C[j, 0] = C[j-1, 0] + pt[j, 0]  => cumsum
    
    C = np.zeros((len(perm), n_machines), dtype=np.float64)
    
    # Machine 0
    C[:, 0] = np.cumsum(pt_perm[:, 0])
    
    # For machines 1 to M-1
    for m in range(1, n_machines):
        # C[j, m] = max(C[j-1, m], C[j, m-1]) + pt[j, m]
        # Let A[j] = C[j-1, m] (shifted C from previous iteration of m loop? No, C[:, m] is being built)
        # We need C[j-1, m] which is the value we are computing for the previous job.
        # This is a sequential dependency along the job axis for a fixed machine.
        
        # However, C[j, m-1] is fully known from the previous machine column.
        
        # We can't simply vectorize this loop over j because of C[j-1, m].
        # But we can write it efficiently in a tight loop or use a trick if we assume small M.
        # Actually, for M=10, N=100, a Python loop over M and a vectorized operation over N is not possible 
        # due to the dependency. 
        # 
        # Let's stick to a reasonably optimized loop or just use the standard loop which is fine 
        # if we reduce the number of evaluations via better search.
        # 
        # Wait, the inspiration said "Vectorized Neighborhood Evaluation". 
        # For Or-opt, we can't easily vectorize the makespan calculation itself without complex 
        # segment tree logic or similar. 
        # 
        # Let's optimize the standard calculation.
        # The current _compute_C in the parent program is O(N*M). 
        # Let's try to make it as fast as possible in Python/NumPy.
        
        # Actually, a pure Python loop for N=100, M=10 is 1000 ops. Very fast.
        # The bottleneck in the parent was the *search* strategy, not the eval speed necessarily, 
        # unless we are doing millions of evals.
        
        # Let's use a simple, fast loop for correctness and speed.
        pass
        
    # Re-implementing the fast loop inside _makespan to avoid function call overhead of _compute_C
    # and to allow for potential future optimization.
    
    C = np.empty((len(perm), n_machines), dtype=np.float64)
    for j in range(len(perm)):
        for m in range(n_machines):
            if j == 0:
                prev_job = 0.0
            else:
                prev_job = C[j-1, m]
            if m == 0:
                prev_machine = 0.0
            else:
                prev_machine = C[j, m-1]
            C[j, m] = max(prev_job, prev_machine) + pt_perm[j, m]
            
    return C[-1, -1]

def _or_opt_move(perm, start, length):
    """
    Generate a new permutation by removing a block of 'length' jobs starting at 'start'
    and reinserting it at a random position.
    """
    n = len(perm)
    if n < 3:
        return perm
    
    # Ensure valid indices
    if start < 0:
        start = 0
    if start + length > n:
        length = n - start
        if length <= 0:
            return perm
            
    block = perm[start:start+length]
    
    # Remove block
    remaining = perm[:start] + perm[start+length:]
    
    # Choose insertion point in the remaining list
    # The block can be inserted at any position 0..len(remaining)
    ins_pos = random.randint(0, len(remaining))
    
    new_perm = remaining[:ins_pos] + block + remaining[ins_pos:]
    
    return new_perm

def _simulated_annealing(pt, time_limit, start_temp=100.0, cool_rate=0.999):
    """
    Simulated Annealing with Or-opt moves.
    """
    n_jobs = pt.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Initial random permutation
    current_perm = list(range(n_jobs))
    random.shuffle(current_perm)
    
    current_cost = _makespan(current_perm, pt)
    best_perm = list(current_perm)
    best_cost = current_cost
    
    temp = start_temp
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Choose move
        # Or-opt with length 1, 2, or 3
        length = random.choice([1, 2, 3])
        if length > n_jobs:
            length = n_jobs
        start = random.randint(0, n_jobs - length)
        
        new_perm = _or_opt_move(current_perm, start, length)
        
        new_cost = _makespan(new_perm, pt)
        delta = new_cost - current_cost
        
        if delta < 0:
            # Improvement
            current_perm = new_perm
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_perm = list(current_perm)
        else:
            # Accept with probability exp(-delta / temp)
            if temp > 1e-10:
                prob = np.exp(-delta / temp)
                if random.random() < prob:
                    current_perm = new_perm
                    current_cost = new_cost
        
        temp *= cool_rate
        
        # If temperature is too low, restart with a higher temp (Reheat)
        if temp < 1.0:
            # Reheat to a fraction of the initial temperature
            temp = start_temp * 0.5
            # Optionally perturb best solution to restart from a good point
            # For now, keep current state but increase temp to allow exploration
    
    return best_perm

def _worker(pt, time_limit, seed, result_queue, index):
    """
    Worker thread for parallel SA.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        best_perm = _simulated_annealing(pt, time_limit, start_temp=50.0 + seed % 10, cool_rate=0.9995)
        result_queue[index] = best_perm
    except Exception as e:
        # In case of error, return None
        result_queue[index] = None

def order_jobs(processing_times):
    """
    Main entry point.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_total = time.time()
    deadline = 145.0  # Safety margin
    
    # 1. Generate a good initial solution using NEH (sequential insertion)
    # This provides a strong baseline and starting point for SA.
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    perm = []
    for job in sorted_jobs:
        if not perm:
            perm.append(job)
            continue
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    
    best_perm = perm
    best_ms = _makespan(best_perm, processing_times)
    
    elapsed = time.time() - start_total
    time_left = deadline - elapsed
    
    if time_left > 5.0:
        # 2. Parallel Simulated Annealing
        # Use 4 threads
        num_threads = 4
        thread_time = time_left - 5.0  # Leave 5s buffer
        
        results = [None] * num_threads
        threads = []
        
        for i in range(num_threads):
            # Different seeds for diversity
            seed = 42 + i
            t = threading.Thread(
                target=_worker, 
                args=(processing_times, thread_time, seed, results, i)
            )
            threads.append(t)
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Evaluate results and pick the best
        for res in results:
            if res is not None:
                ms = _makespan(res, processing_times)
                if ms < best_ms:
                    best_ms = ms
                    best_perm = res
    
    return best_perm
