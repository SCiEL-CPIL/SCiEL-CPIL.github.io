import numpy as np
import time

def _calc_makespan(perm, processing_times, upper_bound=float('inf')):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    
    C_prev = np.zeros(n_machines)
    for job_idx in perm:
        p = processing_times[job_idx]
        C_curr = np.empty(n_machines)
        for m in range(n_machines):
            if m == 0:
                C_curr[0] = C_prev[0] + p[0]
            else:
                C_curr[m] = max(C_prev[m], C_curr[m-1]) + p[m]
            
            if C_curr[m] > upper_bound:
                return C_curr[m]
        C_prev = C_curr
    return float(C_prev[-1])

def _grasp_construct(processing_times, alpha, rng):
    """
    Construct a permutation using GRASP (Greedy Randomized Adaptive Search Procedure).
    alpha controls the greediness. 0 = greedy, 1 = random.
    """
    n_jobs, n_machines = processing_times.shape
    available = list(range(n_jobs))
    current_perm = []
    C_prev = np.zeros(n_machines)
    
    while available:
        if not available:
            break
            
        # Calculate the increase in makespan for adding each available job at the end
        # This is an approximation of the "cost" of adding the job now.
        # A better greedy function might consider the full sequence, but appending is O(n_m).
        # To make it more "flow-shop aware", we can simulate the completion time.
        
        best_increase = float('inf')
        worst_increase = float('-inf')
        
        candidates = []
        for job in available:
            p = processing_times[job]
            # Simulate adding this job to the current sequence
            C_curr = np.empty(n_machines)
            for m in range(n_machines):
                if m == 0:
                    C_curr[0] = C_prev[0] + p[0]
                else:
                    C_curr[m] = max(C_prev[m], C_curr[m-1]) + p[m]
            increase = C_curr[-1] - C_prev[-1]
            
            if increase < best_increase:
                best_increase = increase
            if increase > worst_increase:
                worst_increase = increase
            candidates.append((job, increase))
        
        # Define Restricted Candidate List (RCL)
        threshold = best_increase + alpha * (worst_increase - best_increase)
        rcl = [job for job, inc in candidates if inc <= threshold]
        
        # Select a random job from RCL
        selected_job = rng.choice(rcl)
        current_perm.append(selected_job)
        available.remove(selected_job)
        
        # Update C_prev to reflect the addition of selected_job
        p = processing_times[selected_job]
        C_curr = np.empty(n_machines)
        for m in range(n_machines):
            if m == 0:
                C_curr[0] = C_prev[0] + p[0]
            else:
                C_curr[m] = max(C_prev[m], C_curr[m-1]) + p[m]
        C_prev = C_curr
        
    return current_perm

def _local_search_insertion(perm, processing_times, upper_bound):
    """
    Best-improvement insertion local search.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm, _calc_makespan(perm, processing_times)
        
    best_perm = list(perm)
    best_makespan = _calc_makespan(best_perm, processing_times)
    
    while True:
        improved = False
        best_new_perm = None
        best_new_makespan = best_makespan
        
        # Try moving each job to every other position
        for i in range(n_jobs):
            job = best_perm[i]
            # Remove job i
            perm_without = best_perm[:i] + best_perm[i+1:]
            
            # Try inserting at every position in perm_without
            for j in range(len(perm_without) + 1):
                # Skip original position (j == i) and equivalent (j == i+1 in new list is same as i in old)
                # Actually, inserting at j in the list of length n-1.
                # If j == i, it goes back to original.
                # If j == i+1, it goes to the position after the original (which is i in 0-indexed of new list? No.)
                # Let's just check all j.
                
                new_perm = perm_without[:j] + [job] + perm_without[j:]
                
                # Optimization: Skip if new_perm is same as best_perm
                if new_perm == best_perm:
                    continue
                    
                new_makespan = _calc_makespan(new_perm, processing_times, upper_bound=best_new_makespan)
                
                if new_makespan < best_new_makespan:
                    best_new_makespan = new_makespan
                    best_new_perm = new_perm
        
        if best_new_perm is not None and best_new_makespan < best_makespan:
            best_perm = best_new_perm
            best_makespan = best_new_makespan
            improved = True
        else:
            break
            
    return best_perm, best_makespan

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    rng = np.random.default_rng(42)
    
    best_perm = None
    best_makespan = float('inf')
    
    # 1. Initial Greedy (LPT-like) to get a decent start
    # Sort by sum of processing times descending
    sums = np.sum(processing_times, axis=1)
    init_perm = list(np.argsort(-sums, kind='stable'))
    init_ms = _calc_makespan(init_perm, processing_times)
    best_perm = init_perm
    best_makespan = init_ms
    
    # 2. Multi-start GRASP
    while time.time() - start_time < time_limit:
        # Vary alpha to explore different trade-offs between greediness and randomness
        # Start with greedy (low alpha) and increase
        alpha = 0.1 + 0.8 * (rng.random() ** 2) # Bias towards lower alpha (greedy)
        
        # Construct
        perm = _grasp_construct(processing_times, alpha, rng)
        
        # Local Search
        perm_ls, ms_ls = _local_search_insertion(perm, processing_times, upper_bound=best_makespan)
        
        if ms_ls < best_makespan:
            best_makespan = ms_ls
            best_perm = perm_ls
            
        # Check time
        if time.time() - start_time > time_limit:
            break
            
    return best_perm
