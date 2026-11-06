"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(order, processing_times):
    """
    Fast makespan calculation.
    order: list of job indices
    processing_times: np.ndarray [n_jobs, n_machines]
    """
    n_machines = processing_times.shape[1]
    if len(order) == 0:
        return 0.0
    
    # prev[m] is completion time of previous job on machine m
    prev = np.zeros(n_machines)
    
    for job in order:
        p = processing_times[job]
        # Machine 0
        curr_0 = prev[0] + p[0]
        # Machines 1..M-1
        curr_m_minus_1 = curr_0
        for m in range(1, n_machines):
            curr_m = max(curr_m_minus_1, prev[m]) + p[m]
            prev[m] = curr_m
            curr_m_minus_1 = curr_m
        prev[0] = curr_0
        
    return prev[-1]

def _grasp_construct(processing_times, alpha=0.5):
    """
    Construct a solution using GRASP (Greedy Randomized Adaptive Search Procedure).
    1. Start with empty sequence.
    2. At each step, compute cost of adding each remaining job to the end (or best position).
    3. Build Restricted Candidate List (RCL) of jobs with cost <= (min_cost + alpha * (max_cost - min_cost)).
    4. Pick randomly from RCL.
    """
    n_jobs, n_machines = processing_times.shape
    remaining = set(range(n_jobs))
    sequence = []
    
    for _ in range(n_jobs):
        if not remaining:
            break
            
        # Calculate insertion cost for each job in remaining into the current sequence
        # For efficiency, we usually check inserting at the end or best position.
        # Standard GRASP for flow shop often uses "insertion into best position" for the greedy step.
        # To keep it fast, we'll evaluate inserting each job at every position in the current sequence.
        
        best_costs = []
        for job in remaining:
            min_cost = float('inf')
            for pos in range(len(sequence) + 1):
                candidate = sequence[:pos] + [job] + sequence[pos:]
                ms = _makespan(candidate, processing_times)
                if ms < min_cost:
                    min_cost = ms
            best_costs.append((min_cost, job))
            
        if not best_costs:
            continue
            
        # Sort by cost
        best_costs.sort(key=lambda x: x[0])
        min_c = best_costs[0][0]
        max_c = best_costs[-1][0]
        
        # Build RCL
        threshold = min_c + alpha * (max_c - min_c)
        rcl = [job for cost, job in best_costs if cost <= threshold]
        
        # Pick randomly from RCL
        chosen_job = np.random.choice(rcl)
        
        # Find best position for chosen_job
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [chosen_job] + sequence[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
                
        sequence.insert(best_pos, chosen_job)
        remaining.remove(chosen_job)
        
    return sequence

def _vns_local_search(initial_seq, processing_times, max_neighborhoods=5, max_iters_per_n=100):
    """
    Variable Neighborhood Search (VNS) Local Search.
    Uses insertion moves with increasing radius.
    """
    current_seq = initial_seq[:]
    current_ms = _makespan(current_seq, processing_times)
    best_seq = current_seq[:]
    best_ms = current_ms
    n_jobs = len(current_seq)
    
    # Neighborhoods are defined by the maximum distance a job can be moved
    # k=1: move job by 1 position
    # k=2: move job by 2 positions
    # ...
    
    for k in range(1, max_neighborhoods + 1):
        found_improvement = False
        # Limit iterations to prevent timeout in large neighborhoods
        for _ in range(max_iters_per_n):
            improved = False
            # Randomly select a job and a direction
            job_idx = np.random.randint(n_jobs)
            job = current_seq[job_idx]
            
            # Determine valid insertion positions within distance k
            # Remove job
            temp_seq = current_seq[:job_idx] + current_seq[job_idx+1:]
            n_temp = len(temp_seq)
            
            # Positions: job_idx - k to job_idx + k (clamped to 0..n_temp)
            start_pos = max(0, job_idx - k)
            end_pos = min(n_temp, job_idx + k + 1)
            
            best_pos_in_n = -1
            best_ms_in_n = float('inf')
            
            for pos in range(start_pos, end_pos + 1):
                if pos == job_idx: # Skip original position
                    continue
                candidate = temp_seq[:pos] + [job] + temp_seq[pos:]
                ms = _makespan(candidate, processing_times)
                if ms < best_ms_in_n:
                    best_ms_in_n = ms
                    best_pos_in_n = pos
            
            if best_pos_in_n != -1 and best_ms_in_n < current_ms:
                # Accept improvement
                candidate = temp_seq[:best_pos_in_n] + [job] + temp_seq[best_pos_in_n:]
                current_seq = candidate
                current_ms = best_ms_in_n
                improved = True
                found_improvement = True
                
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_seq = current_seq[:]
                    
        if not found_improvement:
            # Move to next neighborhood structure
            continue
        else:
            # If improvement found, restart from k=1? Or continue?
            # Standard VNS: if improvement, go back to first neighborhood.
            # But for speed in this context, we might just proceed or restart.
            # Let's restart from k=1 to ensure thorough local search.
            # However, to save time, we can just continue if we found improvement, 
            # but typically VNS resets k to 1.
            # Given time constraints, let's reset k to 1 if improved.
            # To avoid infinite loops, we limit total iterations.
            pass 

    return best_seq, best_ms

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    start_time = time.time()
    timeout = 110.0 # Keep 10s buffer
    
    best_global_seq = []
    best_global_ms = float('inf')
    
    # Number of GRASP iterations
    if n_jobs <= 10:
        grasp_iters = 50
    elif n_jobs <= 20:
        grasp_iters = 30
    else:
        grasp_iters = 15
        
    # VNS parameters
    max_neighborhoods = 5
    max_iters_per_n = 50 if n_jobs > 15 else 100
    
    for i in range(grasp_iters):
        if time.time() - start_time > timeout:
            break
            
        # Vary alpha for diversity
        alpha = 0.1 + 0.8 * (i / grasp_iters)
        seq = _grasp_construct(processing_times, alpha=alpha)
        
        # Local Search with VNS
        improved_seq, improved_ms = _vns_local_search(seq, processing_times, 
                                                      max_neighborhoods=max_neighborhoods, 
                                                      max_iters_per_n=max_iters_per_n)
        
        if improved_ms < best_global_ms:
            best_global_ms = improved_ms
            best_global_seq = improved_seq
            
    # If no iterations ran or result is empty, fallback to simple greedy
    if not best_global_seq:
        # Simple greedy: sort by total processing time
        total_times = processing_times.sum(axis=1)
        best_global_seq = list(np.argsort(total_times))
        
    return best_global_seq
# EVOLVE-BLOCK-END
