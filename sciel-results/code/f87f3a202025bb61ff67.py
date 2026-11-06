import numpy as np
import time
import random

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # Helper to calculate makespan for a given permutation
    # Optimized for speed: uses 1D array for machine completion times
    def calc_makespan(perm):
        C = np.zeros(n_machines)
        P = processing_times
        n_m = n_machines
        
        for j in perm:
            pj = P[j]
            # First machine
            C[0] += pj[0]
            curr_completion = C[0]
            
            # Subsequent machines
            for m in range(1, n_m):
                # C[m] is the completion time of the previous job on machine m
                # curr_completion is the completion time of the current job on machine m-1
                if C[m] > curr_completion:
                    curr_completion = C[m] + pj[m]
                else:
                    curr_completion = curr_completion + pj[m]
                C[m] = curr_completion
        return C[-1]

    # 1. NEH Construction
    # Sum processing times for each job
    sums = processing_times.sum(axis=1)
    # Sort by sum descending (NEH rule)
    order = np.argsort(-sums)
    current_perm = list(order)
    current_makespan = calc_makespan(current_perm)
    best_perm = current_perm.copy()
    best_makespan = current_makespan
    
    # 2. Simulated Annealing with Insertion Moves
    # Temperature settings
    T = 10.0
    T_min = 1e-5
    alpha = 0.995 # Cooling rate
    
    # Track no-improvement steps for potential restarts
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if T < T_min or no_improve_count > max_no_improve:
            # Restart
            T = 10.0
            no_improve_count = 0
            # Perturb best solution
            current_perm = best_perm.copy()
            # Perform a few random insertions to escape local optimum
            for _ in range(5):
                i = random.randint(0, n_jobs - 1)
                job = current_perm.pop(i)
                k = random.randint(0, n_jobs - 1)
                current_perm.insert(k, job)
            current_makespan = calc_makespan(current_perm)
        
        # Generate a neighbor via random insertion
        # Pick a job to move
        i = random.randint(0, n_jobs - 1)
        job_to_move = current_perm[i]
        # Remove it
        current_perm.pop(i)
        
        # Pick a new position
        # Insertion position k in [0, n_jobs-1] (since list length is n_jobs-1 now)
        k = random.randint(0, n_jobs - 1)
        current_perm.insert(k, job_to_move)
        
        new_makespan = calc_makespan(current_perm)
        
        delta = new_makespan - current_makespan
        
        # Metropolis acceptance criterion
        accepted = False
        if delta < 0:
            accepted = True
        else:
            # Only compute exp if delta is not too large to save time
            if delta < 50 * T:
                if random.random() < np.exp(-delta / T):
                    accepted = True
        
        if accepted:
            current_makespan = new_makespan
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_perm = current_perm.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Cool down
        T *= alpha
        
    return best_perm
