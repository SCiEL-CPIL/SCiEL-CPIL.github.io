"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(perm, processing_times):
    """
    Optimized makespan calculation for flow shop.
    """
    n_jobs, n_machines = processing_times.shape
    # completion[m] is the time when machine m becomes free
    completion = [0.0] * n_machines
    
    for job in perm:
        p = processing_times[job]
        prev_finish = 0.0
        for m in range(n_machines):
            # Start time is max(machine free time, job finished previous machine)
            start = completion[m] if completion[m] > prev_finish else prev_finish
            finish = start + p[m]
            completion[m] = finish
            prev_finish = finish
            
    return completion[-1]

def _insert_job(perm, job, pos, processing_times):
    """
    Insert job into perm at position pos and return makespan.
    This is used during NEH construction.
    """
    # Create a temporary list to calculate makespan
    # For efficiency in NEH, we can calculate the makespan of the new permutation
    # directly. Since n is small, creating a new list is fine.
    new_perm = perm[:pos] + [job] + perm[pos:]
    return _makespan(new_perm, processing_times)

def neh_heuristic(processing_times):
    """
    Nawaz, Enscore, and Ham (NEH) heuristic for Flow Shop.
    """
    n_jobs, n_machines = processing_times.shape
    
    # 1. Calculate total processing time for each job
    total_times = np.sum(processing_times, axis=1)
    
    # 2. Sort jobs in descending order of total processing time
    jobs = list(range(n_jobs))
    jobs.sort(key=lambda j: -total_times[j])
    
    # 3. Construct schedule
    schedule = []
    for job in jobs:
        best_pos = -1
        best_cost = float('inf')
        
        # Try inserting job into all possible positions
        for pos in range(len(schedule) + 1):
            # Calculate makespan if inserted at pos
            # Optimization: We can compute this incrementally, but for simplicity and correctness first:
            new_perm = schedule[:pos] + [job] + schedule[pos:]
            cost = _makespan(new_perm, processing_times)
            
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        # Insert job at best position
        schedule.insert(best_pos, job)
        
    return schedule

def insertion_local_search(perm, processing_times, max_iters=1000, time_limit=None, start_time=0):
    """
    Local search using insertion moves.
    Moves a job to a different position in the sequence.
    """
    n_jobs, n_machines = processing_times.shape
    current_cost = _makespan(perm, processing_times)
    best_perm = perm[:]
    best_cost = current_cost
    
    iters = 0
    while iters < max_iters:
        if time_limit and (time.time() - start_time > time_limit):
            break
        iters += 1
        
        # Pick a random job
        i = random.randint(0, n_jobs - 1)
        job = perm[i]
        
        # Remove it
        perm.pop(i)
        
        # Try inserting it into all other positions
        best_pos = i
        best_cost_move = current_cost
        
        for pos in range(n_jobs):
            # Insert at pos
            perm.insert(pos, job)
            cost = _makespan(perm, processing_times)
            
            if cost < best_cost_move:
                best_cost_move = cost
                best_pos = pos
            
            # Remove again to try next position
            perm.pop(pos)
        
        # Apply the best move found
        perm.insert(best_pos, job)
        current_cost = best_cost_move
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_perm = perm[:]
            
        # If no improvement was found in this iteration, we could break early
        # but usually we continue to explore. To speed up, we can check if best_pos == i
        # which means no better position was found.
        if best_pos == i:
             # Optional: Early stop if no improvement for many steps? 
             # For now, just continue.
             pass

    return best_perm, best_cost

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    # Time management
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # 1. Initial solution using NEH
    best_perm = neh_heuristic(processing_times)
    best_cost = _makespan(best_perm, processing_times)
    
    # 2. Improve with Local Search
    best_perm, best_cost = insertion_local_search(
        best_perm, processing_times, 
        max_iters=1000, 
        time_limit=time_limit, 
        start_time=start_time
    )
    
    # 3. Iterated Local Search (ILS)
    # Perturb the solution and re-optimize
    # We use a simple perturbation: random insertion of a few jobs
    
    while time.time() - start_time < time_limit:
        # Create a perturbed version of the best solution
        perturbed_perm = best_perm[:]
        
        # Perform 2-3 random insertions
        for _ in range(3):
            i = random.randint(0, n_jobs - 1)
            j = random.randint(0, n_jobs - 1)
            if i == j:
                continue
            job = perturbed_perm.pop(i)
            perturbed_perm.insert(j, job)
        
        # Local search on the perturbed solution
        new_perm, new_cost = insertion_local_search(
            perturbed_perm, processing_times,
            max_iters=500,  # Fewer iterations for ILS loop
            time_limit=time_limit,
            start_time=start_time
        )
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_perm = new_perm
        else:
            # If no improvement, we might want to try a stronger perturbation
            # or just continue.
            pass

    return best_perm
# EVOLVE-BLOCK-END
