"""
Flow-shop permutation construction using Differential Evolution on a Priority-List Model.

Strategy:
1. Represent the job order as a vector of continuous weights w of length N.
2. The permutation is obtained by sorting the jobs by their weights (stable sort).
3. Use Differential Evolution (DE) to optimize the weights to minimize the makespan.
4. DE is a global optimizer that works well in continuous spaces and avoids local minima
   more effectively than simple local search on permutations.
5. We implement a lightweight DE loop with time management to fit within the 150s limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    order: array of job indices
    P: (n_jobs, n_machines)
    """
    if len(order) == 0:
        return 0.0
    n_machines = P.shape[1]
    
    # Vectorized-ish makespan calculation
    # C[m] is the completion time of the last job on machine m
    c_machines = np.zeros(n_machines)
    
    # We iterate through jobs. To speed this up, we process the sequence.
    # The makespan is the maximum completion time over all jobs and machines,
    # but in a flow shop, the makespan is simply the completion time of the last job on the last machine.
    # However, we must update the machine states correctly.
    
    # For each job j in order:
    #   c_curr = 0
    #   for m in range(n_machines):
    #       c_curr = max(c_machines[m], c_curr) + P[j, m]
    #       c_machines[m] = c_curr
    #   makespan = c_curr (of the last job)
    
    # This loop is O(N*M). We can't easily vectorize the dependency across machines for a single job,
    # but we can use a Python loop which is acceptable for N~100-500 if we limit the number of evaluations.
    
    c_curr = 0.0
    for j in order:
        row = P[j]
        for m in range(n_machines):
            # c_curr is the completion time of job j on machine m-1
            # c_machines[m] is the completion time of the previous job on machine m
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
            
    return c_curr

def _weights_to_order(weights):
    """
    Convert a weight vector to a job order (permutation) by stable sorting.
    """
    # np.argsort is stable, so ties are broken by original index
    return np.argsort(weights)

def _evaluate_weights(weights, P):
    """
    Evaluate the makespan for a given weight vector.
    """
    order = _weights_to_order(weights)
    return _makespan(order, P)

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using Differential Evolution on a priority-list model.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # 1. Initialize with NEH-like heuristic to get a good starting point
    # We will use the NEH heuristic to generate an initial weight vector that corresponds to the NEH order.
    # Actually, it's easier to just start DE with random weights, but a good initial population helps.
    # Let's generate a few initial individuals using NEH and random permutations.
    
    # NEH Heuristic
    def _neh_heuristic(P):
        n_jobs, n_machines = P.shape
        totals = P.sum(axis=1)
        order_indices = np.argsort(-totals)
        
        current_seq = []
        for idx in order_indices:
            best_pos = 0
            best_makespan = float('inf')
            
            for pos in range(len(current_seq) + 1):
                new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
                ms = _makespan(new_seq, P)
                if ms < best_makespan:
                    best_makespan = ms
                    best_pos = pos
            
            current_seq.insert(best_pos, idx)
        
        return current_seq

    neh_order = _neh_heuristic(P)
    
    # Create a weight vector that produces the NEH order
    # If we assign weights such that the rank of the weights matches the position in the NEH order,
    # then sorting the weights will yield the NEH order.
    # Let w[j] = position of j in neh_order.
    weights_neh = np.zeros(n_jobs)
    for pos, job in enumerate(neh_order):
        weights_neh[job] = pos
    
    # 2. Differential Evolution Setup
    # Population size
    pop_size = min(50, n_jobs * 2)
    if pop_size < 5:
        pop_size = 5
        
    # DE parameters
    F = 0.8  # Mutation factor
    CR = 0.9  # Crossover rate
    
    # Initialize population
    # One individual is the NEH weights
    # Others are random
    pop = np.random.rand(pop_size, n_jobs)
    pop[0] = weights_neh
    
    # Evaluate initial population
    fits = np.array([_evaluate_weights(ind, P) for ind in pop])
    
    best_idx = np.argmin(fits)
    best_weights = pop[best_idx].copy()
    best_makespan = fits[best_idx]
    
    # 3. DE Loop
    generation = 0
    while (time.time() - start_time) < time_limit:
        generation += 1
        
        for i in range(pop_size):
            # Select three distinct random individuals
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            while r1 == i or r2 == i or r3 == i:
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            v = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Crossover
            trial = pop[i].copy()
            for j in range(n_jobs):
                if np.random.rand() < CR or j == np.random.randint(n_jobs):
                    trial[j] = v[j]
            
            # Evaluate trial
            trial_fit = _evaluate_weights(trial, P)
            
            # Selection
            if trial_fit < fits[i]:
                pop[i] = trial
                fits[i] = trial_fit
                if trial_fit < best_makespan:
                    best_makespan = trial_fit
                    best_weights = trial.copy()
        
        # Check time
        if (time.time() - start_time) > time_limit - 5.0:
            break
    
    # 4. Final Local Search on the best weight vector's permutation
    # Convert best weights to order
    best_order = _weights_to_order(best_weights)
    
    # Perform a quick local search (2-opt or insertion) on the resulting permutation to polish
    # This helps recover from any "noise" in the weight-to-order mapping.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit - 1.0:
        improved = False
        # Simple 2-opt: swap adjacent jobs
        for i in range(len(best_order) - 1):
            new_order = best_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            ms = _makespan(new_order, P)
            if ms < best_makespan:
                best_makespan = ms
                best_order = new_order
                improved = True
                # Restart or continue? Continue for speed, but might miss global improvements.
                # Given time constraints, we break after first improvement to save time for other swaps?
                # No, just break inner loop to re-evaluate from start?
                # Let's just break the for loop and restart the while loop.
                break
    
    return best_order.tolist()
# EVOLVE-BLOCK-END
