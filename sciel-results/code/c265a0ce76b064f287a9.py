"""
Flow-shop permutation construction using Priority-List Dispatching and Differential Evolution.
Strategy:
1. Represent the solution as a vector of continuous priorities (one per job).
2. The permutation is derived by sorting jobs by their priorities.
3. Use Differential Evolution (DE) to optimize the priority vector.
4. The fitness function computes the makespan of the permutation induced by the priorities.
5. This shifts the search from combinatorial permutation space to a continuous landscape,
   allowing gradient-free global optimization methods to escape local minima effectively.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _priority_to_permutation(priorities):
    """
    Convert a vector of priorities to a job permutation.
    Jobs are sorted by priority (ascending). Ties are broken by job index to ensure determinism.
    """
    return np.argsort(priorities)

def _makespan(order, P):
    """
    Compute makespan for a given job order.
    P: (n_jobs, n_machines)
    order: array of job indices
    """
    if len(order) == 0:
        return 0.0
    
    n_machines = P.shape[1]
    c_machines = np.zeros(n_machines)
    
    # Pre-extract rows for the sequence
    seq_P = P[order]
    
    for j in range(len(order)):
        row = seq_P[j]
        c_curr = 0.0
        # Inner loop over machines
        # Using local variables for speed
        cm = c_machines
        for m in range(n_machines):
            val = cm[m]
            if c_curr > val:
                c_curr += row[m]
            else:
                c_curr = val + row[m]
            cm[m] = c_curr
            
    return c_machines[-1]

def _fitness(priorities, P):
    """
    Fitness function for DE.
    Returns the makespan for the permutation induced by the priorities.
    """
    order = _priority_to_permutation(priorities)
    return _makespan(order, P)

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using Differential Evolution on priority lists.
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
    time_limit = 145.0  # Safety margin before 150s kill

    # DE Parameters
    pop_size = max(10, 5 * n_jobs)
    
    rng = np.random.default_rng()
    
    # Initial population: random priorities in [0, 1]
    population = rng.random((pop_size, n_jobs))
    
    # Evaluate initial population
    fitnesses = np.zeros(pop_size)
    for i in range(pop_size):
        fitnesses[i] = _fitness(population[i], P)
    
    best_idx = np.argmin(fitnesses)
    best_individual = population[best_idx].copy()
    best_fitness = fitnesses[best_idx]
    
    # DE Parameters
    F = 0.8  # Scale factor
    CR = 0.9  # Crossover probability
    
    iter_count = 0
    while (time.time() - start_time) < time_limit:
        iter_count += 1
        
        # Check time
        if iter_count % 5 == 0 and (time.time() - start_time) > time_limit:
            break

        # Generate offspring for each individual in population
        for i in range(pop_size):
            # Select three distinct random individuals r1, r2, r3 distinct from i
            # Efficient selection
            r1, r2, r3 = rng.integers(0, pop_size, 3)
            # Ensure distinctness and not equal to i
            # Simple retry loop is fast enough for small pop sizes
            while r1 == i or r2 == i or r3 == i or r1 == r2 or r2 == r3 or r1 == r3:
                r1, r2, r3 = rng.integers(0, pop_size, 3)
            
            # Mutation: v = x_r1 + F * (x_r2 - x_r3)
            v = population[r1] + F * (population[r2] - population[r3])
            
            # Crossover
            # Binomial crossover
            j_rand = rng.integers(0, n_jobs)
            u = np.empty_like(population[i])
            # Vectorized crossover logic is tricky with the "must include j_rand" constraint
            # but we can do it in a loop or use masks.
            # For speed, a simple loop is often faster than complex numpy masks for small N.
            for j in range(n_jobs):
                if rng.random() < CR or j == j_rand:
                    u[j] = v[j]
                else:
                    u[j] = population[i][j]
            
            # Evaluate offspring
            f_u = _fitness(u, P)
            
            # Selection
            if f_u <= fitnesses[i]:
                population[i] = u
                fitnesses[i] = f_u
                if f_u < best_fitness:
                    best_fitness = f_u
                    best_individual = u.copy()
    
    # Return the best permutation found
    best_order = _priority_to_permutation(best_individual)
    return best_order.tolist()

# EVOLVE-BLOCK-END
