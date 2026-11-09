"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def _makespan(processing_times, order):
    """Compute makespan for a given job order using vectorized numpy operations."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    # Extract processing times in the order of jobs
    # p[i, j] is processing time of job order[i] on machine j
    p = processing_times[order, :]
    
    # C[i, j] is completion time of job i on machine j
    # C[i, 0] = sum(p[0..i, 0])
    # C[i, j] = max(C[i-1, j], C[i, j-1]) + p[i, j]
    
    # We can compute this iteratively but vectorized along the job axis is tricky due to dependencies.
    # However, for moderate N, a python loop over machines is fast if the inner job loop is vectorized?
    # No, the dependency is sequential along jobs for a fixed machine, and sequential along machines for a fixed job.
    # C[i,j] depends on C[i-1,j] and C[i,j-1].
    
    # Let's stick to a tight loop but optimize it.
    # Actually, we can use a simple loop which is often faster than complex numpy indexing for small N.
    # But let's try to make it as fast as possible.
    
    completion = np.zeros((n_jobs, n_machines))
    
    # First machine is just cumulative sum
    completion[:, 0] = np.cumsum(p[:, 0])
    
    # For remaining machines
    for j in range(1, n_machines):
        # C[i, j] = max(C[i-1, j], C[i, j-1]) + p[i, j]
        # We can compute this in a loop over i, or potentially use a scan operation if we decouple dependencies?
        # The dependency C[i,j] -> C[i-1,j] is a prefix max/sum kind of thing but mixed with C[i,j-1].
        # It's a standard dynamic programming table.
        
        # Looping over jobs is O(N*M).
        # Let's use a simple loop for clarity and speed in Python for small N.
        for i in range(1, n_jobs):
            # C[i, j] = max(C[i-1, j], C[i, j-1]) + p[i, j]
            # C[i, j-1] is already computed in the previous column
            # C[i-1, j] is the element above in the current column
            
            # We can vectorize the inner loop?
            # C[:, j] depends on C[:, j-1] and C[:-1, j].
            # This is a sequential dependency along the job axis.
            # So we must loop over jobs or use a specialized scan.
            # For N < 1000, a python loop is acceptable.
            
            pass # Placeholder, we will do it properly below

    # Let's rewrite _makespan to be robust and reasonably fast.
    # The previous seed's _makespan was O(N*M) with python loops.
    # We can optimize by noting that C[i,j] = max(C[i-1,j], C[i,j-1]) + p[i,j].
    
    # Let's use the standard efficient loop.
    c = np.zeros((n_jobs, n_machines))
    c[0, 0] = p[0, 0]
    for i in range(1, n_jobs):
        c[i, 0] = c[i-1, 0] + p[i, 0]
    for j in range(1, n_machines):
        c[0, j] = c[0, j-1] + p[0, j]
        
    for i in range(1, n_jobs):
        for j in range(1, n_machines):
            c[i, j] = max(c[i-1, j], c[i, j-1]) + p[i, j]
            
    return c[-1, -1]

def _order_crossover(p1, p2):
    """Order Crossover (OX1)."""
    size = len(p1)
    if size <= 2:
        return list(p1)
    
    # Select two random cut points
    start, end = sorted(np.random.choice(size, 2, replace=False))
    
    # Copy the segment from parent 1
    child = [None] * size
    child[start:end+1] = p1[start:end+1]
    segment = set(p1[start:end+1])
    
    # Fill the rest from parent 2, maintaining order
    fill_index = 0
    pos = 0
    while pos < size:
        if child[pos] is None:
            # Find next job in p2 not in segment
            while p2[fill_index] in segment:
                fill_index += 1
                if fill_index >= size:
                    fill_index = 0
            child[pos] = p2[fill_index]
            segment.add(p2[fill_index])
            fill_index += 1
        pos += 1
        
    return child

def _swap_mutation(ind, rate=0.1):
    """Swap mutation."""
    size = len(ind)
    num_swaps = int(size * rate)
    ind = list(ind)
    for _ in range(num_swaps):
        i, j = np.random.choice(size, 2, replace=False)
        ind[i], ind[j] = ind[j], ind[i]
    return ind

def _inversion_mutation(ind, rate=0.1):
    """Inversion mutation."""
    size = len(ind)
    num_inversions = int(size * rate)
    ind = list(ind)
    for _ in range(num_inversions):
        i, j = sorted(np.random.choice(size, 2, replace=False))
        ind[i:j+1] = reversed(ind[i:j+1])
    return ind

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using a Genetic Algorithm.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Parameters
    pop_size = 100
    generations = 200
    crossover_rate = 0.8
    mutation_rate = 0.2
    tournament_size = 5
    elitism_count = 2
    
    rng = np.random.RandomState(42)
    
    # Initialize population
    # 1. NEH-like (sorted by total processing time)
    total_pt = processing_times.sum(axis=1)
    neh_order = list(np.argsort(-total_pt))
    
    # 2. LPT
    lpt_order = list(np.argsort(-total_pt)) # Same as NEH init usually, let's use different
    # Actually, let's generate diverse initial population
    population = []
    
    # Add heuristic solutions
    population.append(neh_order)
    population.append(list(np.argsort(total_pt))) # SPT
    population.append(list(np.argsort(-processing_times[:, 0]))) # First machine PT
    population.append(list(np.argsort(-processing_times[:, -1]))) # Last machine PT
    
    # Fill rest with random
    while len(population) < pop_size:
        population.append(list(rng.permutation(n_jobs)))
    
    # Evaluate initial population
    fitness = [_makespan(processing_times, ind) for ind in population]
    
    best_order = population[np.argmin(fitness)]
    best_ms = min(fitness)
    
    for gen in range(generations):
        new_population = []
        
        # Elitism
        elite_indices = np.argsort(fitness)[:elitism_count]
        for idx in elite_indices:
            new_population.append(population[idx])
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection
            parents = []
            for _ in range(2):
                # Tournament selection
                competitors = rng.choice(pop_size, tournament_size, replace=False)
                best_comp = competitors[np.argmin([fitness[i] for i in competitors])]
                parents.append(population[best_comp])
            
            # Crossover
            if rng.random() < crossover_rate:
                child = _order_crossover(parents[0], parents[1])
            else:
                child = list(parents[0])
            
            # Mutation
            if rng.random() < mutation_rate:
                # Randomly choose swap or inversion
                if rng.random() < 0.5:
                    child = _swap_mutation(child, mutation_rate)
                else:
                    child = _inversion_mutation(child, mutation_rate)
            
            new_population.append(child)
        
        # Evaluate new population
        population = new_population
        fitness = [_makespan(processing_times, ind) for ind in population]
        
        # Track best
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_ms:
            best_ms = fitness[gen_best_idx]
            best_order = population[gen_best_idx]
            
        # Optional: Early stopping if no improvement for many generations
        # (Omitted for simplicity, but could be added)
        
    return best_order
# EVOLVE-BLOCK-END
