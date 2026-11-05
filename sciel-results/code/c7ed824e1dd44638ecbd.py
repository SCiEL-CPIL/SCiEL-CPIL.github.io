"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using a Genetic Algorithm
    with Order Crossover (OX) and vectorized fitness evaluation.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Fast fallback for very large instances where GA might be slow to initialize
    if n_jobs * n_machines > 200000:
        # Use a strong heuristic for large cases
        totals = processing_times.sum(axis=1)
        return list(np.argsort(-totals))

    # Vectorized Makespan Calculation
    # Input: perms (batch, n_jobs)
    # Output: makespans (batch,)
    def calc_makespans(perms):
        # perms: (batch, n_jobs)
        # processing_times: (n_jobs, n_machines)
        
        # Gather processing times for the jobs in the order of perms
        # p: (batch, n_jobs, n_machines)
        p = processing_times[perms]
        
        # Initialize completion times
        # C: (batch, n_machines)
        C = np.zeros((perms.shape[0], n_machines), dtype=np.float64)
        
        # Iterate over jobs in the sequence (axis=1 of perms)
        for j in range(n_jobs):
            # p_j: (batch, n_machines)
            p_j = p[:, j, :]
            
            # For machine 0:
            # C[:, 0] += p_j[:, 0]
            
            # For machine m > 0:
            # C[:, m] = max(C[:, m], C[:, m-1]) + p_j[:, m]
            
            # Vectorized update
            # We can update all machines in a loop, but since n_machines is usually small,
            # this is efficient.
            C[:, 0] += p_j[:, 0]
            for m in range(1, n_machines):
                # C[:, m] = max(C[:, m], C[:, m-1]) + p_j[:, m]
                C[:, m] = np.maximum(C[:, m], C[:, m-1]) + p_j[:, m]
        
        # Makespan is the last machine's completion time for the last job
        return C[:, -1]

    # Heuristics for Initial Population
    def neh_heuristic():
        totals = processing_times.sum(axis=1)
        order = list(np.argsort(-totals))
        seq = [order[0]]
        for job in order[1:]:
            best_pos = 0
            best_ms = float('inf')
            # For NEH, we can just insert at best position
            # To speed up, we can approximate or just do full check since n is small-ish
            for pos in range(len(seq) + 1):
                trial = seq[:pos] + [job] + seq[pos:]
                ms = calc_makespans(np.array([trial]))[0]
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        return seq

    def palmer_heuristic():
        n = n_jobs
        priorities = np.zeros(n_jobs)
        for job in range(n_jobs):
            priorities[job] = n * processing_times[job, 0] - processing_times[job, -1]
        return list(np.argsort(-priorities))

    def lpt_heuristic():
        totals = processing_times.sum(axis=1)
        return list(np.argsort(-totals))

    def bottleneck_heuristic():
        # Sort by max processing time on any machine
        max_times = processing_times.max(axis=1)
        return list(np.argsort(-max_times))

    def random_heuristic():
        seq = list(range(n_jobs))
        random.shuffle(seq)
        return seq

    # Generate Initial Population
    pop_size = 100
    if n_jobs > 50:
        pop_size = 50
    if n_jobs > 100:
        pop_size = 30

    population = []
    
    # Add specific heuristics
    population.append(neh_heuristic())
    population.append(palmer_heuristic())
    population.append(lpt_heuristic())
    population.append(bottleneck_heuristic())
    
    # Fill rest with random
    while len(population) < pop_size:
        population.append(random_heuristic())

    # Convert to numpy array for vectorized ops
    pop = np.array(population, dtype=np.int64)
    
    # Evaluate Initial Population
    fitness = calc_makespans(pop)
    best_idx = np.argmin(fitness)
    best_seq = pop[best_idx].tolist()
    best_ms = fitness[best_idx]
    
    # Genetic Algorithm Loop
    # Parameters
    # Elitism: keep top 5%
    elite_count = max(1, int(pop_size * 0.05))
    
    # Crossover: Order Crossover (OX)
    def ox_crossover(p1, p2):
        # p1, p2 are lists of job indices
        n = len(p1)
        if n <= 2:
            return p1.copy(), p2.copy()
        
        # Choose two cut points
        c1, c2 = sorted(random.sample(range(n), 2))
        
        # Segment from p1
        segment1 = p1[c1:c2+1]
        # Segment from p2
        segment2 = p2[c1:c2+1]
        
        # Fill rest of c1 from p2, skipping elements in segment1
        # Start from c2+1, wrap around to c1
        def fill_child(parent, segment, c1, c2):
            child = [0] * n
            # Copy segment
            child[c1:c2+1] = segment
            
            # Determine remaining elements
            remaining = [x for x in parent if x not in segment]
            
            # Fill positions outside [c1, c2]
            # Positions: c2+1 to n-1, then 0 to c1-1
            pos_idx = 0
            for i in range(c2+1, n):
                child[i] = remaining[pos_idx]
                pos_idx += 1
            for i in range(0, c1):
                child[i] = remaining[pos_idx]
                pos_idx += 1
            return child
        
        child1 = fill_child(p2, segment1, c1, c2)
        child2 = fill_child(p1, segment2, c1, c2)
        
        return child1, child2

    # Mutation: Swap Mutation
    def swap_mutation(ind, rate=0.1):
        # ind is a list
        n = len(ind)
        if n < 2:
            return ind
        # Number of swaps
        num_swaps = max(1, int(n * rate))
        # To avoid too many swaps, we can just do 1 or 2
        # Standard GA often uses 1 swap or inversion
        # Let's do a random inversion for better diversity
        if random.random() < 0.5:
            # Swap
            i, j = random.sample(range(n), 2)
            ind[i], ind[j] = ind[j], ind[i]
        else:
            # Inversion
            i, j = sorted(random.sample(range(n), 2))
            ind[i:j+1] = reversed(ind[i:j+1])
        return ind

    # Selection: Tournament Selection
    def tournament_select(pop, fit, k=3):
        # Select k random individuals, return the best
        indices = random.sample(range(len(pop)), k)
        best_idx = indices[0]
        best_fit = fit[best_idx]
        for idx in indices[1:]:
            if fit[idx] < best_fit:
                best_fit = fit[idx]
                best_idx = idx
        return pop[best_idx].tolist()

    # Main GA Loop
    while time.time() < deadline:
        new_pop = []
        
        # Elitism
        # Sort population by fitness
        sorted_indices = np.argsort(fitness)
        for i in range(elite_count):
            new_pop.append(pop[sorted_indices[i]].tolist())
        
        # Generate offspring
        while len(new_pop) < pop_size:
            if time.time() > deadline:
                break
            p1 = tournament_select(pop, fitness)
            p2 = tournament_select(pop, fitness)
            
            # Crossover
            if random.random() < 0.9: # Crossover rate
                c1, c2 = ox_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            # Mutation
            if random.random() < 0.3: # Mutation rate
                c1 = swap_mutation(c1)
            if random.random() < 0.3:
                c2 = swap_mutation(c2)
            
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        
        # Convert to numpy
        new_pop_np = np.array(new_pop, dtype=np.int64)
        
        # Evaluate
        new_fitness = calc_makespans(new_pop_np)
        
        # Update best
        current_best_idx = np.argmin(new_fitness)
        if new_fitness[current_best_idx] < best_ms:
            best_ms = new_fitness[current_best_idx]
            best_seq = new_pop_np[current_best_idx].tolist()
            
        # Replace population
        pop = new_pop_np
        fitness = new_fitness

    return best_seq
# EVOLVE-BLOCK-END
