"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(order, processing_times):
    """Calculate the makespan for a given job order."""
    n_machines = processing_times.shape[1]
    prev_completion = np.zeros(n_machines)
    
    for j in order:
        prev_completion[0] += processing_times[j, 0]
        for m in range(1, n_machines):
            prev_completion[m] = max(prev_completion[m-1], prev_completion[m]) + processing_times[j, m]
        
    return prev_completion[-1]

def _neh_heuristic(processing_times):
    """Generate a sequence using NEH heuristic."""
    n_jobs, n_machines = processing_times.shape
    total_times = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_times)
    
    sequence = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        sequence.insert(best_pos, job)
    return sequence

def _order_crossover(parent1, parent2):
    """Order Crossover (OX1) for permutations."""
    n = len(parent1)
    if n < 3:
        return parent1[:]
    
    # Pick two random cut points
    i, j = np.random.choice(n, 2, replace=False)
    if i > j:
        i, j = j, i
        
    # Segment from parent1
    child = [-1] * n
    child[i:j+1] = parent1[i:j+1]
    segment_set = set(parent1[i:j+1])
    
    # Fill remaining positions with elements from parent2 in order
    pos = 0
    for k in range(n):
        if k == i:
            pos = j + 1
            if pos >= n: pos = 0
        elif k == j + 1:
            pos = 0
        if child[k] == -1:
            # Find next element from parent2 not in segment
            while parent2[pos] in segment_set:
                pos += 1
                if pos >= n: pos = 0
            child[k] = parent2[pos]
            pos += 1
            if pos >= n: pos = 0
            
    return child

def _mutate(order, p_mutation):
    """Random insertion mutation."""
    if np.random.random() < p_mutation:
        n = len(order)
        if n < 2:
            return order
        i, j = np.random.choice(n, 2, replace=False)
        elem = order[i]
        del order[i]
        # If i < j, j index shifts left by 1
        if i < j:
            j -= 1
        order.insert(j, elem)
    return order

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    # Time limit check
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Population size
    pop_size = 50
    generations = 100
    
    # Initialize Population
    # 1. One NEH solution
    # 2. One random
    # 3. Rest random
    population = []
    population.append(_neh_heuristic(processing_times))
    population.append(np.random.permutation(n_jobs).tolist())
    
    for _ in range(pop_size - 2):
        population.append(np.random.permutation(n_jobs).tolist())
        
    # Evaluate initial fitness
    fitness = [_makespan(ind, processing_times) for ind in population]
    
    # Track best
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx][:]
    best_makespan = fitness[best_idx]
    
    p_crossover = 0.9
    p_mutation = 0.1
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
            
        new_population = []
        
        # Elitism: Keep the best individual
        new_population.append(best_solution[:])
        
        while len(new_population) < pop_size:
            # Tournament Selection
            # Pick 3 random parents
            parents = np.random.choice(pop_size, 3, replace=False)
            parent_indices = [parents[i] for i in range(3)]
            
            # Select the best of the 3
            best_parent_idx = parent_indices[0]
            best_parent_fit = fitness[best_parent_idx]
            for idx in parent_indices[1:]:
                if fitness[idx] < best_parent_fit:
                    best_parent_fit = fitness[idx]
                    best_parent_idx = idx
            
            parent1 = population[best_parent_idx]
            
            # Select second parent
            other_parent_idx = np.random.choice(pop_size)
            if other_parent_idx == best_parent_idx:
                other_parent_idx = (other_parent_idx + 1) % pop_size
            parent2 = population[other_parent_idx]
            
            # Crossover
            if np.random.random() < p_crossover:
                child = _order_crossover(parent1, parent2)
            else:
                child = parent1[:]
                
            # Mutation
            child = _mutate(child, p_mutation)
            
            # Simple Local Search on Child (Adjacent Swap)
            # Only do a few iterations to save time
            improved = True
            max_ls_iter = 5
            ls_iter = 0
            child_ms = _makespan(child, processing_times)
            
            while improved and ls_iter < max_ls_iter:
                improved = False
                ls_iter += 1
                n = len(child)
                # Random adjacent swap
                if n > 1:
                    i = np.random.randint(0, n-1)
                    child[i], child[i+1] = child[i+1], child[i]
                    new_ms = _makespan(child, processing_times)
                    if new_ms < child_ms:
                        child_ms = new_ms
                        improved = True
                    else:
                        # Revert
                        child[i], child[i+1] = child[i+1], child[i]
                        
            new_population.append(child)
            
        population = new_population
        fitness = [_makespan(ind, processing_times) for ind in population]
        
        # Update best
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_makespan:
            best_makespan = fitness[gen_best_idx]
            best_solution = population[gen_best_idx][:]
            
    return list(best_solution)
# EVOLVE-BLOCK-END
