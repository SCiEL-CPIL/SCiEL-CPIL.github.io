"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(order, processing_times):
    """
    Calculate the makespan for a given job order in a flow shop.
    order: list of job indices
    processing_times: (n_jobs, n_machines) array
    Returns the makespan (total completion time on the last machine).
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    
    # C[m] is the completion time of the current job on machine m
    prev_completion = np.zeros(n_machines)
    
    for j in order:
        curr_completion_0 = prev_completion[0] + processing_times[j, 0]
        
        if n_machines == 1:
            prev_completion[0] = curr_completion_0
            continue
            
        curr_completion = np.empty(n_machines)
        curr_completion[0] = curr_completion_0
        
        for m in range(1, n_machines):
            curr_completion[m] = max(curr_completion[m-1], prev_completion[m]) + processing_times[j, m]
            
        prev_completion = curr_completion
        
    return prev_completion[-1]


def _neh_heuristic(processing_times):
    """
    Generate an initial permutation using the NEH (Nawaz-Enscore-Ham) heuristic.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Calculate sum of processing times for each job
    job_sums = processing_times.sum(axis=1)
    
    # Sort jobs in descending order of their sums
    sorted_jobs = np.argsort(-job_sums)
    
    current_sequence = []
    
    for job in sorted_jobs:
        # Find the best position to insert the current job
        best_pos = 0
        best_ms = float('inf')
        
        # Try inserting at every possible position (0 to len(current_sequence))
        for pos in range(len(current_sequence) + 1):
            new_seq = current_sequence[:pos] + [job] + current_sequence[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        # Insert the job at the best position
        current_sequence.insert(best_pos, job)
        
    return current_sequence


def _oropt_makespan_fast(order, processing_times, block_start, block_size, new_pos):
    """
    Fast makespan evaluation for an Or-opt move:
    - Remove block of length block_size starting at block_start
    - Insert it at position new_pos in the resulting list
    Uses incremental computation to avoid full O(n*m) recalculation.
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    
    # Build the new order
    block = order[block_start:block_start + block_size]
    remaining = order[:block_start] + order[block_start + block_size:]
    
    # new_pos is the insertion index in 'remaining'
    # Clamp new_pos to valid range
    if new_pos < 0:
        new_pos = 0
    if new_pos > len(remaining):
        new_pos = len(remaining)
    
    new_order = remaining[:new_pos] + block + remaining[new_pos:]
    
    # Compute makespan for new_order
    prev_completion = np.zeros(n_machines)
    
    for j in new_order:
        curr_completion_0 = prev_completion[0] + processing_times[j, 0]
        
        if n_machines == 1:
            prev_completion[0] = curr_completion_0
            continue
            
        curr_completion = np.empty(n_machines)
        curr_completion[0] = curr_completion_0
        
        for m in range(1, n_machines):
            curr_completion[m] = max(curr_completion[m-1], prev_completion[m]) + processing_times[j, m]
            
        prev_completion = curr_completion
        
    return prev_completion[-1]


def _local_search(order, processing_times, deadline):
    """
    Perform local search using Or-opt moves.
    """
    current_order = order[:]
    current_makespan = _makespan(current_order, processing_times)
    best_order = current_order[:]
    best_makespan = current_makespan
    
    improved = True
    no_improve_count = 0
    max_no_improve = 3
    
    while improved and time.time() < deadline:
        improved = False
        n = len(current_order)
        
        for block_size in [1, 2, 3]:
            if time.time() > deadline:
                return best_order, best_makespan
            
            for i in range(n - block_size + 1):
                if time.time() > deadline:
                    return best_order, best_makespan
                
                block = current_order[i:i+block_size]
                
                best_pos = -1
                best_ms = current_makespan
                
                # Evaluate all insertion positions
                for j in range(n - block_size + 1):
                    if j == i:
                        continue
                    
                    ms = _oropt_makespan_fast(current_order, processing_times, i, block_size, j)
                    
                    if ms < best_ms:
                        best_ms = ms
                        best_pos = j
                        
                if best_pos != -1:
                    # Apply the best move found for this block_size and i
                    remaining = current_order[:i] + current_order[i+block_size:]
                    current_order = remaining[:best_pos] + block + remaining[best_pos:]
                    current_makespan = best_ms
                    
                    if current_makespan < best_makespan:
                        best_makespan = current_makespan
                        best_order = current_order[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    return best_order, best_makespan


def _order_based_crossover(parent1, parent2):
    """
    Order-based crossover (OX1) for permutations.
    """
    n = len(parent1)
    if n == 0:
        return [], []
    
    # Choose two random cut points
    i, j = sorted(random.sample(range(n), 2))
    
    # Select segment from parent1
    segment = parent1[i:j+1]
    segment_set = set(segment)
    
    # Fill rest from parent2
    child1 = [0] * n
    child1[i:j+1] = segment
    fill_positions = list(range(0, i)) + list(range(j+1, n))
    k = 0
    for pos in fill_positions:
        while k < n and parent2[k] in segment_set:
            k += 1
        if k < n:
            child1[pos] = parent2[k]
            k += 1
    
    # Similarly for child2, swapping parents
    segment2 = parent2[i:j+1]
    segment_set2 = set(segment2)
    child2 = [0] * n
    child2[i:j+1] = segment2
    k = 0
    for pos in fill_positions:
        while k < n and parent1[k] in segment_set2:
            k += 1
        if k < n:
            child2[pos] = parent1[k]
            k += 1
    
    return child1, child2


def _mutation(order, rate=0.1):
    """
    Random swap mutation.
    """
    n = len(order)
    if n < 2:
        return order[:]
    
    mutated = order[:]
    for i in range(n):
        if random.random() < rate:
            j = random.randint(0, n-1)
            mutated[i], mutated[j] = mutated[j], mutated[i]
    
    return mutated


def _genetic_algorithm(processing_times, deadline, pop_size=50):
    """
    Genetic algorithm with order-based crossover and mutation.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return [], 0.0
    if n_jobs == 1:
        return [0], _makespan([0], processing_times)
    
    # Initialize population with NEH and random permutations
    population = []
    nehe_order = _neh_heuristic(processing_times)
    population.append(nehe_order)
    
    for _ in range(pop_size - 1):
        rand_order = list(range(n_jobs))
        random.shuffle(rand_order)
        population.append(rand_order)
    
    # Evaluate initial population
    fitnesses = [_makespan(order, processing_times) for order in population]
    
    best_order = population[fitnesses.index(min(fitnesses))]
    best_makespan = min(fitnesses)
    
    # Main GA loop
    while time.time() < deadline:
        # Selection (tournament selection)
        new_population = []
        for _ in range(pop_size):
            # Tournament of size 3
            tournament = random.sample(range(pop_size), 3)
            winner = min(tournament, key=lambda x: fitnesses[x])
            new_population.append(population[winner][:])
        
        # Crossover and mutation
        offspring = []
        for i in range(0, pop_size, 2):
            if i + 1 >= pop_size:
                break
            p1 = new_population[i]
            p2 = new_population[i+1]
            
            c1, c2 = _order_based_crossover(p1, p2)
            c1 = _mutation(c1, rate=0.1)
            c2 = _mutation(c2, rate=0.1)
            
            offspring.append(c1)
            offspring.append(c2)
        
        # Replace population with offspring (elitism: keep best individual)
        population = offspring
        fitnesses = [_makespan(order, processing_times) for order in population]
        
        # Elitism: replace worst individual with best from previous generation
        worst_idx = fitnesses.index(max(fitnesses))
        if best_makespan < fitnesses[worst_idx]:
            population[worst_idx] = best_order[:]
            fitnesses[worst_idx] = best_makespan
        
        # Update best
        current_best_idx = fitnesses.index(min(fitnesses))
        if fitnesses[current_best_idx] < best_makespan:
            best_makespan = fitnesses[current_best_idx]
            best_order = population[current_best_idx][:]
    
    return best_order, best_makespan


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Generate initial solution using NEH heuristic
    current_order = _neh_heuristic(processing_times)
    current_makespan = _makespan(current_order, processing_times)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # 2. Local search with Or-opt moves
    if time.time() < deadline - 30.0:
        ls_order, ls_makespan = _local_search(current_order, processing_times, deadline - 30.0)
        if ls_makespan < best_makespan:
            best_makespan = ls_makespan
            best_order = ls_order[:]
    
    # 3. Genetic algorithm to explore broader solution space
    if time.time() < deadline - 5.0:
        ga_order, ga_makespan = _genetic_algorithm(processing_times, deadline - 5.0, pop_size=30)
        if ga_makespan < best_makespan:
            best_makespan = ga_makespan
            best_order = ga_order[:]
    
    # 4. Final polish: local search on the best solution found
    if time.time() < deadline - 2.0:
        polish_order, polish_makespan = _local_search(best_order, processing_times, deadline - 2.0)
        if polish_makespan < best_makespan:
            best_makespan = polish_makespan
            best_order = polish_order[:]
    
    return best_order

# EVOLVE-BLOCK-END
