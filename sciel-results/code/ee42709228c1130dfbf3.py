"""Memetic Genetic Algorithm for flow-shop permutation construction.

This strategy uses a population-based approach:
1. Initialize with diverse NEH variants (different sorting keys).
2. Evolve via Order Crossover (OX) to mix good subsequences.
3. Apply Or-opt local search (moving blocks of 1, 2, 3 jobs) to refine offspring.
4. Selection based on makespan with elitism.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    """
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0
    
    c_machines = [0.0] * n_machines
    
    for j in order:
        row = P[j]
        c_curr = 0.0
        for m in range(n_machines):
            if c_machines[m] > c_curr:
                c_curr = c_machines[m]
            c_curr += row[m]
            c_machines[m] = c_curr
            
    return c_machines[-1]

def _neh_variant(P, key_func):
    """
    NEH heuristic with a specific sorting key.
    key_func: function taking P (n_jobs, n_machines) and returning array of length n_jobs.
    """
    n_jobs, n_machines = P.shape
    keys = key_func(P)
    # Sort indices by key descending
    order_indices = np.argsort(-keys)
    
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

def _or_opt_local_search(order, P, max_passes=3, time_limit=None, start_time=None):
    """
    Local search using Or-opt moves (moving blocks of size 1, 2, 3).
    First improvement strategy.
    """
    n_jobs = len(order)
    n_machines = P.shape[1]
    best_order = order[:]
    best_makespan = _makespan(best_order, P)
    
    block_sizes = [1, 2, 3]
    
    for _pass in range(max_passes):
        if time_limit and start_time and (time.time() - start_time) > time_limit:
            break
            
        improved = False
        # Randomize block sizes order to avoid bias
        random.shuffle(block_sizes)
        
        for b_size in block_sizes:
            if b_size > n_jobs:
                continue
                
            for i in range(n_jobs - b_size + 1):
                if time_limit and start_time and (time.time() - start_time) > time_limit:
                    return best_order, best_makespan
                    
                # Extract block
                block = best_order[i:i+b_size]
                
                # Try inserting at all other positions
                # Skip positions that result in the same order
                # Valid insert positions: 0 to n_jobs - b_size
                # But we must avoid re-inserting into the same spot or adjacent spots that don't change order?
                # Actually, just try all valid spots j such that j != i and j+b_size != i (not overlapping in a way that keeps same relative order? No, OX/Or-opt just moves block).
                # If we remove block at i..i+b_size-1, the list shrinks by b_size.
                # We can insert at any position j in the new list from 0 to len(new_list)-b_size.
                # If j == i, it's the same.
                
                for j in range(n_jobs - b_size + 1):
                    if j == i:
                        continue
                    
                    # Construct new order
                    # Remove block
                    new_order = best_order[:i] + best_order[i+b_size:]
                    
                    # Determine insertion index in new_order
                    # If j > i, the index shifts because we removed b_size elements before j?
                    # No, we are iterating j as the target position in the FINAL sequence of length n_jobs.
                    # Let's map target position j in final sequence to insertion index in new_order.
                    # If j < i: insert at j.
                    # If j > i: insert at j - b_size.
                    
                    if j < i:
                        ins_idx = j
                    else:
                        ins_idx = j - b_size
                        
                    # Bounds check for ins_idx
                    if ins_idx < 0 or ins_idx > len(new_order) - b_size:
                        continue
                        
                    new_order = new_order[:ins_idx] + block + new_order[ins_idx:]
                    
                    ms = _makespan(new_order, P)
                    
                    if ms < best_makespan:
                        best_makespan = ms
                        best_order = new_order
                        improved = True
                        # Restart block size loop? Or just break to next i?
                        # First improvement: break inner j loop, continue to next i?
                        # To be aggressive, break j loop and restart i loop?
                        # Let's just break j loop and let i loop continue.
                        break
                        
                if improved:
                    # If we improved, we should probably restart the i loop for this block size?
                    # Or just continue. Continuing is faster.
                    # But the positions have shifted.
                    # Let's break out of i loop to restart scanning from 0 for this block size?
                    # That's safer for finding local optimum.
                    break
                    
            if improved:
                # Restart block sizes?
                # Let's break out of block_sizes loop and start a new pass.
                break
                
    return best_order, best_makespan

def _order_crossover(parent1, parent2):
    """
    Order Crossover (OX).
    1. Pick two random cut points.
    2. Copy the segment between cuts from parent1 to child.
    3. Fill remaining positions with jobs from parent2, starting after the second cut point, wrapping around.
    """
    n = len(parent1)
    if n < 2:
        return parent1[:]
        
    # Pick two distinct cut points
    i = random.randint(0, n - 1)
    j = random.randint(0, n - 1)
    if i == j:
        i += 1
        if i >= n: i = 0
    if i > j:
        i, j = j, i
        
    child = [None] * n
    # Copy segment
    for k in range(i, j + 1):
        child[k] = parent1[k]
        
    # Fill rest
    # The order of jobs in parent2, starting from j+1
    fill_jobs = parent2[j+1:] + parent2[:j+1]
    
    # We need to place these jobs in the empty slots of child, in the order they appear in fill_jobs.
    # We iterate through child positions, and pick the next available job from fill_jobs.
    
    fill_idx = 0
    for k in range(n):
        if child[k] is None:
            # Find next job from fill_jobs that is not already in child (which it won't be, as sets are disjoint)
            # Wait, fill_jobs is a permutation of the remaining jobs.
            # So we just take fill_jobs[fill_idx]
            child[k] = fill_jobs[fill_idx]
            fill_idx += 1
            
    return child

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using a Memetic Genetic Algorithm.
    """
    P = np.asarray(processing_times, dtype=float)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    # Define NEH variants for initialization
    # 1. Sum of processing times
    # 2. First machine processing time
    # 3. Last machine processing time
    # 4. Average processing time (same as sum for ranking)
    # 5. Random permutation
    
    def key_sum(P):
        return P.sum(axis=1)
        
    def key_first(P):
        return P[:, 0]
        
    def key_last(P):
        return P[:, -1]
        
    def key_random(P):
        return np.random.rand(P.shape[0])
        
    # Population size
    # Small enough to allow many generations within time limit.
    # Let's start with 50.
    pop_size = 50
    
    # Initialize population
    population = []
    fitnesses = []
    
    # Ensure diversity in initial population
    keys = [key_sum, key_first, key_last, key_random, key_random, key_random]
    
    for i in range(pop_size):
        if i < len(keys):
            key = keys[i]
        else:
            key = random.choice(keys)
            
        # Add some randomness to the key to avoid exact duplicates in NEH results if keys are tied
        # NEH is deterministic given the sorted order. Ties in keys might lead to same order.
        # We can perturb keys slightly.
        base_keys = key(P)
        # Perturb
        perturbed_keys = base_keys + np.random.rand(n_jobs) * 1e-6
        
        order = _neh_variant(P, lambda p: perturbed_keys)
        
        # Apply a bit of local search to initial population?
        # It's expensive. Maybe just for the top few?
        # Let's just evaluate first.
        ms = _makespan(order, P)
        population.append(order)
        fitnesses.append(ms)
        
    # Track best
    best_idx = np.argmin(fitnesses)
    best_order = population[best_idx][:]
    best_makespan = fitnesses[best_idx]
    
    # Evolution loop
    generation = 0
    no_improve_count = 0
    max_no_improve = 20 # Generations without improvement to trigger restart/diversification
    
    while (time.time() - start_time) < time_limit:
        generation += 1
        
        # Check time before expensive operations
        if (time.time() - start_time) > time_limit - 5:
            break
            
        # Sort population by fitness
        sorted_indices = np.argsort(fitnesses)
        population = [population[i] for i in sorted_indices]
        fitnesses = [fitnesses[i] for i in sorted_indices]
        
        # Elitism: Keep top 2
        elite1 = population[0][:]
        elite2 = population[1][:] if pop_size > 1 else population[0][:]
        
        # Generate offspring
        new_population = [elite1, elite2]
        new_fitnesses = [fitnesses[0], fitnesses[1]]
        
        # Fill rest of population
        while len(new_population) < pop_size:
            # Selection: Tournament selection
            def tournament_select(pop, fit, k=3):
                candidates = random.sample(range(len(pop)), k)
                best_cand = min(candidates, key=lambda i: fit[i])
                return pop[best_cand][:]
                
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            
            # Crossover
            child = _order_crossover(p1, p2)
            
            # Mutation: Random swap or insertion?
            # Let's do a random insertion mutation with low probability
            if random.random() < 0.3:
                i = random.randint(0, n_jobs - 1)
                j = random.randint(0, n_jobs - 1)
                job = child[i]
                child.pop(i)
                child.insert(j, job)
                
            # Local Search on child
            # Limit local search time per child?
            # Or just do a few passes.
            # Let's do 1 pass of Or-opt for speed.
            child, child_ms = _or_opt_local_search(child, P, max_passes=1, time_limit=time_limit, start_time=start_time)
            
            new_population.append(child)
            new_fitnesses.append(child_ms)
            
            # Check time
            if (time.time() - start_time) > time_limit:
                break
                
        # Replace population
        population = new_population
        fitnesses = new_fitnesses
        
        # Update best
        current_best_idx = np.argmin(fitnesses)
        if fitnesses[current_best_idx] < best_makespan:
            best_makespan = fitnesses[current_best_idx]
            best_order = population[current_best_idx][:]
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Diversification: If stuck, inject random solutions or re-initialize part of population
        if no_improve_count > max_no_improve:
            # Replace worst 50% with new NEH variants
            half = pop_size // 2
            for i in range(half, pop_size):
                key = random.choice([key_sum, key_first, key_last, key_random])
                base_keys = key(P)
                perturbed_keys = base_keys + np.random.rand(n_jobs) * 1e-6
                new_order = _neh_variant(P, lambda p: perturbed_keys)
                new_ms = _makespan(new_order, P)
                population[i] = new_order
                fitnesses[i] = new_ms
            no_improve_count = 0
            
    return best_order
# EVOLVE-BLOCK-END
