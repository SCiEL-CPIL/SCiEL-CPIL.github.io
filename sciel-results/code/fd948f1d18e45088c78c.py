"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

def _count_mono4_fast(colours):
    """
    Faster counting using numpy. colours is a numpy array of 0/1 of length 256.
    """
    total = 0
    n = 256
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        idx0 = np.arange(0, max_a)
        idx1 = idx0 + d
        idx2 = idx0 + 2 * d
        idx3 = idx0 + 3 * d
        
        c0 = colours[idx0]
        c1 = colours[idx1]
        c2 = colours[idx2]
        c3 = colours[idx3]
        
        mask = (c0 == c1) & (c1 == c2) & (c2 == c3)
        total += np.sum(mask)
    return total

def _evaluate(colours_list):
    """
    colours_list: list of 256 values in {0,1}
    Returns the count of monochromatic 4-term APs.
    """
    arr = np.array(colours_list, dtype=np.int8)
    return _count_mono4_fast(arr)

def _evaluate_delta(colours, pos, old_val, new_val):
    """
    Compute the change in monochromatic 4-AP count when flipping position pos.
    colours: numpy array of current colours
    pos: index to flip
    old_val: current value at pos
    new_val: new value after flip
    Returns delta (new_score - old_score).
    """
    n = 256
    delta = 0
    # For each d, find all a such that pos is one of a, a+d, a+2d, a+3d
    for d in range(1, n // 3 + 1):
        # pos = a + k*d for k in {0,1,2,3}
        for k in range(4):
            a = pos - k * d
            if a < 0 or a + 3 * d >= n:
                continue
            # The four positions are a, a+d, a+2d, a+3d
            # Check if currently monochromatic
            c0 = colours[a]
            c1 = colours[a + d]
            c2 = colours[a + 2 * d]
            c3 = colours[a + 3 * d]
            if c0 == c1 == c2 == c3:
                # Currently mono, after flip it won't be (since we're flipping one of them)
                delta -= 1
            else:
                # Not currently mono, check if it becomes mono after flip
                # The flipped position will have new_val
                # Check if all four would be new_val
                others = [colours[a], colours[a+d], colours[a+2*d], colours[a+3*d]]
                others[k] = new_val
                if others[0] == others[1] == others[2] == others[3]:
                    delta += 1
    return delta

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    # Generate diverse initial solutions
    def thue_morse(n):
        return bin(n).count('1') % 2
    
    initial_solutions = []
    
    # Thue-Morse
    tm = np.array([thue_morse(i) for i in range(256)], dtype=np.int8)
    initial_solutions.append(tm.copy())
    
    # Alternating patterns
    for block_size in [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]:
        pattern = np.array([(i // block_size) % 2 for i in range(256)], dtype=np.int8)
        initial_solutions.append(pattern.copy())
    
    # Random solutions
    for seed in range(20):
        random.seed(seed)
        rand_sol = np.array([random.randint(0, 1) for _ in range(256)], dtype=np.int8)
        initial_solutions.append(rand_sol)
    
    # Evaluate all initial solutions
    scored_solutions = []
    for sol in initial_solutions:
        score = _evaluate(sol)
        scored_solutions.append((score, sol.copy()))
    
    # Sort by score
    scored_solutions.sort(key=lambda x: x[0])
    
    best_score = scored_solutions[0][0]
    best_colours = scored_solutions[0][1].copy()
    
    # Local search refinement for top candidates
    def local_search(colours, max_iter=50):
        """Iteratively flip bits to reduce the count."""
        current = colours.copy()
        current_score = _evaluate(current)
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline - 10.0:
            improved = False
            iters += 1
            # Try each position
            for i in range(256):
                if time.time() > deadline - 10.0:
                    return current, current_score
                old_val = current[i]
                new_val = 1 - old_val
                delta = _evaluate_delta(current, i, old_val, new_val)
                if delta < 0:
                    current[i] = new_val
                    current_score += delta
                    improved = True
        return current, current_score
    
    # Refine top 5 solutions
    top_k = min(5, len(scored_solutions))
    for i in range(top_k):
        if time.time() > deadline - 30.0:
            break
        refined, refined_score = local_search(scored_solutions[i][1])
        if refined_score < best_score:
            best_score = refined_score
            best_colours = refined.copy()
    
    # Genetic algorithm
    population_size = 30
    # Initialize population with best solutions and random perturbations
    population = []
    for i in range(population_size):
        if i < top_k:
            pop = scored_solutions[i][1].copy()
        else:
            pop = best_colours.copy()
            # Randomly flip some bits
            num_flips = random.randint(1, 10)
            flip_positions = random.sample(range(256), num_flips)
            for fp in flip_positions:
                pop[fp] = 1 - pop[fp]
        pop_score = _evaluate(pop)
        population.append((pop_score, pop.copy()))
    
    # Sort population
    population.sort(key=lambda x: x[0])
    
    # GA loop
    generation = 0
    while time.time() < deadline - 20.0:
        generation += 1
        
        # Selection: keep top half
        half_size = population_size // 2
        parents = [pop[1] for pop in population[:half_size]]
        
        # Crossover and mutation to create new offspring
        offspring = []
        for i in range(population_size - half_size):
            # Pick two parents (tournament selection)
            idx1 = random.randint(0, half_size - 1)
            idx2 = random.randint(0, half_size - 1)
            p1 = parents[idx1]
            p2 = parents[idx2]
            
            # Uniform crossover
            mask = np.random.randint(0, 2, size=256)
            child = np.where(mask == 1, p1, p2).astype(np.int8)
            
            # Mutation
            num_mutations = random.randint(0, 5)
            if num_mutations > 0:
                mut_positions = random.sample(range(256), num_mutations)
                for mp in mut_positions:
                    child[mp] = 1 - child[mp]
            
            child_score = _evaluate(child)
            offspring.append((child_score, child.copy()))
        
        # Combine parents and offspring
        population = population[:half_size] + offspring
        population.sort(key=lambda x: x[0])
        
        # Update best
        if population[0][0] < best_score:
            best_score = population[0][0]
            best_colours = population[0][1].copy()
        
        # Local search on the best solution in population
        if time.time() < deadline - 15.0:
            refined, refined_score = local_search(population[0][1], max_iter=20)
            if refined_score < best_score:
                best_score = refined_score
                best_colours = refined.copy()
                population[0] = (refined_score, refined.copy())
                population.sort(key=lambda x: x[0])
    
    # Final local search polish
    if time.time() < deadline - 5.0:
        refined, refined_score = local_search(best_colours, max_iter=100)
        if refined_score < best_score:
            best_score = refined_score
            best_colours = refined.copy()
    
    return best_colours.tolist()
# EVOLVE-BLOCK-END
