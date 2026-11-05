import numpy as np
import time

def calc_max_discrepancy_vec(signs):
    n = len(signs)
    max_abs = 0
    for d in range(1, n):
        for r in range(d):
            ap = signs[r::d]
            csum = np.cumsum(ap)
            m = np.max(np.abs(csum))
            if m > max_abs:
                max_abs = m
    return max_abs

def local_search(signs, limit_time):
    n = len(signs)
    current_max = calc_max_discrepancy_vec(signs)
    improved = True
    while improved and time.time() < limit_time:
        improved = False
        indices = np.random.permutation(n)
        for i in indices:
            if time.time() > limit_time:
                break
            old_sign = signs[i]
            signs[i] = -old_sign
            new_max = calc_max_discrepancy_vec(signs)
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                signs[i] = old_sign
    return signs, current_max

def construct():
    n = 256
    start_time = time.time()
    limit_time = start_time + 58.0
    
    pop_size = 32
    # Thue-Morse initialization
    tm = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    
    population = [tm.copy()]
    for _ in range(pop_size - 1):
        population.append(np.random.choice([-1, 1], size=n).astype(np.int8))
        
    fitnesses = [calc_max_discrepancy_vec(ind) for ind in population]
    
    best_idx = np.argmin(fitnesses)
    best_signs = population[best_idx].copy()
    best_score = fitnesses[best_idx]
    
    gen = 0
    while time.time() < limit_time:
        # Local search on the current best
        # Use a short time budget for LS to allow more GA generations
        ls_limit = min(time.time() + 0.5, limit_time)
        ls_signs, ls_score = local_search(population[best_idx].copy(), ls_limit)
        if ls_score < best_score:
            best_score = ls_score
            best_signs = ls_signs
            population[best_idx] = ls_signs
            fitnesses[best_idx] = ls_score
            
        # GA Step
        new_population = []
        # Elitism
        indices = np.argsort(fitnesses)
        for i in range(2):
            new_population.append(population[indices[i]].copy())
            
        while len(new_population) < pop_size:
            # Tournament selection
            a = np.random.randint(0, pop_size)
            b = np.random.randint(0, pop_size)
            p1 = population[a] if fitnesses[a] < fitnesses[b] else population[b]
            
            c = np.random.randint(0, pop_size)
            d = np.random.randint(0, pop_size)
            p2 = population[c] if fitnesses[c] < fitnesses[d] else population[d]
            
            # Uniform Crossover
            mask = np.random.randint(0, 2, size=n)
            child = np.where(mask == 1, p1, p2).astype(np.int8)
            
            # Mutation
            mut_mask = np.random.rand(n) < 0.05
            child[mut_mask] = -child[mut_mask]
            
            new_population.append(child)
            
        population = new_population
        fitnesses = [calc_max_discrepancy_vec(ind) for ind in population]
        
        new_best_idx = np.argmin(fitnesses)
        if fitnesses[new_best_idx] < best_score:
            best_score = fitnesses[new_best_idx]
            best_signs = population[new_best_idx].copy()
            
        best_idx = new_best_idx
        gen += 1
        
    return best_signs.tolist()
