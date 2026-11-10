"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

N = 256

def _precompute_aps():
    aps = [[] for _ in range(N)]
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(N - 3 * d):
            idxs = (a, a + d, a + 2 * d, a + 3 * d)
            for idx in idxs:
                others = tuple(x for x in idxs if x != idx)
                aps[idx].append(others)
    return aps

_APS = _precompute_aps()

def _count_mono_4ap_numpy(coloring):
    n = len(coloring)
    count = 0
    max_d = (n - 1) // 3
    c = np.array(coloring, dtype=np.int8)
    for d in range(1, max_d + 1):
        s1 = c[:n - 3 * d]
        s2 = c[d:n - 2 * d]
        s3 = c[2 * d:n - d]
        s4 = c[3 * d:n]
        mono = (s1 == s2) & (s2 == s3) & (s3 == s4)
        count += np.sum(mono)
    return int(count)

def _delta_flip(coloring, idx):
    c = coloring
    old_val = c[idx]
    new_val = 1 - old_val
    delta = 0
    for others in _APS[idx]:
        o0, o1, o2 = others
        v0, v1, v2 = c[o0], c[o1], c[o2]
        if v0 == old_val and v1 == old_val and v2 == old_val:
            delta -= 1
        else:
            if v0 == new_val and v1 == new_val and v2 == new_val:
                delta += 1
    return delta

def _local_search_sa(individual, iterations=500, temp_start=1.0, temp_end=0.01):
    current = list(individual)
    current_fit = _count_mono_4ap_numpy(current)
    n = len(current)
    
    for i in range(iterations):
        t = temp_start * (temp_end / temp_start) ** (i / iterations)
        idx = random.randint(0, n - 1)
        delta = _delta_flip(current, idx)
        
        if delta < 0:
            current[idx] = 1 - current[idx]
            current_fit += delta
        else:
            if t > 0:
                prob = np.exp(-delta / t)
                if random.random() < prob:
                    current[idx] = 1 - current[idx]
                    current_fit += delta
                    
    return current, current_fit

def construct():
    n = 256
    pop_size = 40
    generations = 50
    tournament_size = 5
    mutation_rate = 0.05
    
    start_time = time.time()
    time_limit = 115
    
    random.seed(42)
    np.random.seed(42)
    
    population = []
    population.append([(i % 2) for i in range(n)])
    population.append([(i // 2) % 2 for i in range(n)])
    population.append([(i // 3) % 2 for i in range(n)])
    population.append([(i // 4) % 2 for i in range(n)])
    population.append([(i // 5) % 2 for i in range(n)])
    
    while len(population) < pop_size:
        population.append([random.randint(0, 1) for _ in range(n)])
    
    fitnesses = [_count_mono_4ap_numpy(ind) for ind in population]
    
    best_individual = population[fitnesses.index(min(fitnesses))]
    best_fitness = min(fitnesses)
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
            
        new_population = []
        
        sorted_indices = np.argsort(fitnesses)
        for idx in sorted_indices[:5]:
            new_population.append(list(population[idx]))
        
        while len(new_population) < pop_size:
            parent1_idx = random.sample(range(pop_size), tournament_size)
            parent1_idx = min(parent1_idx, key=lambda i: fitnesses[i])
            parent1 = population[parent1_idx]
            
            parent2_idx = random.sample(range(pop_size), tournament_size)
            parent2_idx = min(parent2_idx, key=lambda i: fitnesses[i])
            parent2 = population[parent2_idx]
            
            child = []
            for p1, p2 in zip(parent1, parent2):
                if random.random() < 0.5:
                    child.append(p1)
                else:
                    child.append(p2)
            
            for i in range(n):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]
            
            child, child_fit = _local_search_sa(child, iterations=200, temp_start=1.0, temp_end=0.05)
            
            new_population.append(child)
        
        population = new_population
        fitnesses = [_count_mono_4ap_numpy(ind) for ind in population]
        
        current_best_idx = fitnesses.index(min(fitnesses))
        current_best_fitness = min(fitnesses)
        
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = list(population[current_best_idx])
            
            if gen % 5 == 0:
                best_individual, best_fitness = _local_search_sa(best_individual, iterations=500, temp_start=1.0, temp_end=0.01)
    
    if time.time() - start_time < time_limit - 5:
        best_individual, best_fitness = _local_search_sa(best_individual, iterations=2000, temp_start=1.0, temp_end=0.001)
        
    return best_individual
# EVOLVE-BLOCK-END
