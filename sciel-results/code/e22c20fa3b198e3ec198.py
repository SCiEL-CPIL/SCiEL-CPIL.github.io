"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

N = 256

def _precompute_aps():
    aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(N - 3 * d):
            idxs = (a, a + d, a + 2 * d, a + 3 * d)
            aps.append(idxs)
    return aps

_APS = _precompute_aps()

# Precompute neighbors for fast delta calculation
# _NEIGHBORS[i] is a list of tuples (o1, o2, o3) for each AP containing i
def _precompute_neighbors():
    neighbors = [[] for _ in range(N)]
    for ap in _APS:
        i0, i1, i2, i3 = ap
        neighbors[i0].append((i1, i2, i3))
        neighbors[i1].append((i0, i2, i3))
        neighbors[i2].append((i0, i1, i3))
        neighbors[i3].append((i0, i1, i2))
    return neighbors

_NEIGHBORS = _precompute_neighbors()

def _count_mono_4ap_numpy(coloring):
    c = np.array(coloring, dtype=np.int8)
    count = 0
    n = len(c)
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        s1 = c[:n - 3 * d]
        s2 = c[d:n - 2 * d]
        s3 = c[2 * d:n - d]
        s4 = c[3 * d:n]
        mono = (s1 == s2) & (s2 == s3) & (s3 == s4)
        count += np.sum(mono)
    return int(count)

def _delta_flip(coloring, idx):
    old_val = coloring[idx]
    new_val = 1 - old_val
    delta = 0
    for o1, o2, o3 in _NEIGHBORS[idx]:
        v1, v2, v3 = coloring[o1], coloring[o2], coloring[o3]
        if v1 == old_val and v2 == old_val and v3 == old_val:
            delta -= 1
        elif v1 == new_val and v2 == new_val and v3 == new_val:
            delta += 1
    return delta

def _local_search_hill_climbing(individual, max_iterations=500):
    current = list(individual)
    current_fit = _count_mono_4ap_numpy(current)
    n = len(current)
    
    for _ in range(max_iterations):
        idx = random.randint(0, n - 1)
        delta = _delta_flip(current, idx)
        if delta < 0:
            current[idx] = 1 - current[idx]
            current_fit += delta
        # We only accept improvements (delta < 0). 
        # This is strict hill climbing.
            
    return current, current_fit

def construct():
    n = 256
    pop_size = 50
    generations = 40
    tournament_size = 5
    mutation_rate = 0.08
    
    start_time = time.time()
    time_limit = 115
    
    random.seed(42)
    np.random.seed(42)
    
    population = []
    
    # Diverse seeds
    population.append([random.randint(0, 1) for _ in range(n)])
    
    def thue_morse(k):
        return bin(k).count("1") % 2
    population.append([thue_morse(i) for i in range(n)])
    
    population.append([(i % 2) for i in range(n)])
    population.append([(i // 4) % 2 for i in range(n)])
    population.append([(i // 7) % 2 for i in range(n)])
    
    while len(population) < pop_size:
        population.append([random.randint(0, 1) for _ in range(n)])
    
    fitnesses = [_count_mono_4ap_numpy(ind) for ind in population]
    
    best_individual = population[fitnesses.index(min(fitnesses))]
    best_fitness = min(fitnesses)
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
        
        sorted_indices = np.argsort(fitnesses)
        
        new_population = []
        
        # Elitism
        for idx in sorted_indices[:5]:
            new_population.append(list(population[idx]))
        
        while len(new_population) < pop_size:
            p1_idx = min(random.sample(range(pop_size), tournament_size), key=lambda i: fitnesses[i])
            p2_idx = min(random.sample(range(pop_size), tournament_size), key=lambda i: fitnesses[i])
            
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            child = [p1[i] if random.random() < 0.5 else p2[i] for i in range(n)]
            
            for i in range(n):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]
            
            child, child_fit = _local_search_hill_climbing(child, max_iterations=100)
            
            new_population.append(child)
        
        population = new_population
        fitnesses = [_count_mono_4ap_numpy(ind) for ind in population]
        
        current_best_idx = fitnesses.index(min(fitnesses))
        current_best_fitness = min(fitnesses)
        
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = list(population[current_best_idx])
            
    if time.time() - start_time < time_limit - 5:
        best_individual, best_fitness = _local_search_hill_climbing(best_individual, max_iterations=2000)
        
    return best_individual
# EVOLVE-BLOCK-END
