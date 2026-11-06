import numpy as np
from random import seed as rand_seed, choice
import time

def construct():
    N = 24
    # 24th roots of unity
    k = np.arange(N)
    theta = 2 * np.pi * k / N
    roots = np.exp(1j * theta)
    
    # Precompute: for each coefficient index j, the vector of roots^j
    # This is an N x N matrix where M[i,j] = roots[i]^j
    M = np.zeros((N, N), dtype=complex)
    for j in range(N):
        M[:, j] = roots ** j
    
    # Target value is the exhaustive optimum: 1.2583758 * sqrt(24)
    # But we minimize the actual max |P(z)| / sqrt(N)
    # So target raw = 1.2583758 * sqrt(24) ≈ 1.2583758 * 4.898979 ≈ 6.1646
    target_raw = 1.2583758 * np.sqrt(N)
    
    # Fitness: minimize max |P(roots)| 
    # For speed, use a proxy: sum of squares of deviations above target, or just max
    # Use max |P(roots)| as fitness (lower is better)
    
    def evaluate(coeffs):
        # coeffs: array of +1/-1, length N
        # P(roots[i]) = sum_j coeffs[j] * roots[i]^j = M @ coeffs
        vals = M @ coeffs
        return np.max(np.abs(vals))
    
    # Initialize with Golay-Rudin-Shapiro as a strong starting point
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    grs = np.array(p[:N], dtype=float)
    
    # Population size
    pop_size = 400
    # Initialize population: one GRS, rest random
    population = np.zeros((pop_size, N), dtype=float)
    population[0] = grs
    for i in range(1, pop_size):
        population[i] = 2 * np.random.choice([0, 1], size=N) - 1  # +1/-1
    
    # Evaluate initial population
    fitness = np.array([evaluate(population[i]) for i in range(pop_size)])
    
    # GA parameters
    generations = 300
    tournament_size = 5
    mutation_rate = 0.15
    crossover_rate = 0.8
    elite_size = 10
    
    best_ever_fitness = np.min(fitness)
    best_ever_idx = np.argmin(fitness)
    best_ever_coeffs = population[best_ever_idx].copy()
    
    start_time = time.time()
    time_limit = 100.0  # leave margin for 120s limit
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
        
        # Sort by fitness
        sorted_indices = np.argsort(fitness)
        sorted_pop = population[sorted_indices]
        sorted_fitness = fitness[sorted_indices]
        
        # Update best
        if sorted_fitness[0] < best_ever_fitness:
            best_ever_fitness = sorted_fitness[0]
            best_ever_coeffs = sorted_pop[0].copy()
        
        # Early stopping if we hit the target
        if best_ever_fitness <= target_raw + 1e-6:
            break
        
        # Create new population
        new_population = np.zeros((pop_size, N), dtype=float)
        
        # Elitism
        new_population[:elite_size] = sorted_pop[:elite_size]
        
        idx = elite_size
        while idx < pop_size:
            # Tournament selection
            def tournament_select():
                candidates = np.random.choice(pop_size, size=tournament_size, replace=False)
                best_cand = candidates[np.argmin(fitness[candidates])]
                return population[best_cand]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover
            if np.random.random() < crossover_rate:
                child = np.zeros(N, dtype=float)
                for j in range(N):
                    if np.random.random() < 0.5:
                        child[j] = parent1[j]
                    else:
                        child[j] = parent2[j]
            else:
                child = parent1.copy()
            
            # Mutation: flip random coefficients
            for j in range(N):
                if np.random.random() < mutation_rate:
                    child[j] = -child[j]
            
            new_population[idx] = child
            idx += 1
        
        population = new_population
        fitness = np.array([evaluate(population[i]) for i in range(pop_size)])
    
    return [int(c) for c in best_ever_coeffs]
