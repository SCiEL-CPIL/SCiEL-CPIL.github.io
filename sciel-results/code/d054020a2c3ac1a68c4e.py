import numpy as np
from itertools import combinations
import random
import time

def construct():
    target_n = 34
    all_triples = list(combinations(range(target_n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    four_sets = list(combinations(range(target_n), 4))
    five_sets = list(combinations(range(target_n), 5))
    
    def fitness(colors):
        """Higher is better. Returns (n - violations) * large_weight - violations"""
        red_k4 = 0
        for fs in four_sets:
            if all(colors[triple_to_idx[t]] == 0 for t in combinations(fs, 3)):
                red_k4 += 1
        blue_k5 = 0
        for fs in five_sets:
            if all(colors[triple_to_idx[t]] == 1 for t in combinations(fs, 3)):
                blue_k5 += 1
        violations = red_k4 + blue_k5
        return -violations  # Maximize this
    
    def genetic_algorithm(pop_size=50, generations=200, mutation_rate=0.01, crossover_rate=0.8):
        # Initialize population
        population = [np.random.randint(0, 2, num_triples).tolist() for _ in range(pop_size)]
        best_individual = max(population, key=fitness)
        best_fitness = fitness(best_individual)
        
        for gen in range(generations):
            # Evaluate fitness
            fitnesses = [fitness(ind) for ind in population]
            
            # Find best
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
            
            if best_fitness == 0:
                break
            
            # Create next generation
            next_gen = []
            
            # Elitism: keep best individual
            next_gen.append(best_individual.copy())
            
            while len(next_gen) < pop_size:
                # Selection
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                
                # Crossover
                if random.random() < crossover_rate:
                    child = parent1.copy()
                    for i in range(num_triples):
                        if random.random() < 0.5:
                            child[i] = parent2[i]
                else:
                    child = parent1.copy()
                
                # Mutation
                for i in range(num_triples):
                    if random.random() < mutation_rate:
                        child[i] = 1 - child[i]
                
                next_gen.append(child)
            
            population = next_gen
        
        return best_individual, best_fitness
    
    start_time = time.time()
    best_solution = None
    best_fitness_val = -float('inf')
    
    for run in range(5):
        if time.time() - start_time > 100:
            break
        
        ind, fit = genetic_algorithm(pop_size=40, generations=150, mutation_rate=0.005, crossover_rate=0.85)
        
        if fit > best_fitness_val:
            best_fitness_val = fit
            best_solution = ind
        
        if best_fitness_val == 0:
            break
    
    if best_solution is not None and best_fitness_val == 0:
        return {'n': target_n, 'colors': best_solution}
    
    # Fallback
    n_trivial = 4
    colors_trivial = [1] * len(list(combinations(range(n_trivial), 3)))
    return {'n': n_trivial, 'colors': colors_trivial}
