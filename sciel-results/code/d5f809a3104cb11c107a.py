import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    
    # Precompute coverage bitsets
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    balls = [0] * total_words
    for i in range(total_words):
        w = all_words[i]
        mask = 1 << i
        balls[i] = mask
        if radius >= 1:
            for j in range(n):
                for v in range(q):
                    if v == w[j]:
                        continue
                    new_w = w.copy()
                    new_w[j] = v
                    idx = word_to_idx[tuple(new_w)]
                    balls[i] |= (1 << idx)
                    
    full_mask = (1 << total_words) - 1
    
    # Greedy initialization
    def greedy_init():
        uncovered = full_mask
        code = []
        while uncovered:
            best_idx = -1
            max_cov = 0
            for i in range(total_words):
                cov = (balls[i] & uncovered).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_idx = i
            if best_idx == -1:
                break
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
        return code

    best_code = greedy_init()
    best_score = len(best_code)
    
    def get_fitness(code):
        if not code:
            return full_mask.bit_count()
        covered = 0
        for i in code:
            covered |= balls[i]
        return (~covered & full_mask).bit_count()
    
    # Genetic Algorithm
    population_size = 100
    mutation_rate = 0.1
    elite_count = 10
    
    # Initialize population
    population = [best_code[:]]
    for _ in range(population_size - 1):
        pop = greedy_init()
        # Add some noise
        if pop:
            for _ in range(random.randint(0, 5)):
                if random.random() < 0.5 and pop:
                    pop.pop(random.randint(0, len(pop)-1))
                else:
                    pop.append(random.randint(0, total_words-1))
        population.append(pop)
    
    def repair(code):
        # Remove duplicates
        code = list(set(code))
        # Ensure coverage
        uncovered = full_mask
        for i in code:
            uncovered &= ~balls[i]
        
        # Add greedy words to cover remaining
        while uncovered:
            best_idx = -1
            max_cov = 0
            for i in range(total_words):
                if i in code:
                    continue
                cov = (balls[i] & uncovered).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_idx = i
            if best_idx == -1:
                break
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
        return code
    
    def crossover(parent1, parent2):
        # Simple crossover: take random subset from each
        size = max(len(parent1), len(parent2))
        child = []
        for _ in range(size):
            if random.random() < 0.5:
                if parent1:
                    child.append(random.choice(parent1))
            else:
                if parent2:
                    child.append(random.choice(parent2))
        return child
    
    def mutate(code):
        new_code = code[:]
        if random.random() < mutation_rate:
            if new_code:
                idx = random.randint(0, len(new_code)-1)
                if random.random() < 0.5:
                    new_code.pop(idx)
                else:
                    new_code.append(random.randint(0, total_words-1))
        return new_code
    
    generations = 50
    for gen in range(generations):
        if time.time() > deadline:
            break
            
        # Evaluate fitness
        scored_pop = [(get_fitness(ind), ind) for ind in population]
        scored_pop.sort(key=lambda x: x[0])
        
        # Update best
        if scored_pop[0][0] == 0 and len(scored_pop[0][1]) < best_score:
            best_code = scored_pop[0][1][:]
            best_score = len(best_code)
            
        # Selection: Keep elites
        new_population = [ind[:] for _, ind in scored_pop[:elite_count]]
        
        # Generate rest
        while len(new_population) < population_size:
            # Tournament selection
            p1 = random.choice(scored_pop[:population_size//2])[1]
            p2 = random.choice(scored_pop[:population_size//2])[1]
            
            child = crossover(p1, p2)
            child = mutate(child)
            child = repair(child)
            new_population.append(child)
            
        population = new_population
        
    return [all_words[i].tolist() for i in best_code]

