import time
import random
import numpy as np

def _score_set(s):
    """
    Score a set of integers (1..512) by counting distinct d > 0 
    such that a, a+d, a+2d are all in the set for some a.
    """
    mask = 0
    for x in s:
        mask |= (1 << x)
    
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _greedy_init():
    """
    Greedy construction: start with empty set, add elements that maximize score increment.
    """
    s = set()
    mask = 0
    best_score = 0
    candidates = list(range(1, 513))
    random.shuffle(candidates)
    
    for _ in range(32):
        best_elem = None
        best_new_score = best_score
        for c in candidates:
            if c in s:
                continue
            new_mask = mask | (1 << c)
            new_score = 0
            for d in range(1, 256):
                if (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0:
                    new_score += 1
            if new_score > best_new_score:
                best_new_score = new_score
                best_elem = c
        if best_elem is not None:
            s.add(best_elem)
            mask |= (1 << best_elem)
            best_score = best_new_score
        else:
            if candidates:
                c = candidates.pop()
                s.add(c)
                mask |= (1 << c)
                best_score = _score_set(s)
    return s, best_score

def _genetic_algorithm(s, initial_score, deadline):
    """
    Genetic algorithm to improve the set.
    """
    population = [set(s)]
    population_scores = [initial_score]
    
    # Add some random individuals to the population
    for _ in range(10):
        random_set = set(random.sample(range(1, 513), 32))
        population.append(random_set)
        population_scores.append(_score_set(random_set))
    
    best_idx = max(range(len(population)), key=lambda i: population_scores[i])
    best = set(population[best_idx])
    best_score = population_scores[best_idx]
    
    while time.time() < deadline:
        # Selection: keep the top half of the population
        population_with_scores = list(zip(population, population_scores))
        population_with_scores.sort(key=lambda x: x[1], reverse=True)
        population = [p for p, _ in population_with_scores[:len(population)//2]]
        population_scores = [s for _, s in population_with_scores[:len(population)//2]]
        
        # Crossover and mutation
        new_population = []
        new_scores = []
        
        while len(new_population) < len(population):
            # Select two parents
            p1, p2 = random.sample(population, 2)
            
            # Crossover: randomly select elements from p1 or p2
            child = set()
            for elem in p1:
                if random.random() < 0.5:
                    child.add(elem)
            for elem in p2:
                if random.random() < 0.5:
                    child.add(elem)
            
            # Ensure the child has exactly 32 elements
            while len(child) < 32:
                child.add(random.randint(1, 512))
            while len(child) > 32:
                child.remove(random.choice(list(child)))
            
            # Mutation: randomly swap elements
            if random.random() < 0.5:
                to_remove = random.choice(list(child))
                to_add = random.randint(1, 512)
                if to_add not in child:
                    child.remove(to_remove)
                    child.add(to_add)
            
            new_population.append(child)
            new_scores.append(_score_set(child))
        
        # Update population
        population = new_population
        population_scores = new_scores
        
        # Update best
        best_idx = max(range(len(population)), key=lambda i: population_scores[i])
        if population_scores[best_idx] > best_score:
            best = set(population[best_idx])
            best_score = population_scores[best_idx]
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe within 150s limit
    
    # Seed random for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Greedy initialization
    s, score = _greedy_init()
    best_s = set(s)
    best_score = score
    
    # Phase 2: Genetic algorithm
    best_s, best_score = _genetic_algorithm(best_s, best_score, deadline)
    
    return sorted(list(best_s))
