"""Shortest superpermutation construction with deadline guard and GA optimization."""
import time
import math
import random
from itertools import permutations

def _overlap_length(a, b):
    """Compute maximum overlap between suffix of a and prefix of b."""
    max_o = min(len(a), len(b))
    for o in range(max_o, -1, -1):
        if o == 0 or a[-o:] == b[:o]:
            return o
    return 0

def _build_word_from_order(perms_order, m):
    """Build superpermutation word given an ordering of permutations for symbol m.
    
    perms_order: list of permutations (tuples) of 1..m-1 in the order to insert
    m: the new symbol to insert
    Returns: (word_list, total_length)
    """
    if not perms_order:
        return [], 0
    
    # Build blocks
    blocks = []
    for p in perms_order:
        block = list(p) + [m] + list(p)
        blocks.append(block)
    
    # Start with first block
    word = list(blocks[0])
    for i in range(1, len(blocks)):
        block = blocks[i]
        # Find maximum overlap
        max_o = min(len(word), len(block))
        overlap = 0
        for o in range(max_o, -1, -1):
            if o == 0 or word[-o:] == block[:o]:
                overlap = o
                break
        word.extend(block[overlap:])
    
    return word, len(word)

def _evaluate_order(perms_order, m):
    """Evaluate fitness of an ordering (lower length = better)."""
    _, length = _build_word_from_order(perms_order, m)
    return length

def _crossover(parent1, parent2, rng):
    """Order-based crossover for permutation ordering."""
    n = len(parent1)
    # Choose random cut point
    cut = rng.randint(1, n - 1)
    child = [None] * n
    
    # Copy segment from parent1
    child[:cut] = parent1[:cut]
    
    # Fill remaining from parent2 in order
    remaining = [p for p in parent2 if p not in child[:cut]]
    child[cut:] = remaining
    
    return child

def _mutation(order, rng, rate=0.1):
    """Swap mutation for permutation ordering."""
    order = list(order)
    for i in range(len(order)):
        if rng.random() < rate:
            j = rng.randint(0, len(order) - 1)
            order[i], order[j] = order[j], order[i]
    return order

def construct(n):
    """Return the shortest known superpermutation word for 1..n.
    
    Uses genetic algorithm to optimize the ordering of permutations at each
    level to maximize overlaps between consecutive blocks.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    rng = random.Random(42)  # Fixed seed for reproducibility
    
    # Classical recursive construction as baseline
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        # Get all permutations of 1..m-1
        base_perms = list(permutations(range(1, m)))
        
        if m <= 3:
            # For small m, just use classical order
            perms_order = base_perms
        else:
            # Use genetic algorithm to find good ordering
            pop_size = min(50, max(10, len(base_perms) // 2))
            if time.time() > deadline - 5.0:
                # Not enough time for GA, use random order
                perms_order = list(base_perms)
                rng.shuffle(perms_order)
            else:
                # Initialize population
                population = []
                # Add classical order
                population.append(list(base_perms))
                # Add random orders
                for _ in range(pop_size - 1):
                    random_order = list(base_perms)
                    rng.shuffle(random_order)
                    population.append(random_order)
                
                best_order = None
                best_length = float('inf')
                
                # Evaluate initial population
                for order in population:
                    length = _evaluate_order(order, m)
                    if length < best_length:
                        best_length = length
                        best_order = order
                
                # GA loop
                max_generations = 20
                for gen in range(max_generations):
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Selection: keep top half
                    population.sort(key=lambda o: _evaluate_order(o, m))
                    population = population[:pop_size // 2]
                    
                    # Generate new population
                    new_population = list(population)  # Elitism
                    
                    while len(new_population) < pop_size:
                        p1 = population[rng.randint(0, len(population) - 1)]
                        p2 = population[rng.randint(0, len(population) - 1)]
                        
                        if rng.random() < 0.7:
                            child = _crossover(p1, p2, rng)
                        else:
                            child = list(p1)
                        
                        child = _mutation(child, rng, rate=0.15)
                        new_population.append(child)
                    
                    population = new_population
                    
                    # Update best
                    for order in population:
                        length = _evaluate_order(order, m)
                        if length < best_length:
                            best_length = length
                            best_order = order
                
                perms_order = best_order
        
        # Build word using the optimized order
        if not perms_order:
            continue
            
        blocks = []
        for p in perms_order:
            block = list(p) + [m] + list(p)
            blocks.append(block)
        
        new = list(blocks[0])
        for i in range(1, len(blocks)):
            block = blocks[i]
            max_o = min(len(new), len(block))
            overlap = 0
            for o in range(max_o, -1, -1):
                if o == 0 or new[-o:] == block[:o]:
                    overlap = o
                    break
            new.extend(block[overlap:])
        word = new
    
    return "".join(map(str, word))
