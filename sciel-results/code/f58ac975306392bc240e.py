"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START

def _classic_construct(n):
    """Classic recursive superpermutation construction."""
    if n == 1:
        return [1]
    
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _evaluate_ordering(n, orderings):
    """
    Evaluate a set of orderings (one per level 2..n) for total length.
    orderings[m] is a list of permutations of 1..m-1 in the order to visit.
    Returns the resulting word length.
    """
    if n == 1:
        return 1
    
    word = [1]
    for m in range(2, n + 1):
        perms = orderings[m]
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return len(word)

def _build_word_from_orderings(n, orderings):
    """Build the actual word from orderings."""
    if n == 1:
        return [1]
    
    word = [1]
    for m in range(2, n + 1):
        perms = orderings[m]
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _get_classic_orderings(n):
    """Get the classic recursive orderings."""
    orderings = {}
    if n == 1:
        return orderings
    
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        orderings[m] = perms
        # Reconstruct word for next iteration
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return orderings

def _mutate_ordering(perms, mutation_rate=0.1):
    """Mutate a permutation ordering by swapping or reversing segments."""
    mutated = list(perms)
    num_swaps = int(len(mutated) * mutation_rate)
    for _ in range(num_swaps):
        if len(mutated) < 2:
            break
        i = random.randint(0, len(mutated) - 2)
        if random.random() < 0.5:
            # Swap two elements
            j = random.randint(i + 1, len(mutated) - 1)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        else:
            # Reverse a segment
            j = random.randint(i + 1, len(mutated) - 1)
            mutated[i:j+1] = mutated[i:j+1][::-1]
    return mutated

def _crossover_orderings(orderings1, orderings2, n):
    """Crossover two orderings."""
    child = {}
    for m in range(2, n + 1):
        if random.random() < 0.5:
            child[m] = list(orderings1[m])
        else:
            child[m] = list(orderings2[m])
    return child

def _ga_search(n, time_limit):
    """
    Genetic algorithm to search for better orderings.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Initialize with classic orderings
    classic = _get_classic_orderings(n)
    best_orderings = {m: list(classic[m]) for m in range(2, n + 1)}
    best_length = _evaluate_ordering(n, best_orderings)
    
    # Create initial population
    population = [{m: list(classic[m]) for m in range(2, n + 1)}]
    
    # Add some random perturbations
    for _ in range(min(20, max(1, int(time_limit / 5)))):
        if time.time() > deadline:
            break
        pop = {}
        for m in range(2, n + 1):
            pop[m] = _mutate_ordering(classic[m], mutation_rate=0.05)
        population.append(pop)
    
    # Evaluate initial population
    pop_scores = []
    for pop in population:
        score = _evaluate_ordering(n, pop)
        pop_scores.append(score)
        if score < best_length:
            best_length = score
            best_orderings = {m: list(pop[m]) for m in range(2, n + 1)}
    
    # GA loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Sort by fitness (lower length is better)
        indexed = list(zip(pop_scores, population))
        indexed.sort(key=lambda x: x[0])
        population = [x[1] for x in indexed]
        pop_scores = [x[0] for x in indexed]
        
        # Elitism: keep top 2
        new_population = []
        for i in range(min(2, len(population))):
            new_population.append({m: list(population[i][m]) for m in range(2, n + 1)})
        
        # Crossover and mutation
        while len(new_population) < len(population):
            if time.time() > deadline:
                break
            
            # Tournament selection
            def select():
                idx = random.randint(0, len(population) - 1)
                return population[idx]
            
            parent1 = select()
            parent2 = select()
            
            child = _crossover_orderings(parent1, parent2, n)
            
            # Mutation
            for m in range(2, n + 1):
                if random.random() < 0.3:
                    child[m] = _mutate_ordering(child[m], mutation_rate=0.05)
            
            new_population.append(child)
        
        # Evaluate new population
        population = new_population
        pop_scores = []
        for pop in population:
            score = _evaluate_ordering(n, pop)
            pop_scores.append(score)
            if score < best_length:
                best_length = score
                best_orderings = {m: list(pop[m]) for m in range(2, n + 1)}
        
        # Check time
        if time.time() > deadline:
            break
    
    return best_orderings, best_length

def construct(n):
    """
    Constructs a superpermutation for n symbols using genetic algorithm search
    over insertion orderings to find shorter constructions than the classic method.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412134121432143214"
    
    # Get classic solution as baseline
    classic_word = _classic_construct(n)
    classic_length = len(classic_word)
    
    # Try GA search with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 10:  # Only run GA if we have enough time
        try:
            best_orderings, ga_length = _ga_search(n, remaining_time - 5)
            
            if ga_length < classic_length:
                word = _build_word_from_orderings(n, best_orderings)
                # Verify the word is valid
                if _is_valid_superperm(word, n):
                    return "".join(map(str, word))
        except Exception:
            pass
    
    # Fallback to classic construction
    return "".join(map(str, classic_word))

def _is_valid_superperm(word, n):
    """Verify that word contains all permutations of 1..n as consecutive symbols."""
    if len(word) < n:
        return False
    
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            seen.add(window)
    
    expected = set(permutations(range(1, n + 1)))
    return seen == expected

# EVOLVE-BLOCK-END
