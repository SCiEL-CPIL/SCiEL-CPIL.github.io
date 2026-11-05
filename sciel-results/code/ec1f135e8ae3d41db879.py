"""Genetic algorithm approach for 3-AP maximization with no 4-AP constraint."""
import numpy as np
from itertools import combinations

# Precompute all 3-APs and 4-APs in [1, 256]
N = 256

def precompute_progressions():
    """Precompute all 3-APs and 4-APs as index triples/quadruples."""
    # 3-AP: (a, a+d, a+2d) with a >= 1, a+2d <= 256, d >= 1
    three_aps = []
    for a in range(1, N + 1):
        for d in range(1, N - a + 1):
            c = a + 2 * d
            if c <= N:
                three_aps.append((a - 1, a + d - 1, c - 1))  # 0-indexed
    three_aps = np.array(three_aps, dtype=np.int32)
    
    # 4-AP: (a, a+d, a+2d, a+3d) with a >= 1, a+3d <= 256, d >= 1
    four_aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            c = a + 3 * d
            if c <= N:
                four_aps.append((a - 1, a + d - 1, a + 2 * d - 1, c - 1))  # 0-indexed
    four_aps = np.array(four_aps, dtype=np.int32)
    
    return three_aps, four_aps

# Precompute at module level
THREE_APS, FOUR_APS = precompute_progressions()

def count_three_aps(mask):
    """Count 3-APs in a bitmask (numpy boolean array of length N)."""
    # For each 3-AP (i, j, k), check if all three are in the set
    # mask is a boolean array; use fancy indexing
    if len(THREE_APS) == 0:
        return 0
    # Check all three positions
    valid = np.all(mask[THREE_APS], axis=1)
    return int(np.sum(valid))

def has_four_ap(mask):
    """Check if mask contains any 4-AP."""
    if len(FOUR_APS) == 0:
        return False
    valid = np.all(mask[FOUR_APS], axis=1)
    return bool(np.any(valid))

def fitness(mask):
    """Fitness = number of 3-APs if no 4-AP, else negative large number."""
    if has_four_ap(mask):
        return -100000
    return count_three_aps(mask)

def random_4ap_free_subset():
    """Generate a random 4-AP-free subset using greedy with randomization."""
    mask = np.zeros(N, dtype=bool)
    # Randomly decide inclusion order
    order = np.random.permutation(N)
    for idx in order:
        # Tentatively add idx
        mask[idx] = True
        if has_four_ap(mask):
            mask[idx] = False
    return mask

def greedy_4ap_free_subset():
    """Greedy: add elements in order, skip if creates 4-AP."""
    mask = np.zeros(N, dtype=bool)
    for idx in range(N):
        mask[idx] = True
        if has_four_ap(mask):
            mask[idx] = False
    return mask

def local_search(mask, max_iter=500):
    """Improve mask via local search: add/remove/swap to increase 3-APs while maintaining 4-AP-free."""
    best_mask = mask.copy()
    best_fit = fitness(best_mask)
    
    for _ in range(max_iter):
        improved = False
        # Try adding a random element
        candidates = np.where(~best_mask)[0]
        if len(candidates) > 0:
            for _ in range(10):
                idx = candidates[np.random.randint(len(candidates))]
                test_mask = best_mask.copy()
                test_mask[idx] = True
                if not has_four_ap(test_mask):
                    new_fit = count_three_aps(test_mask)
                    if new_fit > best_fit:
                        best_mask = test_mask
                        best_fit = new_fit
                        improved = True
                        break
        
        # Try removing an element
        candidates = np.where(best_mask)[0]
        if len(candidates) > 0:
            for _ in range(5):
                idx = candidates[np.random.randint(len(candidates))]
                test_mask = best_mask.copy()
                test_mask[idx] = False
                new_fit = count_three_aps(test_mask)
                if new_fit > best_fit:
                    best_mask = test_mask
                    best_fit = new_fit
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_mask

def crossover(mask1, mask2):
    """Uniform crossover between two masks."""
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    cross_points = np.random.random(N) < 0.5
    child = np.where(cross_points, mask1, mask2)
    # Repair: remove any 4-APs
    while has_four_ap(child):
        # Find a 4-AP and remove one random element from it
        valid = np.all(child[FOUR_APS], axis=1)
        ap_indices = np.where(valid)[0]
        if len(ap_indices) == 0:
            break
        ap_idx = ap_indices[np.random.randint(len(ap_indices))]
        ap = FOUR_APS[ap_idx]
        remove_idx = ap[np.random.randint(4)]
        child[remove_idx] = False
    return child

def mutate(mask, mutation_rate=0.1):
    """Random mutation: flip some bits, then repair 4-APs."""
    child = mask.copy().astype(bool)
    # Flip random bits
    flip_mask = np.random.random(N) < mutation_rate
    child[flip_mask] = ~child[flip_mask]
    # Repair 4-APs
    while has_four_ap(child):
        valid = np.all(child[FOUR_APS], axis=1)
        ap_indices = np.where(valid)[0]
        if len(ap_indices) == 0:
            break
        ap_idx = ap_indices[np.random.randint(len(ap_indices))]
        ap = FOUR_APS[ap_idx]
        remove_idx = ap[np.random.randint(4)]
        child[remove_idx] = False
    return child

def genetic_algorithm(pop_size=50, generations=100, max_local_iter=100):
    """Run genetic algorithm to find a good 4-AP-free subset."""
    np.random.seed(42)
    
    # Initialize population
    population = []
    # Include greedy solution
    population.append(greedy_4ap_free_subset())
    # Include random solutions
    for _ in range(pop_size - 1):
        population.append(random_4ap_free_subset())
    
    # Evaluate initial fitness
    fitnesses = np.array([fitness(mask) for mask in population])
    
    best_mask = population[np.argmax(fitnesses)]
    best_fit = np.max(fitnesses)
    
    for gen in range(generations):
        # Select parents (tournament selection)
        new_population = []
        for _ in range(pop_size):
            # Tournament of size 3
            tournament = np.random.randint(0, pop_size, 3)
            winner = tournament[np.argmax(fitnesses[tournament])]
            parent1 = population[winner]
            
            # Second parent
            tournament2 = np.random.randint(0, pop_size, 3)
            winner2 = tournament2[np.argmax(fitnesses[tournament2])]
            parent2 = population[winner2]
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child, mutation_rate=0.05)
            
            # Local search
            child = local_search(child, max_iter=max_local_iter)
            
            new_population.append(child)
        
        # Evaluate new population
        new_fitnesses = np.array([fitness(mask) for mask in new_population])
        
        # Elitism: keep best individual
        old_best_idx = np.argmax(fitnesses)
        new_best_idx = np.argmax(new_fitnesses)
        
        if new_fitnesses[new_best_idx] > fitnesses[old_best_idx]:
            fitnesses = new_fitnesses
            population = new_population
            best_fit = np.max(fitnesses)
            best_mask = population[np.argmax(fitnesses)]
        else:
            # Replace worst in new population with old best
            worst_idx = np.argmin(new_fitnesses)
            new_population[worst_idx] = population[old_best_idx]
            new_fitnesses[worst_idx] = fitnesses[old_best_idx]
            population = new_population
            fitnesses = new_fitnesses
        
        # Update best
        if np.max(fitnesses) > best_fit:
            best_fit = np.max(fitnesses)
            best_mask = population[np.argmax(fitnesses)]
    
    return best_mask

def construct():
    """Construct a 4-AP-free subset of [1, 256] maximizing 3-APs."""
    # Run genetic algorithm
    best_mask = genetic_algorithm(pop_size=30, generations=50, max_local_iter=50)
    
    # Convert mask to sorted list
    result = sorted([i + 1 for i in range(N) if best_mask[i]])
    
    # Ensure non-empty
    if not result:
        result = [1]
    
    return result

# Pre-run to cache result
_CACHED_RESULT = None

def _ensure_cached():
    global _CACHED_RESULT
    if _CACHED_RESULT is None:
        _CACHED_RESULT = construct()
    return _CACHED_RESULT
