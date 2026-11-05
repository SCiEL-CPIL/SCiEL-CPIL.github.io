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

# Precompute incidence structures for fast fitness
# For each element, which 3-APs does it participate in?
def build_ap_membership():
    """Build membership lists for 3-APs and 4-APs."""
    n3 = len(THREE_APS)
    n4 = len(FOUR_APS)
    
    # 3-AP membership: for each element, list of 3-AP indices containing it
    three_membership = [[] for _ in range(N)]
    for i, (a, b, c) in enumerate(THREE_APS):
        three_membership[a].append(i)
        three_membership[b].append(i)
        three_membership[c].append(i)
    
    # 4-AP membership
    four_membership = [[] for _ in range(N)]
    for i, (a, b, c, d) in enumerate(FOUR_APS):
        four_membership[a].append(i)
        four_membership[b].append(i)
        four_membership[c].append(i)
        four_membership[d].append(i)
    
    return three_membership, four_membership

THREE_MEMBERSHIP, FOUR_MEMBERSHIP = build_ap_membership()

def count_three_aps_fast(mask):
    """Count 3-APs using precomputed structure - check each 3-AP."""
    if len(THREE_APS) == 0:
        return 0
    # Vectorized check
    valid = np.all(mask[THREE_APS], axis=1)
    return int(np.sum(valid))

def has_four_ap_fast(mask):
    """Check if mask contains any 4-AP."""
    if len(FOUR_APS) == 0:
        return False
    valid = np.all(mask[FOUR_APS], axis=1)
    return bool(np.any(valid))

def get_violating_4aps(mask):
    """Get indices of all 4-APs violated by mask."""
    if len(FOUR_APS) == 0:
        return np.array([], dtype=int)
    valid = np.all(mask[FOUR_APS], axis=1)
    return np.where(valid)[0]

def fitness(mask):
    """Fitness = number of 3-APs if no 4-AP, else negative large number."""
    if has_four_ap_fast(mask):
        return -100000
    return count_three_aps_fast(mask)

def repair_mask(mask):
    """Remove elements to eliminate all 4-APs, preferring those with low 3-AP contribution."""
    mask = mask.copy()
    max_iter = 1000
    for _ in range(max_iter):
        violating = get_violating_4aps(mask)
        if len(violating) == 0:
            break
        
        # Find which elements appear in violating 4-APs and count their 3-AP contributions
        # For each violating 4-AP, we want to remove the element that contributes least to 3-APs
        best_removal = None
        best_contribution = float('inf')
        
        for ap_idx in violating:
            ap = FOUR_APS[ap_idx]
            for pos in ap:
                if mask[pos]:
                    # Count how many 3-APs this element is part of (in current mask)
                    contribution = 0
                    for three_idx in THREE_MEMBERSHIP[pos]:
                        a, b, c = THREE_APS[three_idx]
                        if mask[a] and mask[b] and mask[c]:
                            contribution += 1
                    if contribution < best_contribution:
                        best_contribution = contribution
                        best_removal = pos
        
        if best_removal is not None:
            mask[best_removal] = False
        else:
            # Fallback: remove first element of first violating AP
            ap = FOUR_APS[violating[0]]
            mask[ap[0]] = False
    
    return mask

def random_4ap_free_subset():
    """Generate a random 4-AP-free subset using greedy with randomization."""
    mask = np.zeros(N, dtype=bool)
    order = np.random.permutation(N)
    for idx in order:
        mask[idx] = True
        if has_four_ap_fast(mask):
            mask[idx] = False
    return mask

def greedy_4ap_free_subset():
    """Greedy: add elements in order, skip if creates 4-AP."""
    mask = np.zeros(N, dtype=bool)
    for idx in range(N):
        mask[idx] = True
        if has_four_ap_fast(mask):
            mask[idx] = False
    return mask

def greedy_by_3ap_density():
    """Greedy: add elements in order of 3-AP potential."""
    # Count how many 3-APs each element participates in
    density = np.zeros(N, dtype=int)
    for a, b, c in THREE_APS:
        density[a] += 1
        density[b] += 1
        density[c] += 1
    
    mask = np.zeros(N, dtype=bool)
    # Add elements in descending order of density
    order = np.argsort(-density)
    for idx in order:
        mask[idx] = True
        if has_four_ap_fast(mask):
            mask[idx] = False
    return mask

def local_search(mask, max_iter=200):
    """Improve mask via local search: try all additions/removals."""
    best_mask = mask.copy()
    best_fit = fitness(best_mask)
    
    for _ in range(max_iter):
        improved = False
        
        # Try adding each missing element
        missing = np.where(~best_mask)[0]
        for idx in missing:
            test_mask = best_mask.copy()
            test_mask[idx] = True
            if not has_four_ap_fast(test_mask):
                new_fit = count_three_aps_fast(test_mask)
                if new_fit > best_fit:
                    best_mask = test_mask
                    best_fit = new_fit
                    improved = True
                    break
        
        # Try removing each present element (only if it doesn't help)
        if not improved:
            present = np.where(best_mask)[0]
            for idx in present:
                test_mask = best_mask.copy()
                test_mask[idx] = False
                new_fit = count_three_aps_fast(test_mask)
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
    child = repair_mask(child)
    return child

def mutate(mask, mutation_rate=0.1):
    """Random mutation: flip some bits, then repair 4-APs."""
    child = mask.copy().astype(bool)
    flip_mask = np.random.random(N) < mutation_rate
    child[flip_mask] = ~child[flip_mask]
    child = repair_mask(child)
    return child

def genetic_algorithm(pop_size=40, generations=80, max_local_iter=100):
    """Run genetic algorithm to find a good 4-AP-free subset."""
    np.random.seed(42)
    
    # Initialize population with diverse strategies
    population = []
    population.append(greedy_4ap_free_subset())
    population.append(greedy_by_3ap_density())
    for _ in range(pop_size - 2):
        population.append(random_4ap_free_subset())
    
    # Evaluate initial fitness
    fitnesses = np.array([fitness(mask) for mask in population])
    
    best_mask = population[np.argmax(fitnesses)]
    best_fit = np.max(fitnesses)
    
    for gen in range(generations):
        # Adaptive mutation rate
        mutation_rate = 0.1 * (1.0 - gen / generations) + 0.02
        
        new_population = []
        for _ in range(pop_size):
            # Tournament selection
            tournament = np.random.randint(0, pop_size, 3)
            winner1 = tournament[np.argmax(fitnesses[tournament])]
            parent1 = population[winner1]
            
            tournament2 = np.random.randint(0, pop_size, 3)
            winner2 = tournament2[np.argmax(fitnesses[tournament2])]
            parent2 = population[winner2]
            
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate=mutation_rate)
            child = local_search(child, max_iter=max_local_iter)
            
            new_population.append(child)
        
        new_fitnesses = np.array([fitness(mask) for mask in new_population])
        
        # Elitism
        old_best_idx = np.argmax(fitnesses)
        new_best_idx = np.argmax(new_fitnesses)
        
        if new_fitnesses[new_best_idx] > fitnesses[old_best_idx]:
            fitnesses = new_fitnesses
            population = new_population
            best_fit = np.max(fitnesses)
            best_mask = population[np.argmax(fitnesses)]
        else:
            worst_idx = np.argmin(new_fitnesses)
            new_population[worst_idx] = population[old_best_idx]
            new_fitnesses[worst_idx] = fitnesses[old_best_idx]
            population = new_population
            fitnesses = new_fitnesses
        
        if np.max(fitnesses) > best_fit:
            best_fit = np.max(fitnesses)
            best_mask = population[np.argmax(fitnesses)]
    
    return best_mask

def construct():
    """Construct a 4-AP-free subset of [1, 256] maximizing 3-APs."""
    best_mask = genetic_algorithm(pop_size=40, generations=80, max_local_iter=100)
    
    result = sorted([i + 1 for i in range(N) if best_mask[i]])
    
    if not result:
        result = [1]
    
    return result

_CACHED_RESULT = None

def _ensure_cached():
    global _CACHED_RESULT
    if _CACHED_RESULT is None:
        _CACHED_RESULT = construct()
    return _CACHED_RESULT
