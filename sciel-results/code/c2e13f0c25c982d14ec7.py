"""Genetic algorithm approach to minimize monochromatic 4-term APs in [1,256]."""
import numpy as np
from scipy.optimize import minimize
import time

# Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= 256, d >= 1
# Represent each AP as indices (0-based) into the colouring array
def _precompute_aps():
    aps = []
    n = 256
    for d in range(1, n):
        for a in range(1, n - 3 * d + 1):
            # 1-based: a, a+d, a+2d, a+3d
            # 0-based: a-1, a+d-1, a+2d-1, a+3d-1
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    return aps

_AP_LIST = _precompute_aps()
_NUM_APS = len(_AP_LIST)

def _count_mono_aps(colours):
    """Count monochromatic 4-term APs. colours is a numpy array of 0/1 of length 256."""
    # For each AP, check if all 4 entries have the same colour
    # colours[ap[0]], colours[ap[1]], colours[ap[2]], colours[ap[3]]
    # Monochromatic if all equal: c0==c1==c2==c3
    # This is equivalent to: (c0==c1) & (c1==c2) & (c2==c3)
    # Or: XOR-based: c0^c1 == 0 and c1^c2 == 0 and c2^c3 == 0
    # Faster: for each AP, extract the 4 colours and check
    # Vectorized approach:
    idx = np.array(_AP_LIST)  # shape (num_aps, 4)
    cols = colours[idx]  # shape (num_aps, 4)
    # Monochromatic if all 4 are 0 or all 4 are 1
    # All 0: min==0 and max==0 -> all are 0
    # All 1: min==1 and max==1 -> all are 1
    # So: (cols.min(axis=1) == cols.max(axis=1))
    return int(np.sum(cols.min(axis=1) == cols.max(axis=1)))

def _random_initialization(pop_size, seed=42):
    """Generate diverse random initializations."""
    rng = np.random.RandomState(seed)
    pop = []
    for _ in range(pop_size):
        c = rng.randint(0, 2, size=256)
        pop.append(c)
    return pop

def _crossover(parent1, parent2, rng):
    """Uniform crossover."""
    mask = rng.randint(0, 2, size=256)
    child = np.where(mask == 1, parent1, parent2)
    return child

def _mutate(child, mutation_rate, rng):
    """Bit-flip mutation."""
    if rng.random() < mutation_rate:
        # Flip a random position
        pos = rng.randint(0, 256)
        child[pos] = 1 - child[pos]
    return child

def _local_search(colours, max_iters=500, rng=None):
    """Greedy local search: flip a bit if it reduces the count."""
    if rng is None:
        rng = np.random.RandomState(0)
    colours = colours.copy()
    current_count = _count_mono_aps(colours)
    
    for _ in range(max_iters):
        improved = False
        # Try flipping each position, keep the best improvement
        best_gain = 0
        best_pos = -1
        for pos in range(256):
            # Estimate gain by flipping pos
            # We need to count how many APs containing pos change their monochromatic status
            # This is expensive to do exactly per position, so let's just try flipping and counting
            # But that's 256 * O(num_aps) which is too slow for 500 iterations
            # Instead, let's use a smarter approach: precompute which APs contain each position
            
            # Actually, for local search, let's just do random flips and check
            pass
        
        # Simpler: random restart local search with random flips
        for _ in range(10):
            pos = rng.randint(0, 256)
            old_val = colours[pos]
            colours[pos] = 1 - old_val
            new_count = _count_mono_aps(colours)
            if new_count < current_count:
                current_count = new_count
                improved = True
            else:
                colours[pos] = old_val
        
        if not improved:
            break
    
    return colours, current_count

def construct():
    """
    Genetic algorithm to minimize monochromatic 4-term APs in [1,256].
    Returns 256 colours in {0,1}.
    """
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer for overhead
    
    # Precompute AP list is already done at module level
    
    # Initial population: diverse random + some structured
    pop_size = 40
    rng = np.random.RandomState(12345)
    
    # Start with some diverse initializations
    population = []
    
    # 1. Random initializations
    for _ in range(pop_size // 2):
        c = rng.randint(0, 2, size=256)
        population.append(c)
    
    # 2. Block-based patterns
    for block_size in [2, 3, 4, 5, 6, 8]:
        c = [(i // block_size) % 2 for i in range(256)]
        population.append(np.array(c, dtype=np.int8))
    
    # 3. Striped patterns
    for period in [3, 4, 5, 6, 7]:
        c = [(i % period) < period // 2 for i in range(256)]
        population.append(np.array(c, dtype=np.int8))
    
    # 4. Thue-Morse-like
    def thue_morse(n):
        tm = [0]
        for _ in range(8):
            tm = tm + [1 - x for x in tm]
        return tm
    tm_full = thue_morse(256)
    population.append(np.array(tm_full[:256], dtype=np.int8))
    
    # 5. Another structured: parity of popcount
    def popcount(x):
        c = 0
        while x:
            c += x & 1
            x >>= 1
        return c
    c = [popcount(i) % 2 for i in range(256)]
    population.append(np.array(c, dtype=np.int8))
    
    # Ensure we have pop_size individuals
    while len(population) < pop_size:
        c = rng.randint(0, 2, size=256)
        population.append(c)
    
    # Evaluate initial population
    scores = [_count_mono_aps(c) for c in population]
    
    # Local search on the best few
    best_idx = np.argmin(scores)
    best_individual = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # Apply local search to top candidates
    top_k = min(5, len(population))
    sorted_idx = np.argsort(scores)[:top_k]
    for idx in sorted_idx:
        improved_ind, improved_score = _local_search(population[idx], max_iters=200, rng=rng)
        if improved_score < best_score:
            best_score = improved_score
            best_individual = improved_ind
        population[idx] = improved_ind
        scores[idx] = improved_score
    
    # Genetic algorithm loop
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        
        # Sort by score
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Elitism: keep top individuals
        elite_size = 5
        new_population = [population[i].copy() for i in range(elite_size)]
        new_scores = scores[:elite_size]
        
        # Fill the rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def tournament_select(pop, sc, k=3):
                candidates = rng.randint(0, len(pop), size=k)
                best_c = min(candidates, key=lambda i: sc[i])
                return pop[best_c]
            
            p1 = tournament_select(population, scores)
            p2 = tournament_select(population, scores)
            
            child = _crossover(p1, p2, rng)
            child = _mutate(child, mutation_rate=0.1, rng=rng)
            
            # Local search on child (shorter)
            child, score = _local_search(child, max_iters=50, rng=rng)
            
            new_population.append(child)
            new_scores.append(score)
        
        population = new_population
        scores = new_scores
        
        # Update best
        best_idx = np.argmin(scores)
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_individual = population[best_idx].copy()
    
    return [int(x) for x in best_individual]

# Quick sanity check
if __name__ == "__main__":
    result = construct()
    assert len(result) == 256
    assert all(x in (0, 1) for x in result)
    count = _count_mono_aps(np.array(result, dtype=np.int8))
    print(f"Monochromatic 4-APs: {count}")
