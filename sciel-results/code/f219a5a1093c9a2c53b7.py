"""Genetic algorithm approach for 3-AP maximization with no 4-AP constraint."""
import numpy as np
import random
import time

N = 256

# Precompute progressions as bitmasks
def precompute_bitmasks():
    ap3_masks = []
    for a in range(1, N + 1):
        for d in range(1, N - a + 1):
            c = a + 2 * d
            if c <= N:
                # 0-indexed bits
                mask = (1 << (a - 1)) | (1 << (a + d - 1)) | (1 << (c - 1))
                ap3_masks.append(mask)
    
    ap4_masks = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            c = a + 3 * d
            if c <= N:
                mask = (1 << (a - 1)) | (1 << (a + d - 1)) | (1 << (a + 2 * d - 1)) | (1 << (c - 1))
                ap4_masks.append(mask)
                
    return ap3_masks, ap4_masks

AP3_MASKS, AP4_MASKS = precompute_bitmasks()

def has_4ap(mask):
    """Check if mask contains any 4-AP."""
    for m in AP4_MASKS:
        if (mask & m) == m:
            return True
    return False

def count_3aps(mask):
    """Count 3-APs in the mask."""
    count = 0
    for m in AP3_MASKS:
        if (mask & m) == m:
            count += 1
    return count

def fitness(mask):
    """Fitness = number of 3-APs if no 4-AP, else -1."""
    if has_4ap(mask):
        return -1
    return count_3aps(mask)

def repair(mask):
    """Remove elements to eliminate all 4-APs."""
    changed = True
    while changed:
        changed = False
        for m in AP4_MASKS:
            if (mask & m) == m:
                # Remove a random element from this 4-AP
                # Find the bits set in m
                bits = [i for i in range(N) if (m >> i) & 1]
                remove_idx = random.choice(bits)
                mask &= ~(1 << remove_idx)
                changed = True
    return mask

def greedy_init():
    """Greedy initialization: add elements in random order, skip if creates 4-AP."""
    mask = 0
    order = list(range(N))
    random.shuffle(order)
    for i in order:
        new_mask = mask | (1 << i)
        if not has_4ap(new_mask):
            mask = new_mask
    return mask

def random_init():
    """Random initialization with repair."""
    mask = 0
    for i in range(N):
        if random.random() < 0.5:
            mask |= (1 << i)
    return repair(mask)

def local_search(mask, max_iters=200):
    """Local search using swap moves (1-bit out, 1-bit in) and add moves."""
    best_mask = mask
    best_score = count_3aps(best_mask)
    
    for _ in range(max_iters):
        improved = False
        
        # Try adding a random 0-bit
        zeros = [i for i in range(N) if not (best_mask >> i) & 1]
        if zeros:
            i = random.choice(zeros)
            new_mask = best_mask | (1 << i)
            if not has_4ap(new_mask):
                new_score = count_3aps(new_mask)
                if new_score > best_score:
                    best_mask = new_mask
                    best_score = new_score
                    improved = True
        
        # Try swapping: remove one 1-bit, add one 0-bit
        ones = [i for i in range(N) if (best_mask >> i) & 1]
        zeros = [i for i in range(N) if not (best_mask >> i) & 1]
        if ones and zeros:
            i_out = random.choice(ones)
            i_in = random.choice(zeros)
            new_mask = (best_mask & ~(1 << i_out)) | (1 << i_in)
            if not has_4ap(new_mask):
                new_score = count_3aps(new_mask)
                if new_score > best_score:
                    best_mask = new_mask
                    best_score = new_score
                    improved = True
        
        if not improved:
            # Try a few more random swaps before giving up
            for _ in range(10):
                ones = [i for i in range(N) if (best_mask >> i) & 1]
                zeros = [i for i in range(N) if not (best_mask >> i) & 1]
                if ones and zeros:
                    i_out = random.choice(ones)
                    i_in = random.choice(zeros)
                    new_mask = (best_mask & ~(1 << i_out)) | (1 << i_in)
                    if not has_4ap(new_mask):
                        new_score = count_3aps(new_mask)
                        if new_score > best_score:
                            best_mask = new_mask
                            best_score = new_score
                            improved = True
            if not improved:
                break
                
    return best_mask

def crossover(m1, m2):
    """Uniform crossover."""
    child = 0
    for i in range(N):
        if random.random() < 0.5:
            if (m1 >> i) & 1:
                child |= (1 << i)
        else:
            if (m2 >> i) & 1:
                child |= (1 << i)
    return repair(child)

def mutate(m, rate=0.1):
    """Bit flip mutation."""
    child = m
    for i in range(N):
        if random.random() < rate:
            child ^= (1 << i)
    return repair(child)

def genetic_algorithm(pop_size=40, generations=100):
    random.seed(42)
    
    # Initialize population
    population = [greedy_init() for _ in range(pop_size // 2)]
    population += [random_init() for _ in range(pop_size - len(population))]
    
    # Evaluate
    scores = [count_3aps(m) if not has_4ap(m) else -1 for m in population]
    
    best_mask = population[scores.index(max(scores))]
    best_score = max(scores)
    
    for gen in range(generations):
        new_population = []
        new_scores = []
        
        for _ in range(pop_size):
            # Tournament selection
            t1 = random.sample(range(pop_size), 3)
            p1 = population[t1[scores[t1].index(max(scores[t1]))]]
            t2 = random.sample(range(pop_size), 3)
            p2 = population[t2[scores[t2].index(max(scores[t2]))]]
            
            child = crossover(p1, p2)
            child = mutate(child, rate=0.05)
            child = local_search(child, max_iters=50)
            
            new_population.append(child)
            score = count_3aps(child)
            new_scores.append(score)
            
        # Elitism
        old_best_idx = scores.index(max(scores))
        new_best_idx = new_scores.index(max(new_scores))
        
        if new_scores[new_best_idx] > scores[old_best_idx]:
            scores = new_scores
            population = new_population
            best_score = max(scores)
            best_mask = population[scores.index(best_score)]
        else:
            # Replace worst in new with old best
            worst_idx = new_scores.index(min(new_scores))
            new_population[worst_idx] = population[old_best_idx]
            new_scores[worst_idx] = scores[old_best_idx]
            population = new_population
            scores = new_scores
            
        if max(scores) > best_score:
            best_score = max(scores)
            best_mask = population[scores.index(best_score)]
            
    return best_mask

def construct():
    """Construct a 4-AP-free subset of [1, 256] maximizing 3-APs."""
    start_time = time.time()
    # Run GA
    best_mask = genetic_algorithm(pop_size=30, generations=60)
    
    # Convert to list
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    
    if not result:
        result = [1]
        
    return result

_CACHED_RESULT = None

def _ensure_cached():
    global _CACHED_RESULT
    if _CACHED_RESULT is None:
        _CACHED_RESULT = construct()
    return _CACHED_RESULT
