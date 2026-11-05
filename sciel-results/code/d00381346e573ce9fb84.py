"""Genetic algorithm covering design C(19,7,4)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

def _precompute(v, k, t):
    """Precompute all t-subsets and block-to-tsubset mappings."""
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute all possible k-blocks and which t-subsets they cover
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_tsubsets = {}
    for b in all_blocks:
        tsubs = []
        for combo in itertools.combinations(b, t):
            idx = t_index_map.get(combo)
            if idx is not None:
                tsubs.append(idx)
        block_to_tsubsets[b] = np.array(tsubs, dtype=np.int32)
    
    return all_t_subsets, t_index_map, num_t_subsets, all_blocks, block_to_tsubsets

def _greedy_cover_fast(v, k, t, all_blocks, block_to_tsubsets, num_t_subsets, max_blocks=None):
    """Fast greedy covering using precomputed data."""
    uncovered = np.ones(num_t_subsets, dtype=bool)
    blocks = []
    remaining_blocks = list(all_blocks)
    
    if max_blocks is None:
        max_blocks = len(remaining_blocks)
    
    for _ in range(max_blocks):
        if not uncovered.any():
            break
        
        # Find block covering most uncovered t-subsets
        best_count = -1
        best_block = None
        best_idx = -1
        
        for i, b in enumerate(remaining_blocks):
            tsubs = block_to_tsubsets[b]
            count = np.sum(uncovered[tsubs])
            if count > best_count:
                best_count = count
                best_block = b
                best_idx = i
        
        if best_block is None or best_count == 0:
            break
        
        blocks.append(list(best_block))
        remaining_blocks.pop(best_idx)
        
        # Update uncovered
        tsubs = block_to_tsubsets[best_block]
        uncovered[tsubs] = False
    
    return blocks

def _repair_fast(v, k, t, blocks, all_blocks, block_to_tsubsets, num_t_subsets, t_index_map):
    """Repair a partial covering by greedily adding blocks."""
    # Determine which t-subsets are already covered
    covered = np.zeros(num_t_subsets, dtype=bool)
    for block in blocks:
        b = tuple(sorted(block))
        if b in block_to_tsubsets:
            covered[block_to_tsubsets[b]] = True
    
    uncovered = ~covered
    if not uncovered.any():
        return blocks
    
    blocks = list(blocks)
    used_blocks = set(tuple(sorted(b)) for b in blocks)
    remaining_blocks = [b for b in all_blocks if tuple(b) not in used_blocks]
    
    while uncovered.any() and remaining_blocks:
        best_count = -1
        best_block = None
        best_idx = -1
        
        for i, b in enumerate(remaining_blocks):
            tsubs = block_to_tsubsets[b]
            count = np.sum(uncovered[tsubs])
            if count > best_count:
                best_count = count
                best_block = b
                best_idx = i
        
        if best_block is None or best_count == 0:
            break
        
        blocks.append(list(best_block))
        remaining_blocks.pop(best_idx)
        used_blocks.add(tuple(best_block))
        
        tsubs = block_to_tsubsets[best_block]
        uncovered[tsubs] = False
    
    return blocks

def _crossover_merger(parent1, parent2, v, k, t, all_blocks, block_to_tsubsets, num_t_subsets, max_blocks):
    """Crossover: merge two coverings by selecting blocks that cover most uncovered t-subsets."""
    pool = list(parent1) + list(parent2)
    seen = set()
    unique_pool = []
    for b in pool:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            unique_pool.append(key)
    
    uncovered = np.ones(num_t_subsets, dtype=bool)
    child_blocks = []
    
    for _ in range(max_blocks):
        if not uncovered.any():
            break
        
        best_block = None
        best_count = -1
        best_idx = -1
        
        for i, b in enumerate(unique_pool):
            if b in block_to_tsubsets:
                tsubs = block_to_tsubsets[b]
                count = np.sum(uncovered[tsubs])
                if count > best_count:
                    best_count = count
                    best_block = b
                    best_idx = i
        
        if best_block is None or best_count == 0:
            break
        
        child_blocks.append(list(best_block))
        unique_pool.pop(best_idx)
        
        tsubs = block_to_tsubsets[best_block]
        uncovered[tsubs] = False
    
    return child_blocks

def construct(v, k, t):
    """Main entry point: use genetic algorithm to find a covering design."""
    start_time = time.time()
    time_limit = 115.0
    
    all_t_subsets, t_index_map, num_t_subsets, all_blocks, block_to_tsubsets = _precompute(v, k, t)
    
    pop_size = 6
    max_blocks_per_individual = 200
    population = []
    
    greedy1 = _greedy_cover_fast(v, k, t, all_blocks, block_to_tsubsets, num_t_subsets, max_blocks_per_individual)
    population.append(greedy1)
    
    random.shuffle(all_blocks)
    greedy2 = _greedy_cover_fast(v, k, t, all_blocks, block_to_tsubsets, num_t_subsets, max_blocks_per_individual)
    population.append(greedy2)
    
    for i in range(pop_size - 2):
        num_random = random.randint(30, 60)
        random_blocks = [sorted(random.sample(range(v), k)) for _ in range(num_random)]
        repaired = _repair_fast(v, k, t, random_blocks, all_blocks, block_to_tsubsets, num_t_subsets, t_index_map)
        population.append(repaired)
    
    def fitness(blocks):
        return len(blocks)
    
    best_individual = min(population, key=fitness)
    best_fitness = fitness(best_individual)
    
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        
        population.sort(key=fitness)
        
        elite_size = max(1, pop_size // 2)
        elite = population[:elite_size]
        
        new_population = list(elite)
        
        while len(new_population) < pop_size:
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            
            child = _crossover_merger(p1, p2, v, k, t, all_blocks, block_to_tsubsets, num_t_subsets, max_blocks_per_individual)
            child = _repair_fast(v, k, t, child, all_blocks, block_to_tsubsets, num_t_subsets, t_index_map)
            
            if random.random() < 0.4 and child:
                num_mut = random.randint(1, max(1, len(child) // 5))
                for _ in range(num_mut):
                    idx = random.randint(0, len(child) - 1)
                    child[idx] = sorted(random.sample(range(v), k))
                child = _repair_fast(v, k, t, child, all_blocks, block_to_tsubsets, num_t_subsets, t_index_map)
            
            new_population.append(child)
        
        population = new_population
        
        current_best = min(population, key=fitness)
        current_best_fitness = fitness(current_best)
        if current_best_fitness < best_fitness:
            best_individual = current_best
            best_fitness = current_best_fitness
        
        if best_fitness <= 151:
            break
    
    final_blocks = _repair_fast(v, k, t, best_individual, all_blocks, block_to_tsubsets, num_t_subsets, t_index_map)
    
    covered = np.zeros(num_t_subsets, dtype=bool)
    for block in final_blocks:
        b = tuple(sorted(block))
        if b in block_to_tsubsets:
            covered[block_to_tsubsets[b]] = True
    
    if not covered.all():
        final_blocks = _greedy_cover_fast(v, k, t, all_blocks, block_to_tsubsets, num_t_subsets, 500)
    
    return final_blocks
# EVOLVE-BLOCK-END
