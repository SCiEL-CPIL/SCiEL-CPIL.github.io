"""Genetic algorithm approach for covering design C(19,8,5)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct(v, k, t):
    """
    Construct a covering design using a genetic algorithm with repair and local search.
    """
    time_limit = 110  # seconds
    
    # Precompute all t-subsets and their indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute coverage mask for each possible block (k-subset)
    block_masks = {}
    all_blocks = []
    
    # Generate all possible blocks
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx[ts]
            mask |= (1 << idx)
        block_masks[block] = mask
        
    # Precompute for each t-subset, the list of blocks that contain it
    # This helps in fast greedy selection
    tsubset_to_blocks = {ts: [] for ts in all_tsubsets}
    for block in all_blocks:
        for ts in itertools.combinations(block, t):
            tsubset_to_blocks[ts].append(block)

    def greedy_cover():
        uncovered_mask = (1 << num_tsubsets) - 1
        blocks = []
        
        while uncovered_mask:
            # Find the first set bit (first uncovered t-subset)
            # This is inefficient if we scan from 0 every time.
            # But we can just pick the first uncovered one.
            first_uncovered_idx = -1
            for i in range(num_tsubsets):
                if (uncovered_mask >> i) & 1:
                    first_uncovered_idx = i
                    break
                    
            if first_uncovered_idx == -1:
                break
                
            target_ts = all_tsubsets[first_uncovered_idx]
            
            # Get candidate blocks covering target_ts
            candidates = tsubset_to_blocks[target_ts]
            
            best_block = None
            best_gain = -1
            
            for b in candidates:
                m = block_masks[b]
                gain = bin(uncovered_mask & m).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                    
            if best_block is None:
                break # Should not happen
                
            blocks.append(list(best_block))
            uncovered_mask &= ~block_masks[best_block]
            
        return blocks

    def repair_and_trim(blocks):
        # 1. Ensure validity
        cov_mask = 0
        for b in blocks:
            cov_mask |= block_masks[tuple(b)]
            
        uncovered_mask = ((1 << num_tsubsets) - 1) & ~cov_mask
        
        while uncovered_mask:
            # Find first uncovered
            first_uncovered_idx = -1
            for i in range(num_tsubsets):
                if (uncovered_mask >> i) & 1:
                    first_uncovered_idx = i
                    break
            if first_uncovered_idx == -1:
                break
            target_ts = all_tsubsets[first_uncovered_idx]
            candidates = tsubset_to_blocks[target_ts]
            
            best_block = None
            best_gain = -1
            for b in candidates:
                m = block_masks[b]
                gain = bin(uncovered_mask & m).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                    
            if best_block is None:
                break
                
            blocks.append(list(best_block))
            uncovered_mask &= ~block_masks[best_block]
            
        # 2. Trim redundant blocks
        if not blocks:
            return blocks
            
        counts = [0] * num_tsubsets
        for b in blocks:
            for ts in itertools.combinations(b, t):
                idx = tsubset_to_idx[ts]
                counts[idx] += 1
                
        new_blocks = []
        for b in blocks:
            essential = False
            for ts in itertools.combinations(b, t):
                idx = tsubset_to_idx[ts]
                if counts[idx] == 1:
                    essential = True
                    break
            if essential:
                new_blocks.append(b)
            else:
                for ts in itertools.combinations(b, t):
                    idx = tsubset_to_idx[ts]
                    counts[idx] -= 1
                    
        return new_blocks

    def crossover(p1, p2):
        child = []
        for b in p1:
            if random.random() < 0.5:
                child.append(b)
        for b in p2:
            if random.random() < 0.5:
                child.append(b)
        unique_child = []
        seen = set()
        for b in child:
            tb = tuple(b)
            if tb not in seen:
                seen.add(tb)
                unique_child.append(b)
        return unique_child

    def mutate(ind):
        n = len(ind)
        if n == 0:
            return [list(random.choice(all_blocks))]
        
        if random.random() < 0.5:
            idx = random.randint(0, n - 1)
            new_block = list(random.choice(all_blocks))
            ind[idx] = new_block
        else:
            ind.append(list(random.choice(all_blocks)))
        return ind

    # Initialize population
    population = []
    pop_size = 100
    start_time = time.time()
    
    baseline = greedy_cover()
    baseline = repair_and_trim(baseline)
    
    for i in range(pop_size):
        if i == 0:
            population.append(baseline)
        else:
            ind = [b[:] for b in baseline]
            for _ in range(random.randint(5, 20)):
                if ind:
                    ind.pop(random.randint(0, len(ind)-1))
            ind = repair_and_trim(ind)
            population.append(ind)
            
    best_individual = min(population, key=len)
    best_len = len(best_individual)
    
    max_generations = 500
    elite_size = 10
    
    for gen in range(max_generations):
        if time.time() - start_time > time_limit:
            break
            
        population.sort(key=len)
        
        if len(population[0]) < best_len:
            best_len = len(population[0])
            best_individual = population[0]
            
        new_population = []
        for i in range(elite_size):
            new_population.append(population[i])
            
        while len(new_population) < pop_size:
            p1 = random.choice(population[:20])
            p2 = random.choice(population[:20])
            
            child = crossover(p1, p2)
            if random.random() < 0.3:
                child = mutate(child)
            child = repair_and_trim(child)
            new_population.append(child)
            
        population = new_population
        
    return best_individual
# EVOLVE-BLOCK-END
