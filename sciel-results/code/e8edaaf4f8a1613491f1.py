"""
Strategy: Genetic Algorithm for Covering Design C(19,8,5)
- Population of candidate covers (lists of blocks).
- Fitness: Penalize uncovered subsets heavily, then minimize block count.
- Crossover: Merge two parents, then greedily prune redundant blocks.
- Mutation: Randomly swap a block with a random other block.
- Selection: Tournament selection.
- Initialization: Greedy cover with randomization to ensure diversity.
"""

import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    full_mask = (1 << len(all_t_subsets)) - 1
    
    all_blocks = list(itertools.combinations(range(v), k))
    block_idx_map = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute coverage mask for each block
    block_masks = [0] * len(all_blocks)
    for i, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            mask |= (1 << t_sub_to_idx[sub])
        block_masks[i] = mask

    def greedy_cover(seed):
        """Generate a valid cover using greedy with random noise."""
        random.seed(seed)
        uncovered = full_mask
        blocks = []
        
        while uncovered:
            best_block_idx = -1
            best_gain = -1
            # Add noise to break ties and diversify
            for i in range(len(all_blocks)):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > 0:
                    noisy_gain = gain + random.random() * 0.5
                    if noisy_gain > best_gain:
                        best_gain = noisy_gain
                        best_block_idx = i
            
            if best_block_idx == -1:
                break
                
            blocks.append(all_blocks[best_block_idx])
            uncovered &= ~block_masks[best_block_idx]
            
            if time.time() > deadline:
                return None
        
        return blocks

    def prune_cover(blocks):
        """Remove redundant blocks from a valid cover."""
        if not blocks:
            return blocks
        
        # Convert to indices for faster lookup
        block_indices = [block_idx_map[b] for b in blocks]
        current = list(block_indices)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Calculate coverage counts
            counts = defaultdict(int)
            for idx in current:
                mask = block_masks[idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    counts[idx_t] += 1
                    m ^= lsb
            
            # Check each block for redundancy
            new_current = []
            for idx in current:
                mask = block_masks[idx]
                essential = False
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    if counts[idx_t] == 1:
                        essential = True
                        break
                    m ^= lsb
                
                if essential:
                    new_current.append(idx)
                else:
                    changed = True
            
            current = new_current
        
        return [all_blocks[i] for i in current]

    def evaluate(blocks):
        """
        Fitness function:
        1. Check if valid (covers all).
        2. If valid, score = -len(blocks) (minimize length).
        3. If invalid, score = -10000 - uncovered_count.
        """
        if not blocks:
            return -100000
        
        covered = 0
        for b in blocks:
            idx = block_idx_map[b]
            covered |= block_masks[idx]
        
        uncovered_count = (full_mask & ~covered).bit_count()
        
        if uncovered_count == 0:
            return -len(blocks)
        else:
            return -10000 - uncovered_count

    def crossover(p1, p2):
        """
        Crossover: Merge two valid covers and prune.
        """
        merged = list(p1) + list(p2)
        # Remove duplicates
        unique_merged = list(set(merged))
        
        # Prune the merged set
        pruned = prune_cover(unique_merged)
        
        # If pruning didn't result in a valid cover (shouldn't happen if inputs were valid),
        # fallback to greedy repair
        if not pruned:
            return p1
            
        # Verify validity quickly
        covered = 0
        for b in pruned:
            idx = block_idx_map[b]
            covered |= block_masks[idx]
            
        if (full_mask & ~covered) != 0:
            # Repair: use greedy to cover remaining
            uncovered = full_mask & ~covered
            current_blocks = list(pruned)
            while uncovered:
                best_idx = -1
                best_gain = -1
                for i in range(len(all_blocks)):
                    gain = (uncovered & block_masks[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1:
                    break
                current_blocks.append(all_blocks[best_idx])
                uncovered &= ~block_masks[best_idx]
            pruned = prune_cover(current_blocks)
            
        return pruned

    def mutate(blocks, rate=0.1):
        """
        Mutation: Randomly swap some blocks.
        """
        if not blocks:
            return blocks
            
        new_blocks = list(blocks)
        n = len(new_blocks)
        
        # Swap a few blocks with random ones
        num_swaps = max(1, int(n * rate))
        for _ in range(num_swaps):
            if random.random() < rate:
                i = random.randint(0, n - 1)
                new_blocks[i] = random.choice(all_blocks)
        
        # Prune to remove redundancies introduced by mutation
        pruned = prune_cover(new_blocks)
        
        # Ensure validity
        if pruned:
             covered = 0
             for b in pruned:
                 idx = block_idx_map[b]
                 covered |= block_masks[idx]
             if (full_mask & ~covered) != 0:
                 # Repair
                 uncovered = full_mask & ~covered
                 current_blocks = list(pruned)
                 while uncovered:
                     best_idx = -1
                     best_gain = -1
                     for i in range(len(all_blocks)):
                         gain = (uncovered & block_masks[i]).bit_count()
                         if gain > best_gain:
                             best_gain = gain
                             best_idx = i
                     if best_idx == -1:
                         break
                     current_blocks.append(all_blocks[best_idx])
                     uncovered &= ~block_masks[best_idx]
                 pruned = prune_cover(current_blocks)
             return pruned
        return blocks

    def tournament_select(population, fitnesses, k=3):
        """Tournament selection."""
        contenders = random.sample(range(len(population)), k)
        best_idx = max(contenders, key=lambda i: fitnesses[i])
        return population[best_idx]

    # Main GA Loop
    population_size = 50
    generations = 50
    
    # Initialize population with diverse greedy covers
    population = []
    fitnesses = []
    
    for i in range(population_size):
        blocks = greedy_cover(i * 100)
        if blocks is None:
            continue
        blocks = prune_cover(blocks)
        if blocks:
            population.append(blocks)
            fitnesses.append(evaluate(blocks))
            
    if not population:
        # Fallback
        blocks = greedy_cover(0)
        return prune_cover(blocks) if blocks else []

    best_fitness = max(fitnesses)
    best_blocks = population[fitnesses.index(best_fitness)]
    
    for gen in range(generations):
        if time.time() > deadline:
            break
            
        new_population = []
        new_fitnesses = []
        
        for _ in range(population_size):
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            
            child = crossover(p1, p2)
            if random.random() < 0.3:
                child = mutate(child)
                
            if child:
                fit = evaluate(child)
                new_population.append(child)
                new_fitnesses.append(fit)
                
                if fit > best_fitness:
                    best_fitness = fit
                    best_blocks = child
        
        # Elitism: Keep the best individual
        if new_population:
            # Find best in new population
            best_new_idx = max(range(len(new_fitnesses)), key=lambda i: new_fitnesses[i])
            
            # Ensure the global best is in the new population
            # Simple approach: replace worst with global best if global best is better
            if best_fitness > max(new_fitnesses):
                 # Find worst in new population
                 worst_new_idx = min(range(len(new_fitnesses)), key=lambda i: new_fitnesses[i])
                 new_population[worst_new_idx] = best_blocks
                 new_fitnesses[worst_new_idx] = best_fitness
            else:
                 # Global best is already in new population or new population is better
                 pass
            
            population = new_population
            fitnesses = new_fitnesses
        else:
            break

    return best_blocks

# EVOLVE-BLOCK-END
