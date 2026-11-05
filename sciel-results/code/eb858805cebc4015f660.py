"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np
import time

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a Genetic Algorithm.
    """
    # 1. Precomputation
    all_points = range(v)
    all_triples = list(itertools.combinations(all_points, t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Generate all possible blocks
    all_blocks = list(itertools.combinations(all_points, k))
    num_blocks = len(all_blocks)
    
    # Precompute bitsets for all blocks
    # mask[i] has bit j set if block i covers triple j
    block_masks = []
    for block in all_blocks:
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    full_mask = (1 << num_triples) - 1
    
    # Helper: Check if a list of block indices forms a valid cover
    def is_valid_cover(block_indices):
        union_mask = 0
        for idx in block_indices:
            union_mask |= block_masks[idx]
        return union_mask == full_mask
        
    # Helper: Remove redundant blocks from a list of block indices
    def remove_redundant(block_indices):
        if not block_indices:
            return []
        
        # Convert to set for faster lookup? No, we need to check coverage counts.
        # Let's use a bitset union approach for speed in checking redundancy?
        # Actually, for redundancy removal, we need to know if removing block i
        # leaves any triple uncovered.
        # Let's maintain a coverage count array for the current set of blocks.
        
        # Optimization: Use bitsets to represent the set of blocks covering each triple?
        # That's expensive to build.
        
        # Simple approach:
        # 1. Calculate union mask of all blocks. If not full, invalid.
        # 2. For each block, check if (Union of all others) == full_mask.
        # This is O(N^2 * BitOps). N is small (~76). BitOps are fast.
        
        # Even better:
        # Compute prefix and suffix unions?
        # Or just iterate and temporarily remove?
        
        # Let's try a greedy redundancy removal pass.
        # Sort blocks by "uniqueness" or just iterate.
        
        current_blocks = list(block_indices)
        changed = True
        while changed:
            changed = False
            # Shuffle to avoid bias
            indices = list(range(len(current_blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(current_blocks):
                    continue
                block_idx = current_blocks[i]
                
                # Check if removing this block leaves the cover valid
                # We can do this by computing the union of all other blocks.
                # To speed this up, we can precompute the union of all blocks,
                # but we need to know if the removed block was the *only* one covering some triple.
                
                # Alternative:
                # Let U be the union of all blocks in current_blocks.
                # Let U_without_i be the union of all blocks in current_blocks except i.
                # If U_without_i == full_mask, then block i is redundant.
                
                # Computing U_without_i from scratch is O(N * BitOps).
                # Total O(N^2 * BitOps). With N=80, 6400 ops. Very fast.
                
                union_others = 0
                for j, b_idx in enumerate(current_blocks):
                    if j == i:
                        continue
                    union_others |= block_masks[b_idx]
                    
                if union_others == full_mask:
                    # Redundant, remove it
                    current_blocks.pop(i)
                    changed = True
                    break # Restart loop as indices changed
                    
        return current_blocks

    # Helper: Greedy Set Cover (to get initial good solution)
    def greedy_cover():
        uncovered = full_mask
        selected = []
        used = set()
        
        # Pre-sort blocks by size of mask? No, dynamic gain is better.
        # But static size might be a good heuristic for init.
        # Let's just do standard greedy.
        
        while uncovered != 0:
            best_gain = 0
            best_idx = -1
            
            for i, mask in enumerate(block_masks):
                if i in used:
                    continue
                gain = (uncovered & mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    
            if best_idx == -1:
                break # Should not happen
                
            selected.append(best_idx)
            used.add(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected

    # Initial Population
    pop_size = 50
    population = []
    
    # Add a few greedy solutions (maybe with slight perturbations)
    greedy_sol = greedy_cover()
    # Remove redundancy from greedy sol
    greedy_sol = remove_redundant(greedy_sol)
    population.append(list(greedy_sol))
    
    # Generate random initial solutions by picking random blocks until valid
    for _ in range(pop_size - 1):
        sol = []
        # Start with random blocks
        # To speed up validity, we can start with a subset of greedy and add random
        # Or just random and repair.
        
        # Strategy: Start with empty, add random blocks until valid.
        # This might result in large solutions.
        # Better: Start with a random subset of greedy blocks?
        
        # Let's try: Start with random blocks.
        # Max attempts to find valid cover
        valid = False
        for attempt in range(100):
            sol = []
            # Pick random number of blocks, say 80-100
            num_blocks_to_try = random.randint(70, 100)
            candidates = random.sample(range(num_blocks), num_blocks_to_try)
            sol = candidates
            
            if is_valid_cover(sol):
                sol = remove_redundant(sol)
                valid = True
                break
        
        if valid:
            population.append(sol)
        else:
            # If we couldn't find a valid random one, use a mutated greedy
            sol = list(greedy_sol)
            # Swap a few blocks
            for _ in range(5):
                if sol:
                    idx_to_swap = random.randint(0, len(sol)-1)
                    new_block = random.randint(0, num_blocks-1)
                    sol[idx_to_swap] = new_block
                    # Repair: if invalid, add blocks
                    if not is_valid_cover(sol):
                         # Add greedy blocks for uncovered
                         uncovered = full_mask
                         for b in sol:
                             uncovered &= ~block_masks[b]
                         while uncovered != 0:
                             best_gain = 0
                             best_idx = -1
                             for i, mask in enumerate(block_masks):
                                 if i in sol:
                                     continue
                                 gain = (uncovered & mask).bit_count()
                                 if gain > best_gain:
                                     best_gain = gain
                                     best_idx = i
                             if best_idx == -1: break
                             sol.append(best_idx)
                             uncovered &= ~block_masks[best_idx]
                         sol = remove_redundant(sol)
            population.append(sol)

    # Genetic Algorithm Loop
    start_time = time.time()
    timeout = 110 # Leave 10s buffer
    
    best_sol = min(population, key=len)
    
    while time.time() - start_time < timeout:
        new_population = []
        
        # Elitism
        new_population.append(list(best_sol))
        
        while len(new_population) < pop_size:
            # Selection (Tournament)
            p1 = random.choice(population)
            p2 = random.choice(population)
            if len(p1) < len(p2):
                parent1, parent2 = p1, p2
            else:
                parent1, parent2 = p2, p1
            
            # Crossover: Uniform or Slice
            # Let's try Slice Crossover: Take first half from P1, second half from P2
            # But order doesn't matter for sets.
            # So, take a random subset of blocks from P1 and a random subset from P2?
            # Or: Union of P1 and P2, then reduce? That's expensive.
            
            # Better Crossover: 
            # 1. Take random blocks from P1
            # 2. Take random blocks from P2
            # 3. Combine and Repair/Reduce
            
            # Let's try: 
            # Child = (P1 + P2) -> Remove Redundant -> If invalid, Repair -> Remove Redundant
            # This is essentially "Union Crossover".
            
            # To make it more diverse, maybe mix them:
            # Pick 50% of blocks from P1 (randomly) and 50% from P2 (randomly)?
            # No, that might break coverage.
            
            # Let's stick to Union Crossover with aggressive reduction.
            child_blocks = set(parent1) | set(parent2)
            child_list = list(child_blocks)
            
            # Reduce
            child_list = remove_redundant(child_list)
            
            # Repair if invalid
            if not is_valid_cover(child_list):
                uncovered = full_mask
                for b in child_list:
                    uncovered &= ~block_masks[b]
                
                while uncovered != 0:
                    best_gain = 0
                    best_idx = -1
                    # Only consider blocks not in child_list
                    for i, mask in enumerate(block_masks):
                        if i in child_list:
                            continue
                        gain = (uncovered & mask).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = i
                    if best_idx == -1: break
                    child_list.append(best_idx)
                    uncovered &= ~block_masks[best_idx]
                    
                child_list = remove_redundant(child_list)
            
            # Mutation
            # 1. Random Swap
            if random.random() < 0.3 and child_list:
                idx = random.randint(0, len(child_list)-1)
                new_block = random.randint(0, num_blocks-1)
                child_list[idx] = new_block
                # Repair
                if not is_valid_cover(child_list):
                    uncovered = full_mask
                    for b in child_list:
                        uncovered &= ~block_masks[b]
                    while uncovered != 0:
                        best_gain = 0
                        best_idx = -1
                        for i, mask in enumerate(block_masks):
                            if i in child_list:
                                continue
                            gain = (uncovered & mask).bit_count()
                            if gain > best_gain:
                                best_gain = gain
                                best_idx = i
                        if best_idx == -1: break
                        child_list.append(best_idx)
                        uncovered &= ~block_masks[best_idx]
                    child_list = remove_redundant(child_list)
            
            # 2. Random Add
            if random.random() < 0.2:
                new_block = random.randint(0, num_blocks-1)
                if new_block not in child_list:
                    child_list.append(new_block)
                    child_list = remove_redundant(child_list)
            
            # 3. Random Remove (if valid)
            if random.random() < 0.2 and len(child_list) > 1:
                idx = random.randint(0, len(child_list)-1)
                block_to_remove = child_list[idx]
                temp_list = [b for i, b in enumerate(child_list) if i != idx]
                if is_valid_cover(temp_list):
                    child_list = temp_list
                    child_list = remove_redundant(child_list)
            
            new_population.append(child_list)
        
        population = new_population
        current_best = min(population, key=len)
        if len(current_best) < len(best_sol):
            best_sol = current_best

    # Convert indices back to blocks
    return [list(all_blocks[idx]) for idx in best_sol]

# EVOLVE-BLOCK-END
