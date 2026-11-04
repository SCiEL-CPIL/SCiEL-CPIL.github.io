import itertools
import random
import time
import math
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # Pre-compute all t-subsets for the universe
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    
    # Map triple to index for faster counting
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Helper to get indices of triples in a block
    def get_block_triple_indices(block):
        return [triple_to_idx[tri] for tri in itertools.combinations(block, t)]
    
    # Precompute all possible blocks? C(24,7) = 346,104. This is manageable.
    # But storing them and their triple lists might be heavy.
    # Let's just generate them on the fly or use a limited set.
    # Actually, for GA, we need a pool of "good" blocks.
    # Let's generate a pool of random blocks and evaluate their "quality" (number of unique triples they cover in a reference solution?)
    # No, let's just use random blocks for mutation and crossover.
    
    # 1. Initial Population Generation
    # Use a greedy algorithm to create a good initial individual.
    def greedy_construct():
        uncovered = set(range(num_triples))
        blocks = []
        
        while uncovered and time.time() < deadline - 5:
            # Pick a random uncovered triple to start the block
            start_tri_idx = random.choice(list(uncovered))
            start_tri = all_triples[start_tri_idx]
            
            block = list(start_tri)
            
            while len(block) < k:
                current_block_set = set(block)
                candidates = [x for x in range(v) if x not in current_block_set]
                
                best_gain = -1
                best_candidates = []
                
                for x in candidates:
                    # Calculate gain: number of triples in new_block that are in uncovered
                    gain = 0
                    for pair in itertools.combinations(block, t-1):
                        tri = tuple(sorted(pair + (x,)))
                        tri_idx = triple_to_idx[tri]
                        if tri_idx in uncovered:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                        
                if best_candidates:
                    # Random tie-breaking
                    block.append(random.choice(best_candidates))
                else:
                    block.append(random.choice(candidates))
                    
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for tri_idx in get_block_triple_indices(block):
                if tri_idx in uncovered:
                    uncovered.remove(tri_idx)
                    
        return blocks

    # Create initial population
    pop_size = 20
    population = []
    for _ in range(pop_size):
        ind = greedy_construct()
        # Clean up redundancies
        counts = [0] * num_triples
        for b in ind:
            for tri_idx in get_block_triple_indices(b):
                counts[tri_idx] += 1
        
        clean_ind = []
        for b in ind:
            b_indices = get_block_triple_indices(b)
            can_remove = True
            for tri_idx in b_indices:
                if counts[tri_idx] == 1:
                    can_remove = False
                    break
            if not can_remove:
                clean_ind.append(b)
                for tri_idx in b_indices:
                    counts[tri_idx] -= 1
        
        population.append(clean_ind)
    
    best_ind = min(population, key=len)
    best_len = len(best_ind)
    
    # GA Loop
    generations = 0
    no_improve_gen = 0
    
    while time.time() < deadline - 5:
        generations += 1
        new_population = []
        
        # Selection: Keep top half
        population.sort(key=len)
        survivors = population[:pop_size//2]
        
        # Crossover and Mutation to fill the rest
        while len(new_population) < pop_size and time.time() < deadline - 5:
            # Pick parents
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            
            # Crossover: Take blocks from both parents, union, then reduce
            # This is a "union" crossover
            candidate_blocks = list(p1) + list(p2)
            
            # Randomly remove some blocks to encourage diversity/size reduction
            # Or just take a random subset?
            # Let's take all, then do a strong reduction phase.
            
            # Reduction Phase: Greedy removal of redundant blocks
            counts = [0] * num_triples
            for b in candidate_blocks:
                for tri_idx in get_block_triple_indices(b):
                    counts[tri_idx] += 1
            
            # Remove redundant blocks
            valid_blocks = []
            for b in candidate_blocks:
                b_indices = get_block_triple_indices(b)
                can_remove = True
                for tri_idx in b_indices:
                    if counts[tri_idx] == 1:
                        can_remove = False
                        break
                if not can_remove:
                    valid_blocks.append(b)
                    for tri_idx in b_indices:
                        counts[tri_idx] -= 1
            
            # Check if valid (covers all)
            # If not, repair
            # We need to check if counts are all > 0
            # Wait, the reduction logic above keeps a block if it has a unique triple.
            # If we remove all blocks with unique triples, we might break coverage.
            # The logic: "If counts[tri_idx] == 1, we CANNOT remove this block."
            # So we keep it.
            # This ensures that if a triple was covered by exactly 1 block, that block is kept.
            # So the result is always valid if the input was valid.
            
            # However, we might still have redundancy if counts > 1.
            # The loop above removes blocks that do NOT have any unique triple.
            # This is a valid reduction.
            
            # Mutation: Try to swap a block for a better one?
            # Or just add a random block and reduce?
            if random.random() < 0.3:
                # Add a random block
                new_block = tuple(sorted(random.sample(range(v), k)))
                valid_blocks.append(new_block)
                # Re-reduce
                counts = [0] * num_triples
                for b in valid_blocks:
                    for tri_idx in get_block_triple_indices(b):
                        counts[tri_idx] += 1
                
                final_blocks = []
                for b in valid_blocks:
                    b_indices = get_block_triple_indices(b)
                    can_remove = True
                    for tri_idx in b_indices:
                        if counts[tri_idx] == 1:
                            can_remove = False
                            break
                    if not can_remove:
                        final_blocks.append(b)
                        for tri_idx in b_indices:
                            counts[tri_idx] -= 1
                valid_blocks = final_blocks
            
            new_population.append(valid_blocks)
        
        # Replace population
        population = new_population
        
        # Check for improvement
        current_best = min(population, key=len)
        if len(current_best) < best_len:
            best_len = len(current_best)
            best_ind = current_best
            no_improve_gen = 0
        else:
            no_improve_gen += 1
            
        # If stuck, inject a fresh greedy solution to restart diversity
        if no_improve_gen > 5:
            fresh = greedy_construct()
            # Clean it
            counts = [0] * num_triples
            for b in fresh:
                for tri_idx in get_block_triple_indices(b):
                    counts[tri_idx] += 1
            
            clean_fresh = []
            for b in fresh:
                b_indices = get_block_triple_indices(b)
                can_remove = True
                for tri_idx in b_indices:
                    if counts[tri_idx] == 1:
                        can_remove = False
                        break
                if not can_remove:
                    clean_fresh.append(b)
                    for tri_idx in b_indices:
                        counts[tri_idx] -= 1
            
            population[random.randint(0, len(population)-1)] = clean_fresh
            no_improve_gen = 0

    # Final cleanup of best_ind
    counts = [0] * num_triples
    for b in best_ind:
        for tri_idx in get_block_triple_indices(b):
            counts[tri_idx] += 1
            
    final_blocks = []
    for b in best_ind:
        b_indices = get_block_triple_indices(b)
        can_remove = True
        for tri_idx in b_indices:
            if counts[tri_idx] == 1:
                can_remove = False
                break
        if not can_remove:
            final_blocks.append(b)
            for tri_idx in b_indices:
                counts[tri_idx] -= 1
        
    return [list(b) for b in final_blocks]
