import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a Genetic Algorithm.
    
    Strategy:
    1. Initialize a population of valid covers using randomized greedy construction.
    2. Evolve the population using:
       - Crossover: Combine blocks from two parents, repair validity by adding greedy blocks for uncovered t-subsets.
       - Mutation: Randomly swap, add, or remove blocks to explore the neighborhood.
       - Selection: Keep the best half of the population based on size (number of blocks).
    3. Return the best individual found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Helper to get t-subsets of a block
    def get_block_tsubs(block):
        return [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
    
    # Precompute t-subsets for all possible k-blocks to speed up evaluation
    # C(19,8) = 75582, which is manageable
    all_k_blocks = list(itertools.combinations(range(v), k))
    block_tsubs_cache = {}
    for b in all_k_blocks:
        block_tsubs_cache[b] = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
    
    # Helper to compute cover counts for a set of blocks
    def compute_cover_counts(blocks):
        counts = [0] * num_t_subsets
        for b in blocks:
            for idx in block_tsubs_cache[b]:
                counts[idx] += 1
        return counts
    
    # Helper to check if a block set is a valid cover
    def is_valid_cover(blocks):
        counts = compute_cover_counts(blocks)
        return all(c > 0 for c in counts)
        
    # Helper to repair a set of blocks to be a valid cover using greedy addition
    def repair_cover(blocks, cover_counts):
        blocks = list(blocks)
        # Find uncovered t-subsets
        uncovered = [i for i, c in enumerate(cover_counts) if c == 0]
        
        while uncovered:
            # Pick a random uncovered t-subset
            target_idx = random.choice(uncovered)
            target_subset = all_t_subsets[target_idx]
            
            # Find the best block that covers the most uncovered subsets
            best_block = None
            best_gain = -1
            
            # Sample a subset of all blocks to speed up search? 
            # For v=19, k=8, t=5, C(19,8)=75582. 
            # Checking all is fast enough if we do it a few times.
            # Let's try a random sample first, then fallback to full search if needed?
            # Actually, for repair, we just need *a* block that covers the target.
            # To be efficient, we can pick a random block that contains the target subset.
            
            # Generate candidates: blocks that contain target_subset
            # target_subset is a tuple of t points.
            # We need to extend it to k points.
            # Remaining points: v - t
            # Choose k - t from v - t.
            
            remaining_points = [x for x in range(v) if x not in target_subset]
            
            # Randomly select k-t points
            if len(remaining_points) < k - t:
                # Should not happen if v >= k
                pass
            
            selected_extra = random.sample(remaining_points, k - t)
            new_block = tuple(sorted(target_subset + tuple(selected_extra)))
            
            # Check if this block is already in the set
            if new_block in set(blocks):
                 # Try another random extension
                 for _ in range(10):
                     selected_extra = random.sample(remaining_points, k - t)
                     new_block = tuple(sorted(target_subset + tuple(selected_extra)))
                     if new_block not in set(blocks):
                         break
            
            # Add the block
            blocks.append(new_block)
            for idx in block_tsubs_cache[new_block]:
                cover_counts[idx] += 1
            
            # Remove newly covered from uncovered list (approximate, will be fixed next iteration if needed)
            # To be precise, we should update uncovered properly, but for speed, 
            # we can just re-scan if the list gets large. 
            # Simpler: just remove indices that are now covered.
            new_uncovered = []
            for idx in uncovered:
                if cover_counts[idx] == 0:
                    new_uncovered.append(idx)
            uncovered = new_uncovered
            
        return blocks

    # --- Phase 1: Initial Population ---
    def greedy_initial():
        cover_count = [0] * num_t_subsets
        blocks = []
        uncovered_set = set(range(num_t_subsets))
        
        # Randomize point order for diversity
        points = list(range(v))
        
        while uncovered_set:
            # Pick a random uncovered t-subset
            target_idx = random.choice(list(uncovered_set))
            target_subset = all_t_subsets[target_idx]
            
            block_points = set(target_subset)
            remaining_points = [x for x in points if x not in block_points]
            random.shuffle(remaining_points)
            
            # Greedily add points to maximize coverage of uncovered subsets
            for _ in range(k - t):
                best_point = -1
                best_gain = -1
                
                for x in remaining_points:
                    gain = 0
                    for rest in itertools.combinations(block_points, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        idx = t_subset_to_idx.get(new_sub)
                        if idx is not None and idx in uncovered_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point != -1:
                    block_points.add(best_point)
                    remaining_points.remove(best_point)
                else:
                    if remaining_points:
                        p = remaining_points.pop(0)
                        block_points.add(p)
            
            block = tuple(sorted(block_points))
            
            # Avoid duplicate blocks
            if block not in blocks:
                blocks.append(block)
                for ts_idx in get_block_tsubs(block):
                    if ts_idx in uncovered_set:
                        uncovered_set.remove(ts_idx)
        
        return blocks

    # Initialize population
    pop_size = 20
    population = []
    
    for _ in range(pop_size):
        random.seed() # Let random choose, or use different seeds
        blocks = greedy_initial()
        # Ensure validity (greedy should produce valid, but double check/repair just in case)
        counts = compute_cover_counts(blocks)
        if not all(c > 0 for c in counts):
            blocks = repair_cover(blocks, counts)
        population.append(blocks)
    
    best_block = min(population, key=len)
    best_len = len(best_block)
    
    # --- Phase 2: Genetic Algorithm ---
    # Parameters
    mutation_rate = 0.1
    crossover_rate = 0.7
    
    while time.time() < deadline - 5.0:
        # Selection: Keep best half
        population.sort(key=len)
        population = population[:pop_size // 2]
        
        new_population = list(population)
        
        while len(new_population) < pop_size:
            # Select parents
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            if random.random() < crossover_rate:
                # Crossover: Combine parents
                # Union of blocks, then repair
                combined_blocks = list(set(p1 + p2))
                # Randomly shuffle to break bias
                random.shuffle(combined_blocks)
                
                # Repair to ensure validity and potentially reduce size?
                # Actually, union is valid. We want to try to reduce it.
                # Let's try to remove redundant blocks from the union.
                counts = compute_cover_counts(combined_blocks)
                
                # Greedy reduction on the combined set
                changed = True
                while changed and time.time() < deadline - 10.0:
                    changed = False
                    indices = list(range(len(combined_blocks)))
                    random.shuffle(indices)
                    for i in indices:
                        if i >= len(combined_blocks):
                            continue
                        b = combined_blocks[i]
                        removable = True
                        for idx in block_tsubs_cache[b]:
                            if counts[idx] == 1:
                                removable = False
                                break
                        if removable:
                            for idx in block_tsubs_cache[b]:
                                counts[idx] -= 1
                            combined_blocks.pop(i)
                            changed = True
                            break
                
                child = combined_blocks
            else:
                # No crossover, just pick one and mutate
                child = list(p1)
            
            # Mutation
            if random.random() < mutation_rate:
                # Try to swap a block
                if child:
                    i = random.randint(0, len(child) - 1)
                    old_block = child[i]
                    new_block = random.choice(all_k_blocks)
                    if new_block != old_block and new_block not in child:
                        # Check if swap maintains validity
                        # This is expensive to check strictly. 
                        # Let's just do it and repair if invalid.
                        child[i] = new_block
                        counts = compute_cover_counts(child)
                        if not all(c > 0 for c in counts):
                            child = repair_cover(child, counts)
                
                # Try to remove a block (if redundant)
                if child and random.random() < 0.5:
                    i = random.randint(0, len(child) - 1)
                    b = child[i]
                    counts = compute_cover_counts(child)
                    removable = True
                    for idx in block_tsubs_cache[b]:
                        if counts[idx] == 1:
                            removable = False
                            break
                    if removable:
                        child.pop(i)
                
                # Try to add a block (if it helps cover something better? No, just random exploration)
                # Adding a block increases size, so we only do this if we expect to remove more later.
                # Skip adding for now to keep population small.
            
            new_population.append(child)
            
            # Update best
            if len(child) < best_len:
                best_len = len(child)
                best_block = list(child)
        
        population = new_population
        
    # Final Reduction Pass on the best result found
    final_blocks = [tuple(b) for b in best_block]
    final_block_set = set(final_blocks)
    final_cover_count = [0] * num_t_subsets
    for b in final_blocks:
        for idx in block_tsubs_cache[b]:
            final_cover_count[idx] += 1
    
    # Simple greedy reduction
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        indices = list(range(len(final_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if i >= len(final_blocks):
                continue
            b = final_blocks[i]
            removable = True
            for idx in block_tsubs_cache[b]:
                if final_cover_count[idx] == 1:
                    removable = False
                    break
            
            if removable:
                for idx in block_tsubs_cache[b]:
                    final_cover_count[idx] -= 1
                final_blocks.pop(i)
                final_block_set.remove(b)
                changed = True
                break
    
    if len(final_blocks) < len(best_block):
        best_block = [list(b) for b in final_blocks]
        
    return best_block

