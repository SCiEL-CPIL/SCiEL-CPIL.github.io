import itertools
import random
import math
import time
import numpy as np

def construct(v, k, t):
    # Parameters
    points = list(range(v))
    
    # Precompute all subsets of size t
    all_subsets = list(itertools.combinations(points, t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Helper to get covered subset indices for a block
    def get_block_coverage(block):
        # block is a tuple of k distinct ints
        return [subset_to_idx[s] for s in itertools.combinations(block, t)]
    
    # Initial Solution: Random set of blocks
    # Start with a slightly larger set to ensure coverage, then prune
    initial_size = 100
    current_blocks = []
    block_set = set()
    
    # Generate random distinct blocks
    while len(current_blocks) < initial_size:
        b = tuple(sorted(random.sample(points, k)))
        if b not in block_set:
            block_set.add(b)
            current_blocks.append(b)
            
    # Calculate initial coverage counts
    cover_count = np.zeros(n_subsets, dtype=np.int32)
    for b in current_blocks:
        covs = get_block_coverage(b)
        for idx in covs:
            cover_count[idx] += 1
            
    # Current defect: number of subsets with cover_count == 0
    current_defect = np.sum(cover_count == 0)
    
    best_blocks = list(current_blocks)
    best_defect = current_defect
    best_size = len(current_blocks)
    
    # Simulated Annealing Parameters
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.01
    alpha = 0.999 # Cooling rate
    
    T = T_start
    
    # Moves
    def random_block():
        return tuple(sorted(random.sample(points, k)))
        
    def mutate_block(block):
        # Change one element
        block_list = list(block)
        idx_to_change = random.randint(0, k-1)
        old_val = block_list[idx_to_change]
        new_val = random.choice([p for p in points if p != old_val])
        block_list[idx_to_change] = new_val
        new_block = tuple(sorted(block_list))
        return new_block, old_val, new_val, idx_to_change

    steps = 0
    max_steps = 500000
    
    while T > T_end and (time.time() - start_time) < time_limit:
        steps += 1
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.6:
            # Swap: Replace a random block with a new random block
            if not current_blocks:
                continue
            idx = random.randint(0, len(current_blocks) - 1)
            old_block = current_blocks[idx]
            new_block = random_block()
            
            # If new block is same as old, skip
            if new_block == old_block:
                continue
            
            # Check if new block is already in set
            if new_block in block_set:
                # Try a few times to find a unique one, or just accept duplicate removal logic?
                # Better to just pick another one
                # For simplicity in SA, we can allow duplicates in the list but penalize them?
                # No, the problem requires distinct blocks.
                # Let's just retry a few times
                for _ in range(5):
                    new_block = random_block()
                    if new_block not in block_set and new_block != old_block:
                        break
                else:
                    continue
                    
            # Calculate delta defect
            old_covs = get_block_coverage(old_block)
            new_covs = get_block_coverage(new_block)
            
            # Apply change temporarily to counts
            for idx_c in old_covs:
                cover_count[idx_c] -= 1
            for idx_c in new_covs:
                cover_count[idx_c] += 1
                
            new_defect = np.sum(cover_count == 0)
            delta_defect = new_defect - current_defect
            
            # Acceptance
            if delta_defect <= 0 or random.random() < math.exp(-delta_defect / T):
                # Accept
                current_blocks[idx] = new_block
                block_set.discard(old_block)
                block_set.add(new_block)
                current_defect = new_defect
                
                # Update best
                if current_defect < best_defect or (current_defect == best_defect and len(current_blocks) < best_size):
                    best_defect = current_defect
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
            else:
                # Reject: Revert counts
                for idx_c in new_covs:
                    cover_count[idx_c] -= 1
                for idx_c in old_covs:
                    cover_count[idx_c] += 1
                    
        elif move_type < 0.8:
            # Add a block
            if len(current_blocks) < 150: # Cap size
                new_block = random_block()
                if new_block in block_set:
                    continue
                    
                new_covs = get_block_coverage(new_block)
                
                # Apply
                for idx_c in new_covs:
                    cover_count[idx_c] += 1
                    
                new_defect = np.sum(cover_count == 0)
                delta_defect = new_defect - current_defect
                
                # Adding a block increases size, so we prefer it only if it reduces defect significantly
                # Or if it allows removal of others later.
                # Penalize size increase slightly?
                # Let's just accept if defect decreases or with prob
                if delta_defect < 0 or random.random() < math.exp(-delta_defect / T):
                    current_blocks.append(new_block)
                    block_set.add(new_block)
                    current_defect = new_defect
                    
                    if current_defect < best_defect or (current_defect == best_defect and len(current_blocks) < best_size):
                        best_defect = current_defect
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
                else:
                    for idx_c in new_covs:
                        cover_count[idx_c] -= 1
                        
        else:
            # Remove a block
            if len(current_blocks) > 50: # Don't go too small
                idx = random.randint(0, len(current_blocks) - 1)
                old_block = current_blocks[idx]
                old_covs = get_block_coverage(old_block)
                
                # Apply removal
                for idx_c in old_covs:
                    cover_count[idx_c] -= 1
                    
                new_defect = np.sum(cover_count == 0)
                delta_defect = new_defect - current_defect
                
                # Removal increases defect usually, but reduces size.
                # We want to accept if it doesn't increase defect too much, or if it reduces size.
                # Heuristic: Accept if delta_defect == 0 (size reduction) or with prob
                if delta_defect == 0:
                    # Always accept size reduction if defect same
                    current_blocks.pop(idx)
                    block_set.discard(old_block)
                    current_defect = new_defect
                    
                    if current_defect < best_defect or (current_defect == best_defect and len(current_blocks) < best_size):
                        best_defect = current_defect
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
                elif delta_defect > 0 and random.random() < math.exp(-delta_defect / T):
                    current_blocks.pop(idx)
                    block_set.discard(old_block)
                    current_defect = new_defect
                else:
                    # Revert
                    for idx_c in old_covs:
                        cover_count[idx_c] += 1

        # Cool down
        T *= alpha
        
    # Final cleanup: Remove redundant blocks from best solution
    final_blocks = best_blocks
    final_block_set = set(final_blocks)
    
    # Recalculate coverage for final best
    cover_count_final = np.zeros(n_subsets, dtype=np.int32)
    for b in final_blocks:
        covs = get_block_coverage(b)
        for idx in covs:
            cover_count_final[idx] += 1
            
    # Iteratively remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(final_blocks)):
            b = final_blocks[i]
            covs = get_block_coverage(b)
            redundant = True
            for idx in covs:
                if cover_count_final[idx] == 1:
                    redundant = False
                    break
            if redundant:
                for idx in covs:
                    cover_count_final[idx] -= 1
                final_blocks.pop(i)
                final_block_set.discard(b)
                changed = True
                break

    # Convert to list of lists
    return [list(b) for b in final_blocks]
