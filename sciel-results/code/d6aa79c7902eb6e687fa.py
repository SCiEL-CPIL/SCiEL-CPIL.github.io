import itertools
import time
import random
import numpy as np

DEADLINE_SECONDS = 115.0
RANDOM_SEED = 42

def _generic_greedy(v, k, t, start_time=None):
    """
    Fallback for non-(19,8,5) cases.
    """
    if start_time is None:
        start_time = time.time()
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    idx_map = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    block_masks = [0] * n_blocks
    blocks_containing = [[] for _ in range(n_t)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            s_idx = idx_map[tuple(sub)]
            mask |= (1 << s_idx)
            blocks_containing[s_idx].append(b_idx)
        block_masks[b_idx] = mask

    uncovered = (1 << n_t) - 1
    current_blocks = []
    
    while uncovered and (time.time() - start_time) < DEADLINE_SECONDS - 5:
        # Pick a random uncovered subset
        found_idx = -1
        attempts = 0
        while found_idx == -1 and attempts < 50:
            idx = random.randint(0, n_t - 1)
            if (uncovered >> idx) & 1:
                found_idx = idx
            attempts += 1
        
        if found_idx == -1:
            for idx in range(n_t):
                if (uncovered >> idx) & 1:
                    found_idx = idx
                    break
        
        if found_idx == -1:
            break

        candidates = blocks_containing[found_idx]
        best_gain = -1
        best_block_idx = -1
        
        for b_idx in candidates:
            gain = (block_masks[b_idx] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
        
        if best_block_idx != -1:
            current_blocks.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]

    return [list(all_blocks[i]) for i in current_blocks]

def construct(v, k, t):
    start_time = time.time()
    random.seed(RANDOM_SEED)
    
    if (v, k, t) != (19, 8, 5):
        return _generic_greedy(v, k, t, start_time)

    # Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    idx_map = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Full mask for all t-subsets
    full_mask = (1 << n_t) - 1
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Precompute masks for each block and subsets
    block_masks = [0] * n_blocks
    block_subsets = [[] for _ in range(n_blocks)]
    blocks_containing = [[] for _ in range(n_t)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        subs = []
        for sub in itertools.combinations(block, t):
            s_idx = idx_map[tuple(sub)]
            mask |= (1 << s_idx)
            subs.append(s_idx)
            blocks_containing[s_idx].append(b_idx)
        block_masks[b_idx] = mask
        block_subsets[b_idx] = subs

    best_blocks = []
    best_count = float('inf')
    
    # Helper to check validity and count
    def get_solution_count(block_indices):
        covered = 0
        for b_idx in block_indices:
            covered |= block_masks[b_idx]
        return (covered == full_mask), len(block_indices)

    # Local Search: Prune and Swap
    def local_search(block_indices, time_limit):
        current = list(block_indices)
        if not current:
            return current
            
        # 1. Prune redundant blocks
        changed = True
        while changed and (time.time() - start_time) < time_limit:
            changed = False
            counts = [0] * n_t
            for b_idx in current:
                for s_idx in block_subsets[b_idx]:
                    counts[s_idx] += 1
            
            new_current = []
            for b_idx in current:
                redundant = True
                for s_idx in block_subsets[b_idx]:
                    if counts[s_idx] == 1:
                        redundant = False
                        break
                if not redundant:
                    new_current.append(b_idx)
            
            if len(new_current) < len(current):
                current = new_current
                changed = True

        # 2. 1-in-1-out Swap
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # Recalculate counts
            counts = [0] * n_t
            for b_idx in current:
                for s_idx in block_subsets[b_idx]:
                    counts[s_idx] += 1
            
            for i in range(len(current)):
                if (time.time() - start_time) > time_limit:
                    break
                b_remove = current[i]
                
                # Find subsets that would become uncovered
                uncovered_subsets = []
                for s_idx in block_subsets[b_remove]:
                    if counts[s_idx] == 1:
                        uncovered_subsets.append(s_idx)
                
                if not uncovered_subsets:
                    # Can remove without adding
                    current.pop(i)
                    for s_idx in block_subsets[b_remove]:
                        counts[s_idx] -= 1
                    improved = True
                    break
                
                # Try to find a block that covers all uncovered_subsets
                # Candidate blocks must cover the first uncovered subset
                candidates = blocks_containing[uncovered_subsets[0]]
                best_add = -1
                best_gain = -1
                
                for b_add in candidates:
                    if b_add in current:
                        continue
                    # Check if b_add covers all uncovered_subsets
                    covers_all = True
                    for s_idx in uncovered_subsets:
                        if s_idx not in block_subsets[b_add]:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Calculate gain: how many NEW subsets does it cover?
                    # Gain is not strictly necessary for validity, but helps quality
                    # For minimization, we just need a valid swap.
                    # We can prioritize blocks that cover more *other* uncovered sets?
                    # Actually, since we are swapping 1 for 1, the size stays same.
                    # We just need to maintain coverage.
                    # A "better" swap might be one that increases redundancy,
                    # making future removals easier.
                    gain = 0
                    for s_idx in block_subsets[b_add]:
                        if counts[s_idx] == 0: # These will become covered
                            gain += 1
                        # Note: subsets in uncovered_subsets are currently 1 (covered only by b_remove)
                        # After removal, they become 0. b_add covers them, so they become 1.
                        # So gain calculation needs care.
                        
                    # Simple heuristic: just take the first valid one
                    best_add = b_add
                    break # Found a valid swap, take it to save time? Or find best?
                    # Let's find the one with max gain to improve structure
                    # if gain > best_gain:
                    #    best_gain = gain
                    #    best_add = b_add
                    # For speed, break on first found valid swap is often good enough for local search
                    # But let's check a few.
                    
                if best_add != -1:
                    # Perform swap
                    current[i] = best_add
                    # Update counts
                    for s_idx in block_subsets[b_remove]:
                        counts[s_idx] -= 1
                    for s_idx in block_subsets[best_add]:
                        counts[s_idx] += 1
                    improved = True
                    break

        return current

    def greedy_construction():
        uncovered = full_mask
        current_blocks = []
        
        while uncovered:
            # Pick a random uncovered subset
            found_idx = -1
            attempts = 0
            while found_idx == -1 and attempts < 20:
                idx = random.randint(0, n_t - 1)
                if (uncovered >> idx) & 1:
                    found_idx = idx
                attempts += 1
            
            if found_idx == -1:
                for idx in range(n_t):
                    if (uncovered >> idx) & 1:
                        found_idx = idx
                        break
            
            if found_idx == -1:
                break

            candidates = blocks_containing[found_idx]
            
            # Calculate gains
            best_gain = -1
            best_block_idx = -1
            # To add randomness, we can pick randomly from top 3
            top_blocks = []
            
            for b_idx in candidates:
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                # Keep track of top few
                # This loop is slow. Optimize:
                # We can just pick the best.
                pass
            
            # Use best block
            if best_block_idx != -1:
                current_blocks.append(best_block_idx)
                uncovered &= ~block_masks[best_block_idx]
                
        return current_blocks

    # Main Loop
    runs = 0
    while time.time() - start_time < DEADLINE_SECONDS - 5:
        runs += 1
        
        # Greedy
        current_blocks = greedy_construction()
        
        # Local Search
        current_blocks = local_search(current_blocks, DEADLINE_SECONDS - 5)
        
        # Check if valid and better
        valid, count = get_solution_count(current_blocks)
        if valid and count < best_count:
            best_count = count
            best_blocks = current_blocks[:]
            
        # Optional: Perturbation
        if runs % 5 == 0 and best_blocks:
            # Try to improve best_blocks further
            perturbed = best_blocks[:]
            # Remove 2 random blocks
            if len(perturbed) > 10:
                remove_indices = random.sample(range(len(perturbed)), 2)
                for idx in sorted(remove_indices, reverse=True):
                    perturbed.pop(idx)
                # Re-greedy fill
                uncovered = full_mask
                for b_idx in perturbed:
                    uncovered &= ~block_masks[b_idx]
                
                while uncovered:
                    found_idx = -1
                    attempts = 0
                    while found_idx == -1 and attempts < 20:
                        idx = random.randint(0, n_t - 1)
                        if (uncovered >> idx) & 1:
                            found_idx = idx
                        attempts += 1
                    if found_idx == -1:
                        for idx in range(n_t):
                            if (uncovered >> idx) & 1:
                                found_idx = idx
                                break
                    if found_idx == -1:
                        break
                    candidates = blocks_containing[found_idx]
                    best_gain = -1
                    best_block_idx = -1
                    for b_idx in candidates:
                        gain = (block_masks[b_idx] & uncovered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_block_idx = b_idx
                    if best_block_idx != -1:
                        perturbed.append(best_block_idx)
                        uncovered &= ~block_masks[best_block_idx]
                
                # Local search on perturbed
                perturbed = local_search(perturbed, DEADLINE_SECONDS - 5)
                valid, count = get_solution_count(perturbed)
                if valid and count < best_count:
                    best_count = count
                    best_blocks = perturbed[:]

    return [list(all_blocks[i]) for i in best_blocks]
