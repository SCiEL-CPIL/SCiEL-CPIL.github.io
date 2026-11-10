"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _encode_subset(subset, v, t):
    """Encode a t-subset as a unique integer index."""
    # Use combinatorial number system
    code = 0
    for i in range(t):
        # C(subset[i], i)
        s = 0
        for j in range(i + 1):
            s += 1
        # Simpler: use a lookup or direct computation
        pass
    # Alternative: use a hash-based approach with a dict
    return tuple(sorted(subset))


def _get_all_t_subsets(v, t):
    """Generate all t-subsets of range(v)."""
    return list(itertools.combinations(range(v), t))


def _build_coverage_matrix(blocks, v, t):
    """Build a coverage matrix: for each t-subset, how many blocks cover it."""
    all_subs = _get_all_t_subsets(v, t)
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    coverage = np.zeros(n_subs, dtype=np.int32)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            coverage[idx] += 1
    return coverage, sub_to_idx, all_subs


def _greedy_construct(v, k, t, deadline):
    """Fast greedy covering algorithm."""
    all_subs = _get_all_t_subsets(v, t)
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    covered = np.zeros(n_subs, dtype=bool)
    blocks = []
    
    while not covered.all():
        # Find first uncovered subset
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
        start_idx = uncovered_indices[0]
        start_sub = all_subs[start_idx]
        
        # Build block greedily starting from start_sub
        block = list(start_sub)
        
        while len(block) < k:
            # Find point that covers most uncovered t-subsets
            best_point = -1
            best_gain = -1
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in sub_to_idx:
                        idx = sub_to_idx[candidate]
                        if not covered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                break
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update coverage
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            covered[idx] = True
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return blocks


def _remove_redundant_blocks(blocks, v, t, deadline):
    """Remove blocks that are fully covered by other blocks."""
    if len(blocks) <= 1:
        return blocks
    
    all_subs = _get_all_t_subsets(v, t)
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    # Count coverage for each subset
    coverage = np.zeros(n_subs, dtype=np.int32)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            coverage[idx] += 1
    
    new_blocks = []
    for i, block in enumerate(blocks):
        # Check if this block has any unique t-subset
        has_unique = False
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            if coverage[idx] == 1:
                has_unique = True
                break
        
        if not has_unique:
            # Remove this block and update coverage
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx[sub]
                coverage[idx] -= 1
        else:
            new_blocks.append(block)
        
        if time.time() > deadline:
            return new_blocks + blocks[i + 1:]
    
    return new_blocks


def _local_search_sa(blocks, v, k, t, deadline, seed=42):
    """Simulated annealing to reduce block count."""
    rng = np.random.RandomState(seed)
    current_blocks = list(blocks)
    best_blocks = list(blocks)
    best_size = len(current_blocks)
    
    all_subs = _get_all_t_subsets(v, t)
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    # Build coverage
    coverage = np.zeros(n_subs, dtype=np.int32)
    for block in current_blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            coverage[idx] += 1
    
    temp = 1.0
    cooling = 0.995
    min_temp = 0.01
    
    iteration = 0
    while time.time() < deadline and temp > min_temp:
        # Choose a random move
        move_type = rng.randint(0, 3)
        
        if move_type == 0 and len(current_blocks) > 1:
            # Remove a block
            idx = rng.randint(0, len(current_blocks))
            block = current_blocks[idx]
            
            # Check if removing it leaves coverage valid
            can_remove = True
            for sub in itertools.combinations(block, t):
                sidx = sub_to_idx[sub]
                if coverage[sidx] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                for sub in itertools.combinations(block, t):
                    sidx = sub_to_idx[sub]
                    coverage[sidx] -= 1
                current_blocks.pop(idx)
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
                
                # Always accept removals
                temp *= cooling
                iteration += 1
                continue
            else:
                # Reject
                temp *= cooling
                iteration += 1
                continue
        
        elif move_type == 1:
            # Add a random block
            if len(current_blocks) < 151:  # Don't exceed known best
                # Pick a random block that covers many uncovered subsets
                uncovered_mask = coverage == 0
                if not uncovered_mask.any():
                    temp *= cooling
                    iteration += 1
                    continue
                
                # Try to find a good block to add
                best_block = None
                best_gain = -1
                for _ in range(10):  # Try a few random blocks
                    block = sorted(rng.choice(v, size=k, replace=False).tolist())
                    gain = 0
                    for sub in itertools.combinations(block, t):
                        sidx = sub_to_idx[sub]
                        if coverage[sidx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = block
                
                if best_block is not None and best_gain > 0:
                    for sub in itertools.combinations(best_block, t):
                        sidx = sub_to_idx[sub]
                        coverage[sidx] += 1
                    current_blocks.append(best_block)
                
                temp *= cooling
                iteration += 1
                continue
        
        elif move_type == 2 and len(current_blocks) > 1:
            # Swap a block: remove one, add another
            remove_idx = rng.randint(0, len(current_blocks))
            remove_block = current_blocks[remove_idx]
            
            # Check if we can remove
            can_remove = True
            for sub in itertools.combinations(remove_block, t):
                sidx = sub_to_idx[sub]
                if coverage[sidx] <= 1:
                    can_remove = False
                    break
            
            if not can_remove:
                temp *= cooling
                iteration += 1
                continue
            
            # Find a replacement block
            # Temporarily remove
            for sub in itertools.combinations(remove_block, t):
                sidx = sub_to_idx[sub]
                coverage[sidx] -= 1
            
            # Find best block to add
            best_block = None
            best_gain = -1
            for _ in range(10):
                block = sorted(rng.choice(v, size=k, replace=False).tolist())
                gain = 0
                for sub in itertools.combinations(block, t):
                    sidx = sub_to_idx[sub]
                    if coverage[sidx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is not None:
                for sub in itertools.combinations(best_block, t):
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1
                current_blocks[remove_idx] = best_block
                
                # Check if valid
                if not (coverage >= 0).all():
                    # Roll back
                    for sub in itertools.combinations(best_block, t):
                        sidx = sub_to_idx[sub]
                        coverage[sidx] -= 1
                    current_blocks[remove_idx] = remove_block
                    for sub in itertools.combinations(remove_block, t):
                        sidx = sub_to_idx[sub]
                        coverage[sidx] += 1
                else:
                    new_size = len(current_blocks)
                    if new_size < best_size:
                        best_blocks = list(current_blocks)
                        best_size = new_size
            else:
                # Roll back
                for sub in itertools.combinations(remove_block, t):
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1
            
            temp *= cooling
            iteration += 1
            continue
        
        else:
            # Perturb: modify a random block
            if len(current_blocks) == 0:
                temp *= cooling
                iteration += 1
                continue
            
            idx = rng.randint(0, len(current_blocks))
            old_block = current_blocks[idx]
            
            # Create a new block by changing 1-2 elements
            new_block = list(old_block)
            num_changes = rng.randint(1, min(2, k))
            for _ in range(num_changes):
                pos = rng.randint(0, k)
                new_val = rng.randint(0, v)
                if new_val not in new_block:
                    new_block[pos] = new_val
            
            new_block.sort()
            
            if new_block == old_block:
                temp *= cooling
                iteration += 1
                continue
            
            # Calculate delta in coverage
            # This is complex; for now, just check validity
            # Temporarily replace
            current_blocks[idx] = new_block
            
            # Check if coverage is still valid
            valid = True
            for sub in itertools.combinations(new_block, t):
                sidx = sub_to_idx[sub]
                if coverage[sidx] < 1:
                    valid = False
                    break
            
            if valid:
                # Update coverage
                for sub in itertools.combinations(old_block, t):
                    sidx = sub_to_idx[sub]
                    coverage[sidx] -= 1
                for sub in itertools.combinations(new_block, t):
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
            else:
                # Roll back
                current_blocks[idx] = old_block
            
            temp *= cooling
            iteration += 1
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return best_blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Step 1: Greedy construction
    blocks = _greedy_construct(v, k, t, deadline)
    
    # Step 2: Remove redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, t, deadline)
    
    # Step 3: Local search with simulated annealing
    if time.time() < deadline - 5:
        blocks = _local_search_sa(blocks, v, k, t, deadline, seed=42)
    
    # Final cleanup: remove any redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, t, deadline)
    
    # Ensure validity: check all t-subsets are covered
    all_subs = _get_all_t_subsets(v, t)
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    coverage = np.zeros(n_subs, dtype=np.int32)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            coverage[idx] += 1
    
    # If not fully covered, add more blocks greedily
    while not (coverage > 0).all() and time.time() < deadline:
        uncovered_idx = np.where(coverage == 0)[0][0]
        sub = all_subs[uncovered_idx]
        block = list(sub)
        while len(block) < k:
            best_point = -1
            best_gain = -1
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in sub_to_idx:
                        sidx = sub_to_idx[candidate]
                        if coverage[sidx] == 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point == -1:
                break
            block.append(best_point)
        block.sort()
        blocks.append(block)
        for sub2 in itertools.combinations(block, t):
            sidx = sub_to_idx[sub2]
            coverage[sidx] += 1
    
    return blocks
# EVOLVE-BLOCK-END
