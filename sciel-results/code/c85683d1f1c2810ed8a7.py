"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _greedy_construct(v, k, t, deadline):
    """
    Fast greedy covering algorithm using vectorized gain computation.
    State: covered[t-subset-index] = True/False
    For each new block, pick the k points that maximize newly covered t-subsets.
    """
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    covered = np.zeros(n_subs, dtype=bool)
    blocks = []

    # Precompute: for each point x, list of t-subsets containing x (as indices)
    point_to_subs = [[] for _ in range(v)]
    for sub_idx, sub in enumerate(all_subs):
        for x in sub:
            point_to_subs[x].append(sub_idx)
    
    while not covered.all():
        if time.time() > deadline:
            break
        
        # Find first uncovered subset
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
        start_idx = uncovered_indices[0]
        start_sub = all_subs[start_idx]
        
        # Build block greedily starting from start_sub
        block = list(start_sub)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_point = -1
            best_gain = -1
            
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
            block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update coverage
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            covered[idx] = True
    
    return blocks


def _remove_redundant_blocks(blocks, v, t, deadline):
    """Remove blocks that don't uniquely cover any t-subset."""
    if len(blocks) <= 1:
        return blocks
    
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    coverage = np.zeros(n_subs, dtype=np.int32)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            coverage[idx] += 1
    
    new_blocks = []
    for i, block in enumerate(blocks):
        has_unique = False
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            if coverage[idx] == 1:
                has_unique = True
                break
        
        if not has_unique:
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx[sub]
                coverage[idx] -= 1
        else:
            new_blocks.append(block)
        
        if time.time() > deadline:
            return new_blocks + blocks[i + 1:]
    
    return new_blocks


def _local_search(blocks, v, k, t, deadline, seed=42):
    """
    Local search to reduce block count.
    Moves: remove a redundant block, or swap a block for a better one.
    """
    rng = np.random.RandomState(seed)
    current_blocks = list(blocks)
    best_blocks = list(current_blocks)
    best_size = len(current_blocks)
    
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    coverage = np.zeros(n_subs, dtype=np.int32)
    for block in current_blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            coverage[idx] += 1
    
    # Precompute block_subsets for each block
    def get_block_subs(block):
        return list(itertools.combinations(block, t))
    
    temp = 1.0
    cooling = 0.999
    min_temp = 0.001
    
    while time.time() < deadline and temp > min_temp:
        move_type = rng.randint(0, 2)
        
        if move_type == 0 and len(current_blocks) > 1:
            # Try to remove a block
            idx = rng.randint(0, len(current_blocks))
            block = current_blocks[idx]
            block_subs = get_block_subs(block)
            
            can_remove = True
            for sub in block_subs:
                sidx = sub_to_idx[sub]
                if coverage[sidx] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                for sub in block_subs:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] -= 1
                current_blocks.pop(idx)
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
                
                temp *= cooling
                continue
            else:
                temp *= cooling
                continue
        
        elif move_type == 1 and len(current_blocks) > 1:
            # Swap: replace a block with a better one
            idx = rng.randint(0, len(current_blocks))
            old_block = current_blocks[idx]
            old_subs = get_block_subs(old_block)
            
            # Check if we can remove old_block
            can_remove = True
            for sub in old_subs:
                sidx = sub_to_idx[sub]
                if coverage[sidx] <= 1:
                    can_remove = False
                    break
            
            if not can_remove:
                temp *= cooling
                continue
            
            # Temporarily remove
            for sub in old_subs:
                sidx = sub_to_idx[sub]
                coverage[sidx] -= 1
            
            # Find a good replacement block
            best_block = None
            best_gain = -1
            
            # Try several random blocks
            for _ in range(5):
                candidate = sorted(rng.choice(v, size=k, replace=False).tolist())
                gain = 0
                for sub in itertools.combinations(candidate, t):
                    sidx = sub_to_idx[sub]
                    if coverage[sidx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate
            
            if best_block is not None and best_gain > 0:
                for sub in itertools.combinations(best_block, t):
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1
                current_blocks[idx] = best_block
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
            else:
                # Roll back
                for sub in old_subs:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1
            
            temp *= cooling
            continue
        
        else:
            # Perturb: modify a random block
            if len(current_blocks) == 0:
                temp *= cooling
                continue
            
            idx = rng.randint(0, len(current_blocks))
            old_block = current_blocks[idx]
            old_subs = get_block_subs(old_block)
            
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
                continue
            
            # Check if replacing old_block with new_block maintains coverage
            new_subs = get_block_subs(new_block)
            new_sub_set = set(new_subs)
            
            valid = True
            for sub in old_subs:
                sidx = sub_to_idx[sub]
                if coverage[sidx] <= 1 and sub not in new_sub_set:
                    valid = False
                    break
            
            if valid:
                for sub in old_subs:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] -= 1
                for sub in new_subs:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1
                current_blocks[idx] = new_block
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
            
            temp *= cooling
    
    return best_blocks


def _sa_search(blocks, v, k, t, deadline, seed=123):
    """
    Simulated annealing with more aggressive exploration.
    State: set of blocks. Moves: remove redundant, swap, perturb.
    Accept worse solutions with probability exp(-delta/T) to escape local minima.
    """
    rng = np.random.RandomState(seed)
    current_blocks = list(blocks)
    best_blocks = list(current_blocks)
    best_size = len(current_blocks)
    
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    coverage = np.zeros(n_subs, dtype=np.int32)
    for block in current_blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            coverage[idx] += 1
    
    def get_block_subs(block):
        return list(itertools.combinations(block, t))
    
    # Simulated annealing parameters
    temp = 5.0
    cooling = 0.995
    min_temp = 0.01
    max_iter = 5000
    
    iteration = 0
    while time.time() < deadline and temp > min_temp and iteration < max_iter:
        iteration += 1
        move_type = rng.randint(0, 3)
        
        if move_type == 0 and len(current_blocks) > 1:
            # Try to remove a block
            idx = rng.randint(0, len(current_blocks))
            block = current_blocks[idx]
            block_subs = get_block_subs(block)
            
            can_remove = True
            for sub in block_subs:
                sidx = sub_to_idx[sub]
                if coverage[sidx] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                for sub in block_subs:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] -= 1
                current_blocks.pop(idx)
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
                
                temp *= cooling
                continue
            else:
                temp *= cooling
                continue
        
        elif move_type == 1 and len(current_blocks) > 1:
            # Swap: replace a block with a better one
            idx = rng.randint(0, len(current_blocks))
            old_block = current_blocks[idx]
            old_subs = get_block_subs(old_block)
            
            can_remove = True
            for sub in old_subs:
                sidx = sub_to_idx[sub]
                if coverage[sidx] <= 1:
                    can_remove = False
                    break
            
            if not can_remove:
                temp *= cooling
                continue
            
            for sub in old_subs:
                sidx = sub_to_idx[sub]
                coverage[sidx] -= 1
            
            best_block = None
            best_gain = -1
            
            # Try more random blocks for better exploration
            for _ in range(10):
                candidate = sorted(rng.choice(v, size=k, replace=False).tolist())
                gain = 0
                for sub in itertools.combinations(candidate, t):
                    sidx = sub_to_idx[sub]
                    if coverage[sidx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate
            
            if best_block is not None and best_gain > 0:
                for sub in itertools.combinations(best_block, t):
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1
                current_blocks[idx] = best_block
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
            else:
                for sub in old_subs:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1
            
            temp *= cooling
            continue
        
        elif move_type == 2:
            # Perturb: modify a random block more aggressively
            if len(current_blocks) == 0:
                temp *= cooling
                continue
            
            idx = rng.randint(0, len(current_blocks))
            old_block = current_blocks[idx]
            old_subs = get_block_subs(old_block)
            
            # Create a new block by changing 1-3 elements
            new_block = list(old_block)
            num_changes = rng.randint(1, min(3, k))
            for _ in range(num_changes):
                pos = rng.randint(0, k)
                new_val = rng.randint(0, v)
                if new_val not in new_block:
                    new_block[pos] = new_val
            
            new_block.sort()
            
            if new_block == old_block:
                temp *= cooling
                continue
            
            new_subs = get_block_subs(new_block)
            new_sub_set = set(new_subs)
            
            valid = True
            for sub in old_subs:
                sidx = sub_to_idx[sub]
                if coverage[sidx] <= 1 and sub not in new_sub_set:
                    valid = False
                    break
            
            if valid:
                for sub in old_subs:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] -= 1
                for sub in new_subs:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1
                current_blocks[idx] = new_block
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
            
            temp *= cooling
            continue
        
        else:
            # Add a new block (to escape local minima by increasing size temporarily)
            if len(current_blocks) < best_size + 5:
                # Find a good block to add
                best_block = None
                best_gain = -1
                
                for _ in range(10):
                    candidate = sorted(rng.choice(v, size=k, replace=False).tolist())
                    gain = 0
                    for sub in itertools.combinations(candidate, t):
                        sidx = sub_to_idx[sub]
                        if coverage[sidx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = candidate
                
                if best_block is not None and best_gain > 0:
                    for sub in itertools.combinations(best_block, t):
                        sidx = sub_to_idx[sub]
                        coverage[sidx] += 1
                    current_blocks.append(best_block)
                    
                    new_size = len(current_blocks)
                    if new_size < best_size:
                        best_blocks = list(current_blocks)
                        best_size = new_size
                
                temp *= cooling
                continue
            else:
                temp *= cooling
                continue
    
    return best_blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for final validation
    
    # Step 1: Greedy construction
    blocks = _greedy_construct(v, k, t, deadline)
    
    # Step 2: Remove redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, t, deadline)
    
    # Step 3: Local search (first pass)
    if time.time() < deadline - 20:
        blocks = _local_search(blocks, v, k, t, deadline - 10, seed=42)
    
    # Step 4: Simulated annealing (second pass with different seed)
    if time.time() < deadline - 10:
        blocks = _sa_search(blocks, v, k, t, deadline - 5, seed=123)
    
    # Final cleanup: remove any redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, t, deadline)
    
    # Ensure validity: check all t-subsets are covered
    all_subs = list(itertools.combinations(range(v), t))
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
        block_set = set(block)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
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
            block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        for sub2 in itertools.combinations(block, t):
            sidx = sub_to_idx[sub2]
            coverage[sidx] += 1
    
    return blocks
# EVOLVE-BLOCK-END
