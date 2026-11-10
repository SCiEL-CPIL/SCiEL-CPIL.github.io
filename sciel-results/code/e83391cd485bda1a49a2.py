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
    # This is expensive for large v but v=19, t=4 -> C(19,4)=3876, manageable
    point_to_subs = [[] for _ in range(v)]
    for sub_idx, sub in enumerate(all_subs):
        for x in sub:
            point_to_subs[x].append(sub_idx)
    
    # For faster gain computation, we need to know which t-subsets a partial block covers
    # Strategy: for a partial block P (size p), the t-subsets it covers are those t-subsets
    # that are subsets of P. We can precompute for each combination of points.
    
    # Alternative approach: for each candidate point x, the gain from adding x to current block B
    # is the number of uncovered t-subsets that contain x and are subsets of B ∪ {x}.
    # A t-subset S is a subset of B ∪ {x} iff S ⊆ B or (S \ {x} ⊆ B and x ∈ S).
    # So gain(x) = |{S : S ⊆ B ∪ {x}, x ∈ S, not covered[S]}|
    #            = |{S : S \ {x} ⊆ B, S not covered, x ∈ S}|
    
    # For efficiency, we'll compute gain by iterating over t-1 subsets of B
    # and checking if adding x completes them.
    
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
                # A t-subset S is covered by B∪{x} iff S⊆B∪{x} and x∈S
                # i.e., S\{x} ⊆ B and |S\{x}| = t-1
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
    Enhanced local search with simulated annealing.
    Uses a constraint-satisfaction model: maximize covered t-subsets
    with a fixed block budget, then try to reduce the budget.
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
    
    def compute_uncovered():
        return int(np.sum(coverage == 0))
    
    # Phase 1: Annealing to improve coverage density within fixed block count
    # Then try to remove blocks when coverage allows
    
    temp = 100.0
    cooling = 0.9995
    min_temp = 0.01
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline and temp > min_temp and no_improve < max_no_improve:
        move_type = rng.randint(0, 10)
        
        if move_type < 3 and len(current_blocks) > 1:
            # Try to remove a block (only if it doesn't hurt coverage)
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
                    no_improve = 0
                else:
                    no_improve += 1
                
                temp *= cooling
                continue
            else:
                no_improve += 1
                temp *= cooling
                continue
        
        elif move_type < 7:
            # Point swap within a block: replace one point with another
            if len(current_blocks) == 0:
                temp *= cooling
                continue
            
            idx = rng.randint(0, len(current_blocks))
            old_block = list(current_blocks[idx])
            old_subs = get_block_subs(old_block)
            
            # Pick a position to change
            pos = rng.randint(0, k)
            old_val = old_block[pos]
            
            # Try all possible replacements
            best_new_val = None
            best_delta = 0
            
            for new_val in range(v):
                if new_val == old_val or new_val in old_block:
                    continue
                
                new_block = old_block.copy()
                new_block[pos] = new_val
                new_block.sort()
                new_subs = get_block_subs(new_block)
                new_sub_set = set(new_subs)
                
                # Compute delta in coverage
                delta = 0
                for sub in old_subs:
                    sidx = sub_to_idx[sub]
                    if coverage[sidx] == 1 and sub not in new_sub_set:
                        delta -= 100  # Big penalty for losing coverage
                for sub in new_subs:
                    sidx = sub_to_idx[sub]
                    if coverage[sidx] == 0:
                        delta += 1
                
                if delta > best_delta:
                    best_delta = delta
                    best_new_val = new_val
            
            if best_new_val is not None and best_delta > 0:
                new_block = old_block.copy()
                new_block[pos] = best_new_val
                new_block.sort()
                new_subs = get_block_subs(new_block)
                
                for sub in old_subs:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] -= 1
                for sub in new_subs:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1
                current_blocks[idx] = new_block
                
                no_improve = 0
            else:
                no_improve += 1
            
            temp *= cooling
            continue
        
        elif move_type < 9:
            # Pair swap: swap two points between two blocks
            if len(current_blocks) < 2:
                temp *= cooling
                continue
            
            idx1 = rng.randint(0, len(current_blocks))
            idx2 = rng.randint(0, len(current_blocks))
            if idx1 == idx2:
                temp *= cooling
                continue
            
            block1 = list(current_blocks[idx1])
            block2 = list(current_blocks[idx2])
            subs1 = get_block_subs(block1)
            subs2 = get_block_subs(block2)
            
            # Pick a point from each block to swap
            pos1 = rng.randint(0, k)
            pos2 = rng.randint(0, k)
            val1 = block1[pos1]
            val2 = block2[pos2]
            
            if val1 == val2:
                temp *= cooling
                continue
            
            # Check if swap is valid (no duplicates)
            if val2 in block1 or val1 in block2:
                temp *= cooling
                continue
            
            new_block1 = block1.copy()
            new_block2 = block2.copy()
            new_block1[pos1] = val2
            new_block2[pos2] = val1
            new_block1.sort()
            new_block2.sort()
            
            new_subs1 = get_block_subs(new_block1)
            new_subs2 = get_block_subs(new_block2)
            new_sub_set1 = set(new_subs1)
            new_sub_set2 = set(new_subs2)
            
            # Compute delta
            delta = 0
            for sub in subs1:
                sidx = sub_to_idx[sub]
                if coverage[sidx] == 1 and sub not in new_sub_set1:
                    delta -= 100
            for sub in subs2:
                sidx = sub_to_idx[sub]
                if coverage[sidx] == 1 and sub not in new_sub_set2:
                    delta -= 100
            for sub in new_subs1:
                sidx = sub_to_idx[sub]
                if coverage[sidx] == 0:
                    delta += 1
            for sub in new_subs2:
                sidx = sub_to_idx[sub]
                if coverage[sidx] == 0:
                    delta += 1
            
            if delta > 0:
                for sub in subs1:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] -= 1
                for sub in subs2:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] -= 1
                for sub in new_subs1:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1
                for sub in new_subs2:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1
                current_blocks[idx1] = new_block1
                current_blocks[idx2] = new_block2
                no_improve = 0
            else:
                no_improve += 1
            
            temp *= cooling
            continue
        
        else:
            # Random restart: replace a block with a completely new one
            if len(current_blocks) == 0:
                temp *= cooling
                continue
            
            idx = rng.randint(0, len(current_blocks))
            old_block = current_blocks[idx]
            old_subs = get_block_subs(old_block)
            
            # Check if we can remove old_block without losing coverage
            can_remove = True
            for sub in old_subs:
                sidx = sub_to_idx[sub]
                if coverage[sidx] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                for sub in old_subs:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] -= 1
                
                # Generate a new block greedily
                new_block = []
                new_block_set = set()
                while len(new_block) < k:
                    best_point = -1
                    best_gain = -1
                    for x in range(v):
                        if x in new_block_set:
                            continue
                        gain = 0
                        for rest in itertools.combinations(new_block, t - 1):
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
                    new_block.append(best_point)
                    new_block_set.add(best_point)
                
                new_block.sort()
                new_subs = get_block_subs(new_block)
                
                for sub in new_subs:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1
                current_blocks[idx] = new_block
                no_improve = 0
            else:
                no_improve += 1
            
            temp *= cooling
    
    return best_blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for final validation
    
    # Step 1: Greedy construction
    blocks = _greedy_construct(v, k, t, deadline)
    
    # Step 2: Remove redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, t, deadline)
    
    # Step 3: Local search with multiple restarts
    if time.time() < deadline - 15:
        # Run local search with different seeds to explore different regions
        best_result = blocks
        best_size = len(blocks)
        
        for seed in [42, 123, 456, 789, 1024]:
            if time.time() > deadline - 10:
                break
            if len(best_result) <= 1:
                break
            
            result = _local_search(blocks, v, k, t, deadline - 5, seed=seed)
            if len(result) < best_size:
                best_result = result
                best_size = len(result)
        
        blocks = best_result
    
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
