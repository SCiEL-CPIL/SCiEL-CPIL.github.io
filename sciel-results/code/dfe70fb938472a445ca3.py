"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    deadline = time.time() + 140  # 140s safety margin
    
    # Precompute all t-subsets as bitmasks for fast coverage checking
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For each point, precompute which t-subsets it participates in
    point_subsets = [[] for _ in range(v)]
    for i, s in enumerate(all_t_subsets):
        for p in s:
            point_subsets[p].append(i)
    
    # Coverage bitset: bit i set means subset i is covered
    covered = 0
    total_mask = (1 << n_subsets) - 1 if n_subsets < 64 else None
    
    # For large n_subsets, use a set instead of bitset
    use_bitset = n_subsets <= 60
    if use_bitset:
        uncovered_count = n_subsets
    else:
        uncovered_set = set(range(n_subsets))
        uncovered_count = n_subsets
    
    blocks = []
    
    # Greedy construction
    while uncovered_count > 0:
        if time.time() > deadline - 30:
            break
        
        # Find an uncovered subset to seed a new block
        if use_bitset:
            # Find first uncovered subset
            mask = ~covered & total_mask
            first_idx = mask & (-mask)
            first_idx = first_idx.bit_length() - 1
            seed = list(all_t_subsets[first_idx])
        else:
            seed = list(all_t_subsets[next(iter(uncovered_set))])
        
        # Build block greedily
        block = seed[:]
        
        while len(block) < k:
            if time.time() > deadline - 25:
                break
            
            # For each candidate point not in block, compute how many new subsets it covers
            best_point = None
            best_gain = -1
            best_new_mask = 0
            
            for x in range(v):
                if x in block:
                    continue
                
                # Compute which subsets would be newly covered by adding x
                # A subset {s1,...,s_{t-1}, x} is newly covered if it's uncovered
                # and {s1,...,s_{t-1}} ⊆ block
                new_mask = 0
                gain = 0
                
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    idx = subset_to_idx[sub]
                    if use_bitset:
                        bit = 1 << idx
                        if not (covered & bit):
                            new_mask |= bit
                            gain += 1
                    else:
                        if idx in uncovered_set:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    best_new_mask = new_mask
            
            if best_point is None or best_gain <= 0:
                # Pick any point not in block
                for x in range(v):
                    if x not in block:
                        best_point = x
                        break
            
            block.append(best_point)
        
        if len(block) < k:
            # Fill remaining with arbitrary points
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block:
                    block.append(x)
        
        block.sort()
        blocks.append(block)
        
        # Update coverage
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            if use_bitset:
                if not (covered & (1 << idx)):
                    covered |= (1 << idx)
                    uncovered_count -= 1
            else:
                if idx in uncovered_set:
                    uncovered_set.discard(idx)
                    uncovered_count -= 1
        
        if uncovered_count == 0:
            break
    
    # Check if we have a valid covering
    if uncovered_count > 0:
        # Emergency: add more blocks to cover remaining subsets
        while uncovered_count > 0 and time.time() < deadline - 5:
            if use_bitset:
                mask = ~covered & total_mask
                first_idx = mask & (-mask)
                first_idx = first_idx.bit_length() - 1
                seed = list(all_t_subsets[first_idx])
            else:
                seed = list(all_t_subsets[next(iter(uncovered_set))])
            
            block = seed[:]
            while len(block) < k:
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
            block.sort()
            blocks.append(block)
            
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                if use_bitset:
                    if not (covered & (1 << idx)):
                        covered |= (1 << idx)
                        uncovered_count -= 1
                else:
                    if idx in uncovered_set:
                        uncovered_set.discard(idx)
                        uncovered_count -= 1
    
    # Local search: try to remove or swap blocks to reduce count
    if time.time() < deadline - 10:
        blocks = _local_search(blocks, v, k, t, deadline)
    
    return blocks


def _local_search(blocks, v, k, t, deadline):
    """Try to reduce block count by removing redundant blocks or swapping."""
    subset_to_idx = {}
    all_t_subsets = list(itertools.combinations(range(v), t))
    for i, s in enumerate(all_t_subsets):
        subset_to_idx[s] = i
    n_subsets = len(all_t_subsets)
    use_bitset = n_subsets <= 60
    
    # Count how many blocks cover each subset
    if use_bitset:
        cover_count = [0] * n_subsets
        for block in blocks:
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                cover_count[idx] += 1
    else:
        from collections import defaultdict
        cover_count = defaultdict(int)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                cover_count[idx] += 1
    
    # Try to remove blocks that don't uniquely cover any subset
    removed = set()
    for i in range(len(blocks)):
        if time.time() > deadline - 5:
            break
        if i in removed:
            continue
        
        # Check if block i is the only cover for any subset
        is_critical = False
        for sub in itertools.combinations(blocks[i], t):
            idx = subset_to_idx[sub]
            if cover_count[idx] == 1:
                is_critical = True
                break
        
        if not is_critical:
            # Remove block i
            removed.add(i)
            for sub in itertools.combinations(blocks[i], t):
                idx = subset_to_idx[sub]
                cover_count[idx] -= 1
    
    if removed:
        blocks = [b for i, b in enumerate(blocks) if i not in removed]
    
    return blocks
# EVOLVE-BLOCK-END
