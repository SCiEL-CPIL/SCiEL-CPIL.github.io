# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _fast_greedy(v, k, t, rng, deadline):
    """
    Fast greedy covering design construction using NumPy vectorization.
    Returns list of blocks (each a sorted list of k integers).
    """
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute masks for t-subsets
    tsub_masks = np.zeros(n_tsubs, dtype=np.int32)
    for i, ts in enumerate(all_tsubs):
        mask = 0
        for x in ts:
            mask |= (1 << x)
        tsub_masks[i] = mask
        
    # Precompute for each point x, the indices of t-subsets containing x
    point_to_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for x in ts:
            point_to_tsubs[x].append(i)
    point_to_tsubs_arr = [np.array(pts, dtype=np.int32) for pts in point_to_tsubs]
    
    uncovered = np.ones(n_tsubs, dtype=bool)
    blocks = []
    
    while np.any(uncovered):
        if time.time() > deadline - 2.0:
            break
        
        # Pick a random uncovered t-subset to start
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        start_idx = rng.choice(uncovered_indices)
        start = all_tsubs[start_idx]
        block = list(start)
        block_set = set(block)
        block_mask = 0
        for x in block:
            block_mask |= (1 << x)
        
        while len(block) < k:
            if time.time() > deadline - 2.0:
                break
            
            best_x = -1
            best_gain = -1
            
            # Get all (t-1)-subsets of block? No, we use masks.
            # For each candidate x not in block, compute gain
            candidates = [x for x in range(v) if x not in block_set]
            
            for x in candidates:
                new_mask = block_mask | (1 << x)
                idxs = point_to_tsubs_arr[x]
                if len(idxs) == 0:
                    continue
                
                masks = tsub_masks[idxs]
                # Check which of these t-subsets are subsets of new_mask
                # (masks & ~new_mask) == 0
                mask_ok = (masks & ~new_mask) == 0
                # And are currently uncovered
                unc = uncovered[idxs]
                
                gain = np.sum(unc & mask_ok)
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain <= 0:
                # Pick a random point not in block
                if not candidates:
                    break
                best_x = rng.choice(candidates)
            
            block.append(best_x)
            block_set.add(best_x)
            block_mask |= (1 << best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        # Find all t-subsets of the new block
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            uncovered[idx] = False
            
    return blocks


def _local_search(v, k, t, blocks, deadline, rng):
    """
    Local search: try to remove blocks that are redundant.
    """
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Build coverage count
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    block_tsubs = []
    for block in blocks:
        ts_list = list(itertools.combinations(block, t))
        block_tsubs.append(ts_list)
        for ts in ts_list:
            cover_count[tsub_to_idx[ts]] += 1
    
    removed = [False] * len(blocks)
    improved = True
    
    while improved and time.time() < deadline - 5.0:
        improved = False
        active = [i for i in range(len(blocks)) if not removed[i]]
        rng.shuffle(active)
        
        for bi in active:
            if removed[bi]:
                continue
            if time.time() > deadline - 5.0:
                break
            
            # Check if block bi is redundant
            is_redundant = True
            for ts in block_tsubs[bi]:
                idx = tsub_to_idx[ts]
                if cover_count[idx] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block bi
                removed[bi] = True
                for ts in block_tsubs[bi]:
                    cover_count[tsub_to_idx[ts]] -= 1
                improved = True
            else:
                # Try to find a replacement block
                unique_subs = []
                for ts in block_tsubs[bi]:
                    idx = tsub_to_idx[ts]
                    if cover_count[idx] == 1:
                        unique_subs.append(ts)
                
                if not unique_subs:
                    continue
                
                required = set()
                for ts in unique_subs:
                    required.update(ts)
                
                if len(required) > k:
                    continue
                
                need_more = k - len(required)
                remaining_points = [x for x in range(v) if x not in required]
                
                # Try a few random completions
                found = False
                for _ in range(5):
                    if need_more == 0:
                        new_block = sorted(required)
                    else:
                        extra = rng.sample(remaining_points, need_more)
                        new_block = sorted(required | set(extra))
                    
                    # Check if new_block covers all unique_subs
                    # It does by construction (required is union of unique_subs)
                    # But we must ensure new_block is valid (size k, distinct)
                    # And we should check if it's better than just removing? 
                    # No, we are replacing.
                    
                    # Add new_block, remove bi
                    new_tsubs = list(itertools.combinations(new_block, t))
                    
                    # Check if this replacement is actually valid (covers all unique_subs)
                    # Yes, because required points are in new_block.
                    
                    # Update counts
                    for ts in new_tsubs:
                        cover_count[tsub_to_idx[ts]] += 1
                    for ts in block_tsubs[bi]:
                        cover_count[tsub_to_idx[ts]] -= 1
                    
                    removed[bi] = True
                    blocks.append(new_block)
                    block_tsubs.append(new_tsubs)
                    improved = True
                    found = True
                    break
                    # Note: If we add a new block, the loop over 'active' is over the old indices.
                    # We should be careful with indices.
                    # Since we append to blocks, the new block is at the end.
                    # 'active' was generated before, so it doesn't contain the new index.
                    # This is fine for this pass.
    
    result = [blocks[i] for i in range(len(blocks)) if not removed[i]]
    return result


def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin before 120s kill
    rng = random.Random(42)
    
    # Phase 1: Fast greedy baseline
    blocks = _fast_greedy(v, k, t, rng, deadline)
    best = blocks
    best_len = len(blocks)
    
    # Phase 2: Local search
    if time.time() < deadline - 10.0:
        blocks = _local_search(v, k, t, blocks, deadline, rng)
        if len(blocks) < best_len:
            best = blocks
            best_len = len(blocks)
    
    # Phase 3: Multiple greedy restarts with different seeds
    seed = 43
    while time.time() < deadline - 15.0:
        rng2 = random.Random(seed)
        candidate = _fast_greedy(v, k, t, rng2, deadline - 5.0)
        if len(candidate) < best_len:
            best = candidate
            best_len = len(candidate)
            # Try local search on the better one
            if time.time() < deadline - 10.0:
                candidate = _local_search(v, k, t, candidate, deadline, rng2)
                if len(candidate) < best_len:
                    best = candidate
                    best_len = len(candidate)
        seed += 1
    
    return best
# EVOLVE-BLOCK-END
