import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Precompute mappings
    # Map each t-subset to an index
    subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(subs)}
    n_subs = len(subs)
    
    # Point masks: bit i is 1 if subs[i] contains point x
    point_masks = [0] * v
    # Pair masks: bit i is 1 if subs[i] contains both x and y
    # We only need pair masks for t=3. For general t, we'd need (t-1)-subset masks.
    # Since t=3, we need pair masks.
    pair_masks = {}
    
    for i, sub in enumerate(subs):
        bit = 1 << i
        for x in sub:
            point_masks[x] |= bit
        if t == 3:
            for j in range(3):
                for l in range(j+1, 3):
                    key = (sub[j], sub[l])
                    if key not in pair_masks:
                        pair_masks[key] = 0
                    pair_masks[key] |= bit
                    # Also store the reverse for O(1) lookup if we don't sort keys
                    # But we will sort keys when accessing.
    
    def get_pair_mask(a, b):
        if a > b: a, b = b, a
        return pair_masks.get((a, b), 0)

    # Greedy Construction
    uncovered = (1 << n_subs) - 1  # All bits set
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick the first uncovered triple to start the block
        # Find lowest set bit
        lsb = uncovered & -uncovered
        idx = lsb.bit_length() - 1
        seed_sub = subs[idx]
        
        block = list(seed_sub)
        # We need to fill the block up to size k
        # Current block size is t. We need k-t more points.
        
        # To optimize, we can pre-calculate the "current pair union"
        # But it's easier to just compute gain for candidates.
        
        while len(block) < k:
            if time.time() > deadline:
                break
                
            # Calculate pair union for current block
            # This is the union of all pair masks for pairs in the current block
            pair_union = 0
            for i in range(len(block)):
                for j in range(i+1, len(block)):
                    pair_union |= get_pair_mask(block[i], block[j])
            
            best_gain = -1
            best_x = -1
            # Candidates are points not in block
            for x in range(v):
                if x in block:
                    continue
                
                # Gain is number of uncovered triples containing x and at least t-1 points from block
                # For t=3, this is triples containing x and a pair from block.
                # Mask of triples containing x: point_masks[x]
                # Mask of triples containing a pair from block: pair_union
                # Intersection: triples containing x AND a pair from block
                # We only care about uncovered ones.
                
                # Note: If t > 3, this logic changes. But task is t=3.
                # General logic for t:
                # We need (t-1)-subset masks.
                # For t=3, (t-1)=2, so pairs.
                
                candidate_mask = point_masks[x] & pair_union
                gain = (uncovered & candidate_mask).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1:
                # Should not happen if uncovered is not empty
                break
            block.append(best_x)
        
        if len(block) < k:
             # If we couldn't fill the block, add random points to make it valid size?
             # Or just break.
             pass
             
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        block_mask = 0
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            block_mask |= (1 << idx)
        uncovered &= ~block_mask

    # If we broke out of the loop due to time, we might have uncovered triples.
    # We must return a valid covering.
    # If uncovered is not 0, we need to cover the rest.
    # This is a fallback. Ideally, the greedy should finish.
    # If it didn't, we add blocks for remaining uncovered triples greedily (slow but necessary for validity).
    if uncovered:
        while uncovered:
            if time.time() > deadline:
                break
            lsb = uncovered & -uncovered
            idx = lsb.bit_length() - 1
            seed_sub = subs[idx]
            block = list(seed_sub)
            while len(block) < k:
                # Simple random or greedy add
                # To save time, just pick random points not in block
                candidates = [x for x in range(v) if x not in block]
                if not candidates: break
                x = random.choice(candidates)
                block.append(x)
            block.sort()
            blocks.append(block)
            block_mask = 0
            for sub in itertools.combinations(block, t):
                idx2 = sub_to_idx[sub]
                block_mask |= (1 << idx2)
            uncovered &= ~block_mask
    
    # Local Search: Remove redundant blocks
    # Calculate coverage count for each triple
    # count[i] = number of blocks containing subs[i]
    counts = [0] * n_subs
    block_indices = []
    for b in blocks:
        for sub in itertools.combinations(b, t):
            idx = sub_to_idx[sub]
            counts[idx] += 1
            block_indices.append(idx)
            
    # Try to remove blocks
    # Iterate multiple times as removing one block might make another redundant
    # But usually one pass or a few passes is enough.
    # We check if all triples in a block have count > 1. If so, the block is redundant.
    # Wait, if count > 1, it means it's covered by at least one OTHER block.
    # So if for all sub in block, count[sub] > 1, then block is redundant.
    
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for b in blocks:
            redundant = True
            for sub in itertools.combinations(b, t):
                idx = sub_to_idx[sub]
                if counts[idx] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove block b
                for sub in itertools.combinations(b, t):
                    idx = sub_to_idx[sub]
                    counts[idx] -= 1
                changed = True
            else:
                new_blocks.append(b)
        blocks = new_blocks
        
        if time.time() > deadline:
            break

    return blocks
