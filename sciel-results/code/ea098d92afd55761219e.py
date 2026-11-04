"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Strong greedy covering design with bitset-accelerated gain calculation.
    Uses ratio-based selection and multiple-seed strong greedy.
    """
    deadline = time.time() + 145  # 145s deadline to be safe within 150s
    time_budget = 145
    
    # Enumerate all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subs)}
    
    # Precompute for each point x, the list of t-subset indices containing x
    t_subs_containing = [[] for _ in range(v)]
    for i, ts in enumerate(all_t_subs):
        for x in ts:
            t_subs_containing[x].append(i)
    
    # Precompute for each pair (x, y) with x < y, the t-subset indices containing both
    # For t=3, this is just the triplet containing x, y, and z for each z
    # For general t, precompute for each (t-1)-subset the list of t-subsets extending it
    # We'll compute gain on the fly using bitsets
    
    # Bitset of all uncovered triplets (initially all 1)
    # Use Python int as bitset
    full_mask = (1 << n_t) - 1
    uncovered = full_mask
    
    # For each point x, precompute bitmask of t-subsets containing x
    point_mask = [0] * v
    for x in range(v):
        mask = 0
        for idx in t_subs_containing[x]:
            mask |= (1 << idx)
        point_mask[x] = mask
    
    # For each pair (i,j), precompute bitmask of t-subsets containing both
    # For t=3: triplet (i,j,k) for each k not in {i,j}
    # General: for each (t-1)-subset, the t-subsets that extend it
    # We'll compute partial masks on the fly for efficiency
    
    blocks = []
    block_set = set()
    
    def gain_of_point(x, current_block, current_uncovered):
        """Compute how many currently-uncovered t-subsets would be covered by adding x to current_block.
        A t-subset is covered by block B if all its elements are in B.
        Adding x covers t-subsets that contain x and (t-1) elements from current_block."""
        if len(current_block) < t - 1:
            return 0
        # We need t-subsets that contain x and (t-1) points from current_block
        # For t=3: triplets (x, a, b) where a,b in current_block
        # Count those that are in current_uncovered
        count = 0
        if t == 3:
            for i in range(len(current_block)):
                for j in range(i+1, len(current_block)):
                    a = current_block[i]
                    b = current_block[j]
                    # Find the triplet index for (x, a, b)
                    tri = tuple(sorted((x, a, b)))
                    idx = t_sub_to_idx.get(tri, -1)
                    if idx >= 0 and (current_uncovered >> idx) & 1:
                        count += 1
        else:
            # General case: for each (t-1)-subset of current_block, form t-subset with x
            for rest in itertools.combinations(current_block, t-1):
                ts = tuple(sorted(rest + (x,)))
                idx = t_sub_to_idx.get(ts, -1)
                if idx >= 0 and (current_uncovered >> idx) & 1:
                    count += 1
        return count
    
    def build_block_from_seed(seed_tuple, current_uncovered):
        """Greedily build a block starting from seed_tuple, maximizing gain at each step."""
        block = list(seed_tuple)
        block_set_local = set(block)
        
        while len(block) < k:
            best_x = -1
            best_gain = -1
            best_ratio = -1
            
            for x in range(v):
                if x in block_set_local:
                    continue
                gain = gain_of_point(x, block, current_uncovered)
                # Ratio: gain per point (always 1 point added, so ratio = gain)
                # But we can use a slightly different metric: prefer points that cover
                # triplets that are "harder" to cover
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_ratio = gain
            
            if best_x >= 0 and best_gain > 0:
                block.append(best_x)
                block_set_local.add(best_x)
            else:
                # No positive gain, pick random remaining
                remaining = [x for x in range(v) if x not in block_set_local]
                if remaining:
                    block.append(random.choice(remaining))
                    block_set_local.add(block[-1])
                break
        
        if len(block) == k:
            block.sort()
            return block
        return None
    
    def compute_block_mask(block):
        """Compute bitmask of t-subsets covered by a block."""
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_sub_to_idx[ts]
            mask |= (1 << idx)
        return mask
    
    # Strong greedy: try multiple seeds, pick the best block
    while uncovered and time.time() < deadline - 5:
        # Find uncovered triplets
        # For efficiency, sample a few uncovered triplets as seeds
        # Get indices of uncovered triplets
        uncov_indices = []
        # Sample up to 200 uncovered triplets for seed selection
        # First, find some uncovered indices
        # We'll iterate through and collect up to a limit
        count = 0
        sample_limit = 200
        for i in range(n_t):
            if (uncovered >> i) & 1:
                uncov_indices.append(i)
                count += 1
                if count >= sample_limit:
                    break
        
        if not uncov_indices:
            break
        
        # Try multiple seeds
        best_block = None
        best_gain = -1
        best_new_mask = 0
        
        # Shuffle for diversity
        random.shuffle(uncov_indices)
        
        # Try up to 30 seeds
        max_seeds = 30
        for seed_idx in uncov_indices[:max_seeds]:
            seed_tuple = all_t_subs[seed_idx]
            new_block = build_block_from_seed(seed_tuple, uncovered)
            if new_block is None:
                continue
            new_mask = compute_block_mask(new_block)
            new_gain = bin(new_mask & uncovered).count('1')
            if new_gain > best_gain:
                best_gain = new_gain
                best_block = new_block
                best_new_mask = new_mask
                if best_gain >= 30:  # Good enough, stop early
                    break
        
        if best_block is None:
            # Fallback: just pick one uncovered triplet and build
            seed_tuple = all_t_subs[uncov_indices[0]]
            best_block = build_block_from_seed(seed_tuple, uncovered)
            if best_block is None:
                break
            best_new_mask = compute_block_mask(best_block)
        
        if best_block in block_set:
            # Skip duplicate, try another seed
            continue
        
        blocks.append(best_block)
        block_set.add(tuple(best_block))
        uncovered &= ~best_new_mask
    
    # If we still have uncovered triplets, add more blocks
    while uncovered and time.time() < deadline - 3:
        # Find first uncovered
        for i in range(n_t):
            if (uncovered >> i) & 1:
                seed_tuple = all_t_subs[i]
                new_block = build_block_from_seed(seed_tuple, uncovered)
                if new_block is not None and new_block not in block_set:
                    new_mask = compute_block_mask(new_block)
                    blocks.append(new_block)
                    block_set.add(tuple(new_block))
                    uncovered &= ~new_mask
                break
    
    # Post-processing: remove redundant blocks
    if time.time() < deadline - 10:
        changed = True
        while changed and time.time() < deadline - 8:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 8:
                    break
                # Check if block i is redundant
                # Compute mask of all blocks except i
                mask_without_i = 0
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    mask_without_i |= compute_block_mask(blocks[j])
                
                mask_i = compute_block_mask(blocks[i])
                # If mask_i is subset of mask_without_i, block i is redundant
                if mask_i & ~mask_without_i == 0:
                    blocks.pop(i)
                    changed = True
    
    # Verify coverage and add any missing
    if time.time() < deadline - 2:
        covered_mask = 0
        for b in blocks:
            covered_mask |= compute_block_mask(b)
        
        missing = full_mask & ~covered_mask
        if missing:
            # Add blocks for missing triplets
            while missing and time.time() < deadline - 1:
                for i in range(n_t):
                    if (missing >> i) & 1:
                        seed_tuple = all_t_subs[i]
                        new_block = build_block_from_seed(seed_tuple, missing)
                        if new_block is not None:
                            new_mask = compute_block_mask(new_block)
                            if new_block not in block_set:
                                blocks.append(new_block)
                                block_set.add(tuple(new_block))
                            missing &= ~new_mask
                        break
    
    # Remove duplicates
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    
    return unique_blocks

# EVOLVE-BLOCK-END
