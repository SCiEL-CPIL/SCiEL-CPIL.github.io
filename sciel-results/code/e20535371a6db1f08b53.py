import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin before 120s kill
    
    # Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # For each block (k-subset), precompute which t-subsets it covers
    # But there are C(24,7) = 346104 possible blocks, too many to precompute all
    # Instead, compute on the fly but efficiently
    
    # Precompute for each point, the list of t-1 subsets that can be paired with it
    # Actually, let's precompute the mask of t-subsets covered by each possible block
    # That's too many. Instead, let's be smarter.
    
    # Key insight: for a block B of size k, the t-subsets it covers are all C(k,t) t-subsets of B
    # We can compute this quickly if we have the subset_to_idx mapping
    
    # Precompute: for each t-subset, its index (already done)
    
    # Strategy: Greedy with lookahead + local search
    # 1. Greedy: repeatedly pick the block that covers the most uncovered t-subsets
    # 2. Local search: try to replace blocks with better ones, remove redundant blocks
    
    def block_mask(block):
        """Compute bitmask of t-subsets covered by this block"""
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
        return mask
    
    def count_bits(mask):
        """Count set bits in a Python int bitmask"""
        return bin(mask).count('1')
    
    def greedy_construct(seed_rng, use_lookahead=False):
        blocks = []
        local_uncovered = (1 << n_subsets) - 1
        
        while local_uncovered != 0:
            if time.time() > deadline:
                return None
            
            # Find the block that covers the most uncovered t-subsets
            best_block = None
            best_gain = -1
            best_mask = 0
            
            # For efficiency, we can't check all C(24,7) blocks
            # Instead, use a heuristic: start from an uncovered t-subset and expand
            # Pick a random uncovered t-subset as seed
            seed_idx = -1
            while seed_idx == -1:
                candidate = seed_rng.randint(0, n_subsets - 1)
                if (local_uncovered >> candidate) & 1:
                    seed_idx = candidate
            
            seed_sub = all_subsets[seed_idx]
            
            # Build a candidate block around the seed
            # Start with the seed elements, then greedily add points
            block = list(seed_sub)
            block_set = set(block)
            
            # Greedily add points to maximize new coverage
            while len(block) < k:
                block_set = set(block)
                best_point = None
                best_gain = -1
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Gain = number of uncovered t-subsets that contain x and (t-1) elements from block
                    gain = 0
                    for sub in itertools.combinations(block, t - 1):
                        tri = tuple(sorted(sub + (x,)))
                        idx = subset_to_idx[tri]
                        if (local_uncovered >> idx) & 1:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        
                if best_point is None or best_gain <= 0:
                    available = [x for x in range(v) if x not in block_set]
                    if available:
                        best_point = seed_rng.choice(available)
                    else:
                        break
                block.append(best_point)
                
            if len(block) < k:
                block_set = set(block)
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block_set:
                        block.append(x)
                        
            block.sort()
            
            # Compute the actual gain of this block
            bmask = block_mask(block)
            gain = count_bits(bmask & local_uncovered)
            
            # Try a few alternative blocks to see if we can do better
            # Try expanding from different seeds
            for _ in range(3):
                alt_seed_idx = -1
                while alt_seed_idx == -1:
                    candidate = seed_rng.randint(0, n_subsets - 1)
                    if (local_uncovered >> candidate) & 1:
                        alt_seed_idx = candidate
                
                alt_seed_sub = all_subsets[alt_seed_idx]
                alt_block = list(alt_seed_sub)
                alt_block_set = set(alt_block)
                
                while len(alt_block) < k:
                    alt_block_set = set(alt_block)
                    best_point = None
                    best_gain = -1
                    
                    for x in range(v):
                        if x in alt_block_set:
                            continue
                        
                        gain = 0
                        for sub in itertools.combinations(alt_block, t - 1):
                            tri = tuple(sorted(sub + (x,)))
                            idx = subset_to_idx[tri]
                            if (local_uncovered >> idx) & 1:
                                gain += 1
                                
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                            
                    if best_point is None or best_gain <= 0:
                        available = [x for x in range(v) if x not in alt_block_set]
                        if available:
                            best_point = seed_rng.choice(available)
                        else:
                            break
                    alt_block.append(best_point)
                    
                if len(alt_block) < k:
                    alt_block_set = set(alt_block)
                    for x in range(v):
                        if len(alt_block) >= k:
                            break
                        if x not in alt_block_set:
                            alt_block.append(x)
                            
                alt_block.sort()
                alt_bmask = block_mask(alt_block)
                alt_gain = count_bits(alt_bmask & local_uncovered)
                
                if alt_gain > gain:
                    block = alt_block
                    bmask = alt_bmask
                    gain = alt_gain
            
            blocks.append(block)
            local_uncovered &= ~bmask
            
        return blocks
    
    def remove_redundant(blocks):
        """Remove blocks that are redundant (all their t-subsets covered by other blocks)"""
        if not blocks:
            return blocks
        
        n = len(blocks)
        # For each block, compute which t-subsets it covers
        block_triples = []
        for b in blocks:
            triples = list(itertools.combinations(b, t))
            block_triples.append(triples)
        
        # For each t-subset, find which blocks cover it
        subset_coverers = {}
        for i, subs in enumerate(block_triples):
            for sub in subs:
                if sub not in subset_coverers:
                    subset_coverers[sub] = set()
                subset_coverers[sub].add(i)
        
        # A block is redundant if for every t-subset it covers, there's another block covering it
        keep = [True] * n
        for i in range(n):
            redundant = True
            for sub in block_triples[i]:
                coverers = subset_coverers.get(sub, set())
                if len(coverers) == 1 and i in coverers:
                    redundant = False
                    break
            if redundant:
                keep[i] = False
        
        return [blocks[i] for i in range(n) if keep[i]]
    
    def local_search(blocks, seed_rng, max_iters=100):
        """Try to improve by removing redundant blocks or finding better alternatives"""
        blocks = list(blocks)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try to remove redundant blocks
            new_blocks = remove_redundant(blocks)
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                continue
            
            # Try to replace a block with a better one
            # Pick a random block, remove it, see what's uncovered, try to cover with one new block
            if len(blocks) > 1:
                idx = seed_rng.randint(0, len(blocks) - 1)
                old_block = blocks[idx]
                remaining = blocks[:idx] + blocks[idx+1:]
                
                # Compute what's uncovered by remaining
                remaining_mask = 0
                for b in remaining:
                    remaining_mask |= block_mask(b)
                
                # What does old_block cover that remaining doesn't?
                old_mask = block_mask(old_block)
                missing = old_mask & ~remaining_mask
                
                if missing == 0:
                    # old_block is redundant
                    blocks = remaining
                    continue
                
                # Try to find a new block that covers all of missing
                # This is hard, so just try to find a block that covers more of missing
                best_new_block = None
                best_new_gain = -1
                
                # Try expanding from a random missing t-subset
                missing_idx = -1
                while missing_idx == -1:
                    candidate = seed_rng.randint(0, n_subsets - 1)
                    if (missing >> candidate) & 1:
                        missing_idx = candidate
                
                seed_sub = all_subsets[missing_idx]
                new_block = list(seed_sub)
                new_block_set = set(new_block)
                
                while len(new_block) < k:
                    new_block_set = set(new_block)
                    best_point = None
                    best_gain = -1
                    
                    for x in range(v):
                        if x in new_block_set:
                            continue
                        
                        gain = 0
                        for sub in itertools.combinations(new_block, t - 1):
                            tri = tuple(sorted(sub + (x,)))
                            idx = subset_to_idx[tri]
                            if (missing >> idx) & 1:
                                gain += 1
                                
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                            
                    if best_point is None or best_gain <= 0:
                        available = [x for x in range(v) if x not in new_block_set]
                        if available:
                            best_point = seed_rng.choice(available)
                        else:
                            break
                    new_block.append(best_point)
                    
                if len(new_block) < k:
                    new_block_set = set(new_block)
                    for x in range(v):
                        if len(new_block) >= k:
                            break
                        if x not in new_block_set:
                            new_block.append(x)
                            
                new_block.sort()
                new_mask = block_mask(new_block)
                new_gain = count_bits(new_mask & missing)
                
                # Check if new_block covers all of missing
                if (new_mask & missing) == missing:
                    # Replace old_block with new_block
                    blocks[idx] = new_block
                    continue
                
                # Even if it doesn't cover all, if it covers more than old_block did of missing
                # and the remaining is covered by other blocks, we might be able to add it
                # This is complex, so skip for now
            
            break  # No more easy improvements
        
        return blocks
    
    # Run multiple greedy restarts with different seeds
    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 0
    while time.time() < deadline - 30:
        seed_rng = random.Random(num_restarts)
        blocks = greedy_construct(seed_rng)
        if blocks is None:
            continue
        
        # Post-processing
        pruned_blocks = remove_redundant(blocks)
        pruned_blocks = local_search(pruned_blocks, seed_rng, max_iters=50)
        
        if len(pruned_blocks) < best_count:
            best_count = len(pruned_blocks)
            best_blocks = pruned_blocks
        
        num_restarts += 1
    
    if best_blocks is None:
        if time.time() < deadline - 5:
            seed_rng = random.Random(999)
            blocks = greedy_construct(seed_rng)
            if blocks:
                return blocks
        return []
        
    return best_blocks
