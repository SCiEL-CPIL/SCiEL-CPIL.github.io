"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute all k-blocks and their coverage masks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, compute bitmask of t-subsets it covers
    # Using Python integers as bitmasks (n_t ~ 11628 bits, fits in Python int)
    block_masks = [0] * n_blocks
    for bi, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            mask |= (1 << idx)
        block_masks[bi] = mask
    
    # Full mask of all t-subsets uncovered
    full_mask = (1 << n_t) - 1
    
    def greedy_cover(noise_seed=None, alpha=0.0):
        """
        Greedy set cover with optional random noise.
        alpha: noise magnitude (0 = pure greedy, >0 = stochastic)
        """
        if noise_seed is not None:
            rng = random.Random(noise_seed)
        else:
            rng = random.Random()
            
        uncovered = full_mask
        blocks = []
        
        while uncovered:
            best_block = None
            best_gain = -1
            
            # Find block with maximum coverage of uncovered t-subsets
            # Optimization: skip blocks with 0 gain
            for bi in range(n_blocks):
                gain = (uncovered & block_masks[bi]).bit_count()
                if gain > 0:
                    # Add noise for diversification
                    if alpha > 0:
                        gain += rng.random() * alpha
                    if gain > best_gain:
                        best_gain = gain
                        best_block = bi
                        
            if best_block is None:
                break
                
            blocks.append(best_block)
            uncovered &= ~block_masks[best_block]
            
            if time.time() > deadline:
                return None  # Timeout
                
        return blocks
    
    def prune_blocks(blocks):
        """
        Remove redundant blocks iteratively.
        A block is redundant if all t-subsets it covers are covered by other blocks.
        """
        if not blocks:
            return blocks
        
        # Work with indices for efficiency
        current = list(blocks)
        
        # Iterate until no more changes
        changed = True
        while changed and time.time() < deadline:
            changed = False
            n = len(current)
            
            if n == 0:
                break
            
            # Count how many blocks cover each t-subset
            count_cover = [0] * n_t
            
            for bi in current:
                mask = block_masks[bi]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    count_cover[idx] += 1
                    m ^= lsb
            
            # Identify essential blocks (those with at least one uniquely covered t-subset)
            essential = [False] * n
            for i, bi in enumerate(current):
                mask = block_masks[bi]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if count_cover[idx] == 1:
                        essential[i] = True
                        break
                    m ^= lsb
            
            # Remove non-essential blocks
            new_current = [current[i] for i in range(n) if essential[i]]
            
            if len(new_current) < len(current):
                changed = True
                current = new_current
        
        return current
    
    # Run multiple greedy covers with different noise levels and seeds
    best_blocks = None
    best_count = float('inf')
    
    # First, run pure greedy
    pure_result = greedy_cover(noise_seed=0, alpha=0.0)
    if pure_result is not None:
        pruned = prune_blocks(pure_result)
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_blocks = pruned
    
    # Then, run stochastic greedy with various alpha values and seeds
    alphas = [0.1, 0.5, 1.0, 2.0]
    seeds = range(1, 20)
    
    for alpha in alphas:
        for seed in seeds:
            if time.time() > deadline:
                break
            
            result = greedy_cover(noise_seed=seed, alpha=alpha)
            if result is not None:
                pruned = prune_blocks(result)
                if len(pruned) < best_count:
                    best_count = len(pruned)
                    best_blocks = pruned
    
    # If we still have time, try more aggressive pruning with local search
    if best_blocks is not None and time.time() < deadline - 5.0:
        # Try to improve by swapping blocks
        # For each block, try replacing it with a different block that maintains coverage
        # This is expensive, so only do a few iterations
        current = list(best_blocks)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline - 2.0:
                    break
                # Compute coverage of current block i
                mask_i = block_masks[current[i]]
                
                # Check if any other block can replace it while maintaining coverage
                # i.e., all t-subsets in mask_i are covered by other blocks
                # This is equivalent to: for each t-subset in mask_i, count_cover >= 2
                # But we need to check after removing block i
                
                # Simpler: temporarily remove block i and check if all its t-subsets are still covered
                # This is O(n_t) per check, which is ~11628 ops. With ~300 blocks, that's ~3.5M ops.
                # Might be too slow if done naively.
                
                # Alternative: use the same essential check
                # Count coverage without block i
                count_without_i = [0] * n_t
                for j, bi in enumerate(current):
                    if j == i:
                        continue
                    mask = block_masks[bi]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        count_without_i[idx] += 1
                        m ^= lsb
                
                # Check if all t-subsets in mask_i are covered by other blocks
                m = mask_i
                can_remove = True
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if count_without_i[idx] == 0:
                        can_remove = False
                        break
                    m ^= lsb
                
                if can_remove:
                    current.pop(i)
                    improved = True
                    break  # Restart the loop since indices changed
        
        if len(current) < best_count:
            best_count = len(current)
            best_blocks = current
    
    # Return the best blocks found
    if best_blocks is None:
        # Fallback: return pure greedy result
        return pure_result if pure_result is not None else []
    
    # Convert block indices back to actual blocks
    return [list(all_blocks[bi]) for bi in best_blocks]
# EVOLVE-BLOCK-END
