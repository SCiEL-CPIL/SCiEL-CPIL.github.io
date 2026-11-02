import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 135.0
    rng = random.Random(42)
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute coverage bitmasks for all possible k-subsets? No, too many (C(24,7) ~ 346k).
    # Instead, compute on the fly but efficiently.
    
    def get_block_mask(block):
        """Compute a bitmask of t-subsets covered by a given block."""
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        return mask

    def greedy_cover(seed):
        """Fast greedy covering using bitmasks."""
        rng_local = random.Random(seed)
        uncovered_mask = (1 << n_t) - 1
        blocks = []
        
        while uncovered_mask:
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random uncovered t-subset to start the block
            # We need to pick an index where the bit is set
            # Efficient way: sample random indices until we find one? Or iterate?
            # n_t is 2024 for v=24, t=3. Iterating is fine.
            start_idx = -1
            while start_idx == -1:
                start_idx = rng_local.randint(0, n_t - 1)
                if (uncovered_mask >> start_idx) & 1:
                    break
            
            start_ts = all_t_subsets[start_idx]
            block = list(start_ts)
            
            # Greedily add points to maximize coverage of remaining uncovered
            while len(block) < k:
                best_x = None
                best_gain = -1
                
                for x in range(v):
                    if x in block:
                        continue
                    # Calculate gain: how many new t-subsets does adding x cover?
                    # A t-subset is formed by x and t-1 elements from the current block.
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = t_subset_to_idx[ts]
                        if (uncovered_mask >> idx) & 1:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                            
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered mask
            block_mask = get_block_mask(block)
            uncovered_mask &= ~block_mask
            
        return blocks

    def local_search_optimize(blocks, seed):
        """
        Local search: try to remove blocks. If removal breaks coverage, 
        try to repair by adding one block that covers the most exposed t-subsets.
        """
        rng_local = random.Random(seed)
        
        # Initial coverage
        current_blocks = list(blocks)
        # Compute coverage counts for each t-subset
        # coverage_count[i] = number of blocks covering t-subset i
        coverage_count = np.zeros(n_t, dtype=np.int32)
        
        for b in current_blocks:
            mask = get_block_mask(b)
            # Iterate set bits
            m = mask
            while m:
                # Get index of lowest set bit
                idx = (m & -m).bit_length() - 1
                coverage_count[idx] += 1
                m &= (m - 1)
                
        # Helper to check if a removal is safe
        def is_removal_safe(block_idx, block):
            # Check if any t-subset in this block has coverage_count == 1
            mask = get_block_mask(block)
            m = mask
            while m:
                idx = (m & -m).bit_length() - 1
                if coverage_count[idx] == 1:
                    return False
                m &= (m - 1)
            return True

        def repair(exposed_mask, current_blocks):
            """
            Find a block that covers the most t-subsets in exposed_mask.
            """
            best_block = None
            best_gain = 0
            
            # Sample a few random blocks from the universe? Or generate candidates?
            # Generating candidates: pick a random exposed t-subset, build a block around it greedily?
            # Or just try random k-subsets?
            
            # Let's try a few random k-subsets
            # And also try blocks based on exposed t-subsets
            
            candidates = set()
            
            # Add some random blocks
            for _ in range(10):
                cand = tuple(sorted(rng_local.sample(range(v), k)))
                candidates.add(cand)
                
            # Add blocks built from exposed t-subsets
            # Pick a few exposed t-subsets
            exposed_indices = []
            m = exposed_mask
            while m and len(exposed_indices) < 5:
                idx = (m & -m).bit_length() - 1
                exposed_indices.append(idx)
                m &= (m - 1)
                
            for idx in exposed_indices:
                start_ts = all_t_subsets[idx]
                block = list(start_ts)
                while len(block) < k:
                    # Add random point
                    cand_pts = [x for x in range(v) if x not in block]
                    if not cand_pts: break
                    x = rng_local.choice(cand_pts)
                    block.append(x)
                block.sort()
                candidates.add(tuple(block))
                
            # Evaluate candidates
            for cand in candidates:
                cand_mask = get_block_mask(cand)
                # Count how many exposed t-subsets this covers
                gain = bin(cand_mask & exposed_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block = list(cand)
                    
            return best_block

        # Main local search loop
        steps = 0
        while time.time() < deadline - 10.0:
            steps += 1
            # Try to remove a random block
            if not current_blocks:
                break
                
            idx = rng_local.randint(0, len(current_blocks) - 1)
            block_to_remove = current_blocks[idx]
            
            # Check if safe to remove
            if is_removal_safe(idx, block_to_remove):
                # Remove it
                mask = get_block_mask(block_to_remove)
                m = mask
                while m:
                    i = (m & -m).bit_length() - 1
                    coverage_count[i] -= 1
                    m &= (m - 1)
                current_blocks.pop(idx)
                # Accept immediately? Or keep track for backtracking?
                # For simplicity, accept if it works. This is a hill-climbing move.
            else:
                # Try to repair
                # Find exposed t-subsets (those with count 1 that are in this block)
                exposed_mask = 0
                mask = get_block_mask(block_to_remove)
                m = mask
                while m:
                    i = (m & -m).bit_length() - 1
                    if coverage_count[i] == 1:
                        exposed_mask |= (1 << i)
                    m &= (m - 1)
                    
                if exposed_mask == 0:
                    # Shouldn't happen if is_removal_safe was false
                    continue
                    
                new_block = repair(exposed_mask, current_blocks)
                if new_block is None:
                    continue
                    
                # Check if new_block actually covers all exposed
                new_mask = get_block_mask(new_block)
                if (new_mask & exposed_mask) != exposed_mask:
                    continue
                    
                # Also check if new_block is distinct
                if new_block in current_blocks:
                    continue
                    
                # Apply: Remove old, Add new
                # Update coverage counts
                old_mask = get_block_mask(block_to_remove)
                m = old_mask
                while m:
                    i = (m & -m).bit_length() - 1
                    coverage_count[i] -= 1
                    m &= (m - 1)
                    
                m = new_mask
                while m:
                    i = (m & -m).bit_length() - 1
                    coverage_count[i] += 1
                    m &= (m - 1)
                    
                current_blocks[idx] = new_block
                current_blocks[idx].sort()

        return current_blocks

    # 1. Generate initial coverings
    best_blocks = None
    best_count = float('inf')
    
    # Try a few greedy seeds
    for seed in range(5):
        if time.time() > deadline - 20.0:
            break
        blocks = greedy_cover(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        return []
        
    # 2. Optimize using local search
    # We have time left, use it for local search on the best found so far
    # And maybe restarts?
    
    # Let's do a few local search passes on different seeds
    # But we must ensure we return a valid covering.
    
    # Strategy: 
    # - Keep the best greedy result as a fallback.
    # - Run local search on the best greedy result.
    # - If local search improves it, use that.
    # - If time permits, run local search on other greedy results.
    
    # To be safe and fast, let's just run local search on the single best greedy result.
    # Or maybe a few?
    
    # Let's try to optimize the best one found so far.
    # We need to be careful with the time.
    
    # Let's run local search on the best_blocks
    # But local search might be slow if not careful.
    
    # Let's try one local search pass with a tight inner deadline.
    # Actually, the local_search_optimize function has its own deadline check.
    
    # Let's run it.
    # Note: local_search_optimize might return a covering that is NOT valid if we bug out.
    # But we verify at the end.
    
    # Let's run local search on the best greedy result.
    optimized_blocks = local_search_optimize(best_blocks, 0)
    
    if len(optimized_blocks) < best_count:
        best_blocks = optimized_blocks
        best_count = len(best_blocks)
        
    # If we have time, try another seed for greedy + local search?
    if time.time() < deadline - 30.0:
        for seed in range(5, 10):
            if time.time() > deadline - 15.0:
                break
            blocks = greedy_cover(seed)
            if len(blocks) < best_count:
                # Try to optimize this new best
                opt_blocks = local_search_optimize(blocks, seed)
                if len(opt_blocks) < best_count:
                    best_blocks = opt_blocks
                    best_count = len(best_blocks)
                else:
                    best_blocks = blocks
                    best_count = len(blocks)
            # Even if not better, maybe optimizing it would help?
            # But that's expensive. Let's stick to optimizing only if it's already competitive.
            # Actually, let's just optimize the current best_blocks again if we have time?
            # No, local search is stochastic.
            
    # Final Verification
    # Ensure best_blocks is valid
    covered_mask = 0
    for b in best_blocks:
        covered_mask |= get_block_mask(b)
        
    if covered_mask != (1 << n_t) - 1:
        # Invalid, fall back to a guaranteed valid greedy
        # The greedy_cover function is guaranteed to produce a valid covering if it runs to completion.
        # But it might have stopped early due to time.
        # Let's run a simple, safe greedy that doesn't have a time limit inside the loop, 
        # but we have to be careful.
        # Actually, the first greedy_cover(0) should have been valid.
        # Let's just return the first valid one we found?
        # We didn't store them separately.
        
        # Let's run a fresh, fast greedy with no time limit inside (but we are near deadline)
        # This is risky.
        
        # Better: Re-run greedy_cover(0) and assume it's valid.
        # Or just return best_blocks if we trust the logic.
        # The logic in greedy_cover ensures every step covers at least one new t-subset 
        # (by picking an uncovered one to start). So it should be valid.
        # Why would it be invalid? 
        # Ah, if `time.time() > deadline - 5.0` triggers, it breaks.
        # So it might be incomplete.
        
        # Let's run a "complete" greedy that ignores time until the very end?
        # No, that might timeout.
        
        # Let's just return the best_blocks we have. If it's invalid, we score 0.
        # But we want to avoid that.
        
        # Let's do a quick fix: if invalid, run a simple greedy that is very fast 
        # and likely to finish.
        
        # Simple greedy:
        uncovered_mask = (1 << n_t) - 1
        blocks = []
        while uncovered_mask:
            # Pick first uncovered
            start_idx = 0
            while not ((uncovered_mask >> start_idx) & 1):
                start_idx += 1
            start_ts = all_t_subsets[start_idx]
            block = list(start_ts)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block: continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = t_subset_to_idx[ts]
                        if (uncovered_mask >> idx) & 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None: break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            block_mask = get_block_mask(block)
            uncovered_mask &= ~block_mask
            if time.time() > deadline - 1.0:
                break
                
        if len(blocks) < len(best_blocks):
             best_blocks = blocks
             
    return best_blocks
