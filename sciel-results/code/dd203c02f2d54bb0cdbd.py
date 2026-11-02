import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    t0 = time.time()
    # Safety margin: 110s out of 120s limit
    DEADLINE = t0 + 110.0 

    # Helper for combinations count
    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # Fallback for non-standard parameters (though task specifies 19,8,5)
    if v != 19 or k != 8 or t != 5:
        # Simple greedy fallback
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered and time.time() < DEADLINE:
            target = min(uncovered)
            remaining = [x for x in range(v) if x not in target]
            need = k - t
            best_block = None
            best_gain = -1
            # For small v, k, t this is fine. For 19,8,5 we use the optimized path below.
            for comb in itertools.combinations(remaining, need):
                block = tuple(sorted(target + comb))
                gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            if best_block is None:
                break
            blocks.append(list(best_block))
            uncovered.difference_update(itertools.combinations(best_block, t))
        return blocks

    # --- Optimized Path for v=19, k=8, t=5 ---
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute coverage structures
    # block_to_ts[j]: list of t-subset indices covered by block j
    block_to_ts = []
    # ts_to_blocks[i]: list of block indices covering t-subset i
    ts_to_blocks = [[] for _ in range(n_t)]
    
    # C(k, t) = C(8, 5) = 56
    ts_per_block = C(k, t)
    
    for j, block in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(covered_indices)
        
    # Convert to numpy arrays for faster access where possible
    # block_ts_arrays is a list of arrays, but we mostly need the list of indices for bincount
    # ts_blocks_arrays is a list of lists of block indices
    
    # Precompute frequency of each t-subset and static rarity score for blocks
    freq = np.zeros(n_t, dtype=np.int32)
    for ts_idx in range(n_t):
        freq[ts_idx] = len(ts_to_blocks[ts_idx])
        
    inv_freq = np.zeros(n_t, dtype=np.float32)
    mask = freq > 0
    inv_freq[mask] = 1.0 / freq[mask]
    
    static_rarity = np.zeros(n_blocks, dtype=np.float32)
    for j in range(n_blocks):
        static_rarity[j] = np.sum(inv_freq[block_to_ts[j]])

    def run_greedy(seed):
        random.seed(seed)
        # gain[j] = number of currently uncovered t-subsets covered by block j
        # Initialize with max possible gain (C(k,t)) for all blocks
        gain = np.full(n_blocks, ts_per_block, dtype=np.int32)
        is_uncovered = np.full(n_t, True, dtype=bool)
        is_selected = np.full(n_blocks, False, dtype=bool)
        sol = []
        
        # We need to track which t-subsets are covered to update gains correctly
        # Actually, we only need to know which t-subsets were *newly* covered to decrement gains of other blocks.
        
        while True:
            if time.time() > DEADLINE - 5.0:
                return None
            
            # Find max gain among unselected blocks
            # Mask out selected blocks
            masked_gain = gain.copy()
            masked_gain[is_selected] = -1
            max_gain = np.max(masked_gain)
            
            if max_gain <= 0:
                # If max gain is 0, we might be stuck or finished.
                # If we have uncovered subsets but no block covers them, it's invalid.
                # But since we pick from all blocks, if uncovered exists, some block must cover it.
                # Unless we selected all blocks? Unlikely.
                if not is_uncovered.all(): # If some are uncovered
                     # This shouldn't happen if we pick correctly, but break to avoid infinite loop
                     break
                else:
                    break
                
            # Find candidates with max gain
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            # Pick the candidate with highest static_rarity to prioritize covering rare subsets
            # This helps break ties in a way that might lead to better global coverage
            rarities = static_rarity[candidates]
            max_rarity = np.max(rarities)
            # Find all candidates with max rarity to allow some randomness if there are ties
            best_candidates = candidates[rarities == max_rarity]
            b_idx = best_candidates[random.randint(0, len(best_candidates) - 1)]
            
            sol.append(list(all_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1 # Mark as selected (prevent re-selection)
            
            # Identify newly covered t-subsets
            newly_covered_ts = []
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    newly_covered_ts.append(ts_idx)
            
            if not newly_covered_ts:
                continue
                
            # Vectorized gain update
            # We need to decrement gain for all blocks that contain any of the newly_covered_ts
            # and are not selected.
            
            # Collect all block indices that contain any of these t-subsets
            # This list will have duplicates if a block covers multiple newly covered t-subsets.
            # We want to decrement gain by the COUNT of newly covered t-subsets shared with this block.
            
            blocks_to_decrement = []
            for ts_idx in newly_covered_ts:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                
            if blocks_to_decrement:
                # bincount gives the number of times each block index appears in the list
                # This is exactly the number of newly covered t-subsets this block shares
                counts = np.bincount(blocks_to_decrement, minlength=n_blocks)
                gain -= counts
                # Ensure selected blocks stay low so they aren't picked again
                # (Though is_selected mask handles it, keeping gain low is good practice)
                gain[is_selected] = -1000

        # Check if valid
        if is_uncovered.any():
            return None
        return sol

    def optimize_redundancy(blocks):
        """Remove redundant blocks."""
        if not blocks:
            return blocks
            
        # Map blocks to their t-subset indices
        # We need to handle blocks that might not be in the standard set? 
        # No, we only pick from all_blocks, so they are in block_to_idx.
        
        # cover_count[i] = number of blocks covering t-subset i
        cover_count = np.zeros(n_t, dtype=np.int32)
        block_ts_indices = []
        
        for b in blocks:
            idx = block_to_idx.get(tuple(b), -1)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                # Should not happen if we only pick from all_blocks
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(t_subset_to_idx[ts])
                block_ts_indices.append(ts_list)
                
        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > DEADLINE - 2.0:
                    return blocks
                    
                # Check if block i is redundant: does it cover any t-subset uniquely?
                redundant = True
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    blocks.pop(i)
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    block_ts_indices.pop(i)
                    changed = True
                    break
                    
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple randomized greedy restarts
    # The vectorized greedy is fast. Let's try to run as many as possible.
    # Each restart takes ~O(n_blocks * avg_coverage) time for updates.
    # With 75k blocks and ~56 tsubs per block, and ~300 blocks in solution, 
    # updates are heavy but numpy bincount is fast.
    
    num_restarts = 0
    while time.time() < DEADLINE - 15.0:
        num_restarts += 1
        sol = run_greedy(num_restarts)
        if sol is None:
            continue
        
        # Apply redundancy removal
        sol = optimize_redundancy(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_blocks = sol
            # If we get a very good solution, we might want to stop or keep going
            # Best known is 335. If we get close, keep trying.
            
    if best_blocks is None:
        # Fallback: return a valid (though potentially large) cover
        # Just pick random blocks until covered? Or return empty if timeout?
        # Must return a valid cover.
        # Let's do a simple greedy without vectorization as a last resort if we failed above
        # But usually run_greedy succeeds.
        # If it fails, it's likely due to timeout.
        # Let's try one more time with a fixed seed and less strict deadline?
        # No, just return what we have or a simple valid set.
        # A simple valid set: all blocks containing a specific point? No, that's not a cover.
        # Actually, if run_greedy returns None, it means it didn't finish or got stuck.
        # Given the density, it should finish.
        
        # Emergency fallback: return a large valid cover
        # Pick blocks randomly until covered
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        random.seed(42)
        while uncovered and time.time() < DEADLINE - 1.0:
            b = random.sample(all_blocks, 1)[0]
            if b in blocks: continue
            blocks.append(list(b))
            for ts in itertools.combinations(b, t):
                if ts in uncovered:
                    uncovered.remove(ts)
        return blocks

    return best_blocks
