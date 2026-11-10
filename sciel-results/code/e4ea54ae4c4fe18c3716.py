import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # Safety margin

    # Precompute t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}
    
    # Precompute t-subset indices for each point combination
    # For a block (k-subset), its t-subsets are all C(k,t) combinations
    # We'll cache these
    
    block_tsub_cache = {}
    
    def get_block_tsub_indices(block):
        """Get the indices of t-subsets covered by a block."""
        key = tuple(sorted(block))
        if key in block_tsub_cache:
            return block_tsub_cache[key]
        
        idxs = np.array([tsub_index[combo] for combo in itertools.combinations(key, t)], dtype=np.int32)
        block_tsub_cache[key] = idxs
        return idxs

    def get_block_mask(block):
        """Get the coverage bitset for a block."""
        key = tuple(sorted(block))
        if key in block_tsub_cache:
            idxs = block_tsub_cache[key]
        else:
            idxs = np.array([tsub_index[combo] for combo in itertools.combinations(key, t)], dtype=np.int32)
            block_tsub_cache[key] = idxs
        
        # Convert to bitset
        mask = 0
        for idx in idxs:
            mask |= (1 << idx)
        return mask

    best_blocks = []
    best_count = float('inf')
    
    # Multiple restarts
    max_restarts = 150
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        rng = random.Random(restart * 7919 + 13)
        
        # Uncovered t-subsets as a bitset
        uncovered = (1 << n_tsubs) - 1  # All bits set initially
        
        blocks = []
        block_masks = []
        
        # Greedy Phase with bitset operations
        while True:
            if time.time() > deadline - 5.0:
                break
            if uncovered == 0:
                break
            
            # Pick a random uncovered t-subset
            # For efficiency, use a simple approach: find a set bit
            # Use bit manipulation to find a random set bit
            # Convert to numpy for random selection (slower but more diverse)
            
            # Quick approach: pick from a small sample of uncovered indices
            # Find some uncovered indices by checking bits
            # For v=24, n_tsubs=2024, this is manageable
            
            # Get all uncovered indices (could be slow if many uncovered)
            # Alternative: use a probabilistic approach
            # Let's just pick the lowest set bit for speed, with some randomness
            
            # To get diversity, we can use the restart index to bias selection
            # Simple approach: try a few random positions and pick the first uncovered
            crit_idx = None
            for _ in range(20):  # Try up to 20 random positions
                candidate = rng.randint(0, n_tsubs - 1)
                if (uncovered >> candidate) & 1:
                    crit_idx = candidate
                    break
            
            if crit_idx is None:
                # Fallback: find lowest set bit
                if uncovered & 1:
                    crit_idx = 0
                else:
                    temp = uncovered
                    idx = 0
                    while (temp & 1) == 0:
                        temp >>= 1
                        idx += 1
                    crit_idx = idx
            
            crit_tsub = all_tsubs[crit_idx]
            
            # Generate all k-subsets containing crit_tsub
            crit_set = set(crit_tsub)
            remaining_points = [x for x in range(v) if x not in crit_set]
            
            best_block = None
            best_gain = -1
            best_mask = 0
            
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(crit_tsub + combo))
                mask = get_block_mask(block)
                
                # Gain = number of newly covered t-subsets
                new_cover = mask & uncovered
                gain = bin(new_cover).count('1')
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_mask = mask
            
            if best_block is None or best_gain <= 0:
                break
            
            blocks.append(list(best_block))
            block_masks.append(best_mask)
            
            # Update uncovered: clear the bits in best_mask
            uncovered &= ~best_mask
        
        # Local Search: Remove redundant blocks and try swaps
        if blocks:
            # Build coverage count using numpy
            cov_count = np.zeros(n_tsubs, dtype=np.int32)
            for b in blocks:
                idxs = get_block_tsub_indices(b)
                cov_count[idxs] += 1
            
            final_blocks = list(blocks)
            improved = True
            
            # Phase 1: Remove redundant blocks
            while improved and time.time() < deadline - 3.0:
                improved = False
                # Recalculate counts
                cov_count = np.zeros(n_tsubs, dtype=np.int32)
                for b in final_blocks:
                    idxs = get_block_tsub_indices(b)
                    cov_count[idxs] += 1
                
                # Try to remove each block
                for i in range(len(final_blocks) - 1, -1, -1):
                    if time.time() > deadline - 3.0:
                        break
                    b = final_blocks[i]
                    idxs = get_block_tsub_indices(b)
                    # Check if any t-subset in b has count == 1
                    if np.any(cov_count[idxs] == 1):
                        continue
                    # Redundant: remove it
                    final_blocks.pop(i)
                    improved = True
                    break  # Restart loop as counts changed
            
            # Phase 2: Try swaps to reduce block count
            # For each block, try to replace it with a different block that covers
            # the same "essential" t-subsets (those with count == 1)
            if time.time() < deadline - 2.0:
                cov_count = np.zeros(n_tsubs, dtype=np.int32)
                for b in final_blocks:
                    idxs = get_block_tsub_indices(b)
                    cov_count[idxs] += 1
                
                # Identify essential t-subsets (count == 1) for each block
                swap_improved = True
                max_swap_iters = 20
                for swap_iter in range(max_swap_iters):
                    if time.time() > deadline - 2.0:
                        break
                    swap_improved = False
                    
                    cov_count = np.zeros(n_tsubs, dtype=np.int32)
                    for b in final_blocks:
                        idxs = get_block_tsub_indices(b)
                        cov_count[idxs] += 1
                    
                    for i in range(len(final_blocks)):
                        if time.time() > deadline - 2.0:
                            break
                        
                        b = final_blocks[i]
                        idxs = get_block_tsub_indices(b)
                        essential = idxs[cov_count[idxs] == 1]
                        
                        if len(essential) == 0:
                            # No essential t-subsets, already handled by redundancy removal
                            continue
                        
                        # Try to find a replacement block that covers all essential t-subsets
                        # and has good coverage of other uncovered t-subsets
                        # Generate candidates: blocks containing all essential t-subsets
                        # This is complex, so use a heuristic:
                        # Pick a random essential t-subset and find blocks containing it
                        
                        if len(essential) > 0:
                            # Pick a random essential t-subset
                            ess_idx = rng.choice(essential)
                            ess_tsub = all_tsubs[ess_idx]
                            ess_set = set(ess_tsub)
                            
                            # Find blocks that contain this t-subset
                            remaining_points = [x for x in range(v) if x not in ess_set]
                            
                            best_swap_block = None
                            best_swap_score = -float('inf')
                            
                            # Sample candidates for speed
                            n_candidates = min(500, len(list(itertools.combinations(remaining_points, k - t))))
                            all_combos = list(itertools.combinations(remaining_points, k - t))
                            if len(all_combos) > n_candidates:
                                sampled = rng.sample(all_combos, n_candidates)
                            else:
                                sampled = all_combos
                            
                            for combo in sampled:
                                candidate = tuple(sorted(ess_tsub + combo))
                                if candidate == tuple(sorted(b)):
                                    continue
                                
                                cand_idxs = get_block_tsub_indices(candidate)
                                
                                # Check if candidate covers all essential t-subsets of b
                                if not np.all(np.isin(essential, cand_idxs)):
                                    continue
                                
                                # Score: number of t-subsets covered by candidate that are
                                # currently covered by exactly 1 block (would become 0 if we remove b)
                                # plus some weight for covering other t-subsets
                                overlap_with_essential = np.intersect1d(cand_idxs, essential)
                                score = len(overlap_with_essential)
                                
                                # Also consider how many unique t-subsets candidate covers
                                # that are not covered by other blocks
                                # This is approximate
                                
                                if score > best_swap_score:
                                    best_swap_score = score
                                    best_swap_block = candidate
                            
                            if best_swap_block is not None and best_swap_score > 0:
                                # Try the swap
                                old_block = final_blocks[i]
                                final_blocks[i] = list(best_swap_block)
                                
                                # Check if this allows removing another block
                                new_cov = np.zeros(n_tsubs, dtype=np.int32)
                                for b2 in final_blocks:
                                    idxs2 = get_block_tsub_indices(b2)
                                    new_cov[idxs2] += 1
                                
                                # Check if any block is now redundant
                                removed = False
                                for j in range(len(final_blocks) - 1, -1, -1):
                                    if j == i:
                                        continue
                                    b2 = final_blocks[j]
                                    idxs2 = get_block_tsub_indices(b2)
                                    if np.any(new_cov[idxs2] == 1):
                                        continue
                                    # Redundant
                                    final_blocks.pop(j)
                                    removed = True
                                    swap_improved = True
                                    break
                                
                                if not removed:
                                    # Swap didn't help, revert
                                    final_blocks[i] = old_block
                                else:
                                    break  # Restart swap phase
                            
                    if not swap_improved:
                        break
            
            if len(final_blocks) < best_count:
                best_count = len(final_blocks)
                best_blocks = final_blocks

    return best_blocks
