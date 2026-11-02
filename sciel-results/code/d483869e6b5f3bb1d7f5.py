import itertools
import time
import random
import heapq
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill
    
    # 1. Precompute
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    # Map each block to the indices of t-subsets it covers
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    block_t_indices = []
    block_masks = []
    for b in all_blocks:
        idxs = [t_index[ts] for ts in itertools.combinations(b, t)]
        mask = 0
        for idx in idxs:
            mask |= (1 << idx)
        block_masks.append(mask)
        block_t_indices.append(idxs)
        
    # For fast lookup: which blocks contain a given t-subset?
    t_containing_blocks = [[] for _ in range(num_t)]
    for b_idx, b in enumerate(all_blocks):
        for idx in block_t_indices[b_idx]:
            t_containing_blocks[idx].append(b_idx)
            
    best_blocks = None
    best_size = float('inf')
    
    # 2. Greedy + Refinement Loop with multiple strategies
    while time.time() < deadline:
        # Strategy A: Standard Greedy with random seeds
        cover_count = [0] * num_t
        uncovered_mask = (1 << num_t) - 1
        current_blocks = []
        
        while uncovered_mask:
            if time.time() > deadline:
                break
            
            # Pick a few random uncovered t-subsets to consider
            candidates_seeds = []
            attempts = 0
            while len(candidates_seeds) < 5 and attempts < 50:
                idx = random.randint(0, num_t - 1)
                if (uncovered_mask >> idx) & 1:
                    candidates_seeds.append(idx)
                attempts += 1
            
            if not candidates_seeds:
                for i in range(num_t):
                    if (uncovered_mask >> i) & 1:
                        candidates_seeds.append(i)
                        break
            
            best_block_idx = -1
            best_gain = -1
            
            for seed_idx in candidates_seeds:
                for b_idx in t_containing_blocks[seed_idx]:
                    gain = (block_masks[b_idx] & uncovered_mask).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
            
            current_blocks.append(best_block_idx)
            b_mask = block_masks[best_block_idx]
            for idx in block_t_indices[best_block_idx]:
                cover_count[idx] += 1
                if cover_count[idx] == 1:
                    uncovered_mask &= ~(1 << idx)
        
        if time.time() > deadline:
            break
            
        # Refinement Phase 1: Randomized Deletion
        random.shuffle(current_blocks)
        for _ in range(2):
            random.shuffle(current_blocks)
            new_blocks = []
            for b_idx in current_blocks:
                can_remove = True
                for idx in block_t_indices[b_idx]:
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for idx in block_t_indices[b_idx]:
                        cover_count[idx] -= 1
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
        
        # Refinement Phase 2: Pair Replacement (2-for-1 swap)
        # Try to replace two blocks with one that covers all their unique t-subsets
        if time.time() < deadline - 5.0 and len(current_blocks) > 1:
            # Build a set of current block indices for quick lookup
            block_set = set(current_blocks)
            
            # For each pair of blocks, check if their union of t-subsets is covered by a single block
            # This is expensive, so we sample
            pairs_to_try = min(100, len(current_blocks) * (len(current_blocks) - 1) // 2)
            tried_pairs = set()
            improvements = 0
            
            while improvements < 10 and time.time() < deadline - 3.0:
                # Pick a random pair
                i, j = random.sample(range(len(current_blocks)), 2)
                if (i, j) in tried_pairs or (j, i) in tried_pairs:
                    continue
                tried_pairs.add((i, j))
                tried_pairs.add((j, i))
                
                b1 = current_blocks[i]
                b2 = current_blocks[j]
                
                # Get t-subsets covered by b1 or b2 but not by other blocks
                # Actually, we want to find a single block that covers all t-subsets 
                # that are ONLY covered by b1 or b2 (i.e., no other block covers them)
                
                # Collect t-subsets that have cover_count == 1 and are in b1 or b2
                unique_ts = set()
                for idx in block_t_indices[b1]:
                    if cover_count[idx] == 1:
                        unique_ts.add(idx)
                for idx in block_t_indices[b2]:
                    if cover_count[idx] == 1:
                        unique_ts.add(idx)
                
                if not unique_ts:
                    continue
                
                # Find a block that covers all these unique t-subsets
                # We can check if any block in t_containing_blocks for one of these ts covers all
                candidate_block = -1
                for ts_idx in list(unique_ts)[:3]:  # Check first few
                    for b_idx in t_containing_blocks[ts_idx]:
                        if b_idx == b1 or b_idx == b2:
                            continue
                        # Check if this block covers all unique_ts
                        b_mask = block_masks[b_idx]
                        all_covered = True
                        for ts_idx2 in unique_ts:
                            if not (b_mask >> ts_idx2) & 1:
                                all_covered = False
                                break
                        if all_covered:
                            candidate_block = b_idx
                            break
                    if candidate_block != -1:
                        break
                
                if candidate_block != -1:
                    # Remove b1 and b2, add candidate_block
                    # Update cover counts
                    for idx in block_t_indices[b1]:
                        cover_count[idx] -= 1
                    for idx in block_t_indices[b2]:
                        cover_count[idx] -= 1
                    for idx in block_t_indices[candidate_block]:
                        cover_count[idx] += 1
                    
                    # Update current_blocks
                    # Remove b1 and b2 (be careful with indices)
                    if i < j:
                        current_blocks.pop(j)
                        current_blocks.pop(i)
                    else:
                        current_blocks.pop(i)
                        current_blocks.pop(j)
                    current_blocks.append(candidate_block)
                    improvements += 1
        
        # Check if this is better
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            best_blocks = current_blocks[:]
            
            # If we've reached a good bound, we can stop early
            if best_size <= 140:
                break

    # Convert indices back to blocks
    if best_blocks is None:
        return []
        
    return [list(all_blocks[b_idx]) for b_idx in best_blocks]
