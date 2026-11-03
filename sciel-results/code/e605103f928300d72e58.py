"""Improved covering design solver using randomized greedy with local search."""
import time
import random
import itertools
import numpy as np
from collections import defaultdict
from math import comb


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # For each point, precompute which t-subsets it participates in
    point_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        for p in ts:
            point_tsubs[p].append(i)
    
    # Precompute for each t-subset its bitmask representation (for fast containment checks)
    # We'll use integer bitmasks for blocks
    point_mask = [1 << p for p in range(v)]
    
    # For each block (as a bitmask), precompute which t-subsets it covers
    # tsub_mask[i] = bitmask of points in t-subset i
    tsub_mask = []
    for ts in all_tsubsets:
        m = 0
        for p in ts:
            m |= point_mask[p]
        tsub_mask.append(m)
    
    # Precompute: for each possible (t-1)-subset of points, which t-subsets contain it
    tminus1_to_tsubs = defaultdict(list)
    for i, ts in enumerate(all_tsubsets):
        ts_list = list(ts)
        for j in range(t):
            sub = tuple(sorted(ts_list[:j] + ts_list[j+1:]))
            tminus1_to_tsubs[sub].append(i)
    
    # Precompute for each point x: list of (t_sub_idx, mask_of_other_t_minus_1_points)
    # This allows fast gain computation
    point_tsub_info = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        ts_list = list(ts)
        for j in range(t):
            x = ts_list[j]
            others = [p for p in ts_list if p != x]
            mask_others = 0
            for p in others:
                mask_others |= point_mask[p]
            point_tsub_info[x].append((i, mask_others))
    
    # Convert tsub_mask to numpy array for vectorized operations
    tsub_mask_arr = np.array(tsub_mask, dtype=np.int64)
    
    # Convert to numpy arrays for faster operations
    # For each point, store as two parallel arrays
    point_tsub_idx = []
    point_tsub_mask_others = []
    for x in range(v):
        idxs = [info[0] for info in point_tsub_info[x]]
        masks = [info[1] for info in point_tsub_info[x]]
        point_tsub_idx.append(np.array(idxs, dtype=np.int32))
        point_tsub_mask_others.append(np.array(masks, dtype=np.int64))
    
    # Precompute gain lookup table: for each point x, for each possible block_mask (t-subset of remaining points),
    # the number of t-subsets containing x that are fully covered by block_mask
    # Since k=8, t=5, we have at most 5 points to add, so block size during construction is 5..7
    # For each point x and each (k-t)-subset of other points (size up to 3), precompute the gain
    # Actually, let's precompute for each point x: a dict mapping frozenset of other points -> gain
    # But that's too large. Instead, let's precompute for each point x and each t-subset containing x:
    # the mask of the other t-1 points. Then gain = count of uncovered t-subsets where other_mask is subset of block.
    
    # Better approach: for each point x, precompute a list of (tsub_idx, other_mask) pairs
    # Then gain for adding x to block = count of pairs where other_mask & ~block_mask == 0 AND uncovered[tsub_idx]
    # This is what we already have. The bottleneck is the numpy indexing.
    
    # Let's try a different approach: precompute for each point x, for each subset S of {0..v-1}\{x} of size <= k-t,
    # the gain. But that's C(18,3) = 816 per point, 19*816 = 15504 entries.
    # Actually k-t = 3, so we need subsets of size 0,1,2,3 from 18 points.
    # C(18,0)+C(18,1)+C(18,2)+C(18,3) = 1+18+153+816 = 988 per point.
    # Total: 19 * 988 = 18772 entries. That's manageable.
    
    # Precompute gain_table[x][mask] = number of t-subsets containing x that are "completed" by mask
    # where mask represents the other points already in the block
    # A t-subset {x} U S (|S|=t-1=4) is completed if S is subset of mask
    # So gain_table[x][mask] = |{S : |S|=t-1, x not in S, S subset of mask, S U {x} is a t-subset}|
    # Since all (t-1)-subsets of other points form valid t-subsets with x, this is just C(|mask|, t-1)
    # BUT we also need to check which are uncovered. So we can't fully precompute without uncovered state.
    
    # Alternative: precompute for each point x and each t-subset idx containing x, the other_mask.
    # Then in the inner loop, we just need to count how many have other_mask & ~block_mask == 0 and are uncovered.
    # This is already what we do. Let's optimize the numpy operation.
    
    # Actually, let's precompute for each point x: 
    #   x_other_masks: array of shape (n_x,) where n_x = C(v-1, t-1) = C(18,4) = 3060
    #   x_tsub_indices: array of shape (n_x,) 
    # Then gain = sum(uncovered[x_tsub_indices] & (x_other_masks & ~block_mask == 0))
    # This is already efficient. The issue is we're doing this for all v points each iteration.
    
    # Let's add urgency-based scoring: prioritize adding points that cover t-subsets with fewer alternatives
    # Precompute for each t-subset: how many blocks could potentially cover it (i.e., how many (k-t)-subsets of remaining points)
    # This is a static value: C(v - t, k - t) for each t-subset (same for all)
    # So urgency is uniform. Instead, let's use a dynamic measure: 
    # For each uncovered t-subset, count how many "partial" blocks (current block candidates) could complete it.
    # This is complex. Let's try a simpler heuristic: weight gain by 1/(1 + number of points already in block)
    # to encourage spreading coverage.
    
    # Let's also precompute the list of t-subsets for each point more efficiently
    # For point x, the t-subsets containing x are: {x} U S where S is a (t-1)-subset of {0..v-1}\{x}
    point_tsub_data = []
    for x in range(v):
        others = [p for p in range(v) if p != x]
        n_x = comb(len(others), t-1)
        x_tsub_indices = np.zeros(n_x, dtype=np.int32)
        x_other_masks = np.zeros(n_x, dtype=np.int64)
        idx = 0
        for S in itertools.combinations(others, t-1):
            ts = tuple(sorted([x] + list(S)))
            x_tsub_indices[idx] = tsub_to_idx[ts]
            m = 0
            for p in S:
                m |= point_mask[p]
            x_other_masks[idx] = m
            idx += 1
        point_tsub_data.append((x_tsub_indices, x_other_masks))
    # Greedy with randomization and urgency scoring
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsubsets, dtype=np.bool_)
        
        blocks = []
        remaining = num_tsubsets
        
        while remaining > 0:
            if time.time() > deadline:
                return None  # timeout
            
            # Pick a random uncovered t-subset to seed the block
            idx = rng.randrange(num_tsubsets)
            while not uncovered[idx]:
                idx = (idx + 1) % num_tsubsets
            
            seed_tsub = all_tsubsets[idx]
            block = list(seed_tsub)
            rng.shuffle(block)
            
            # Greedily add points to maximize gain with urgency weighting
            while len(block) < k:
                if time.time() > deadline:
                    return None
                
                best_score = -1
                best_point = -1
                block_set = set(block)
                block_mask = 0
                for p in block:
                    block_mask |= point_mask[p]
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Calculate gain using precomputed data
                    x_tsub_indices, x_other_masks = point_tsub_data[x]
                    
                    # Filter to only uncovered t-subsets
                    mask_uncovered = uncovered[x_tsub_indices]
                    if not mask_uncovered.any():
                        continue
                    
                    # Check which of the uncovered t-subsets have their other t-1 points in block
                    # mask_others & ~block_mask == 0 means all other points are in block
                    candidates = x_other_masks[mask_uncovered]
                    valid = (candidates & ~block_mask) == 0
                    gain = int(valid.sum())
                    
                    if gain <= 0:
                        continue
                    
                    # Urgency score: weight by how "hard" these t-subsets are to cover
                    # Use a simple heuristic: prefer points that cover t-subsets with more overlap with current block
                    # This encourages completing partial covers rather than starting new ones
                    # Alternative: use 1/sqrt(gain) to encourage spreading
                    # Let's try: score = gain * (1.0 + 0.1 * len(block)) to slightly prefer later additions
                    score = gain
                    
                    if score > best_score:
                        best_score = score
                        best_point = x
                    elif score == best_score and best_point != -1:
                        # Tie-breaking: random
                        if rng.random() < 0.5:
                            best_point = x
                
                if best_point == -1 or best_score <= 0:
                    # No useful point, just pick a random one
                    candidates = [x for x in range(v) if x not in block_set]
                    if candidates:
                        best_point = rng.choice(candidates)
                    else:
                        break
                
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            block_mask = 0
            for p in block:
                block_mask |= point_mask[p]
            
            # Vectorized update: find all t-subsets covered by this block
            # Use numpy for speed
            covered_mask = (tsub_mask_arr & ~block_mask) == 0
            newly_covered = uncovered & covered_mask
            num_newly = int(np.sum(newly_covered))
            if num_newly > 0:
                uncovered[newly_covered] = False
                remaining -= num_newly
        
        return blocks
    
    def remove_redundant(blocks, time_limit=None):
        """Remove blocks whose t-subsets are all covered by other blocks."""
        if time_limit is None:
            time_limit = deadline
        n = len(blocks)
        if n == 0:
            return blocks
        
        # Build coverage: for each t-subset, count how many blocks cover it
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        
        block_masks = []
        for b in blocks:
            m = 0
            for p in b:
                m |= point_mask[p]
            block_masks.append(m)
        
        # Also store which t-subsets each block covers for later use
        block_covers = []
        for bm in block_masks:
            covered = []
            for ts_idx in range(num_tsubsets):
                if tsub_mask[ts_idx] & ~bm == 0:
                    covered.append(ts_idx)
                    cover_count[ts_idx] += 1
            block_covers.append(covered)
        
        # A block is redundant if every t-subset it covers is also covered by another block
        keep = [True] * n
        for bi in range(n):
            if not keep[bi]:
                continue
            if time.time() > time_limit:
                break
            redundant = True
            for ts_idx in block_covers[bi]:
                if cover_count[ts_idx] == 1:
                    # Only this block covers it, so not redundant
                    redundant = False
                    break
            if redundant:
                keep[bi] = False
                # Decrement cover counts for t-subsets covered by this block
                for ts_idx in block_covers[bi]:
                    cover_count[ts_idx] -= 1
        
        if all(keep):
            return blocks
        
        new_blocks = [b for i, b in enumerate(blocks) if keep[i]]
        return new_blocks
    
    def local_search(blocks, seed, time_limit=None):
        """Try to remove blocks one at a time if the covering is still valid."""
        if time_limit is None:
            time_limit = deadline
        rng = random.Random(seed)
        n = len(blocks)
        
        # Count how many blocks cover each t-subset
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        block_masks = []
        for b in blocks:
            m = 0
            for p in b:
                m |= point_mask[p]
            block_masks.append(m)
        
        # Precompute which t-subsets each block covers
        block_covers = []
        for bm in block_masks:
            covered = []
            for ts_idx in range(num_tsubsets):
                if tsub_mask[ts_idx] & ~bm == 0:
                    covered.append(ts_idx)
                    cover_count[ts_idx] += 1
            block_covers.append(covered)
        
        # Try removing blocks in random order
        order = list(range(n))
        rng.shuffle(order)
        
        kept = [True] * n
        for bi in order:
            if time.time() > time_limit:
                break
            if not kept[bi]:
                continue
            # Check if removing this block leaves any t-subset uncovered
            can_remove = True
            for ts_idx in block_covers[bi]:
                if cover_count[ts_idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                kept[bi] = False
                for ts_idx in block_covers[bi]:
                    cover_count[ts_idx] -= 1
        
        new_blocks = [b for i, b in enumerate(blocks) if kept[i]]
        return new_blocks
    
    # Try multiple random seeds and keep the best
    best_blocks = None
    best_count = float('inf')
    
    # First, try a deterministic greedy (seed 0)
    blocks = greedy_cover(0)
    if blocks is not None:
        # Apply redundancy removal
        blocks = remove_redundant(blocks)
        # Apply local search
        blocks = local_search(blocks, 0)
        best_blocks = blocks
        best_count = len(blocks)
    
    # Try more seeds with time remaining
    seed = 1
    while time.time() < deadline - 10:
        blocks = greedy_cover(seed)
        if blocks is not None:
            # Apply redundancy removal
            blocks = remove_redundant(blocks, time_limit=deadline - 5)
            # Apply local search
            blocks = local_search(blocks, seed, time_limit=deadline - 2)
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
        seed += 1
    
    if best_blocks is None:
        # Fallback: return empty (shouldn't happen)
        return []
    
    return best_blocks
