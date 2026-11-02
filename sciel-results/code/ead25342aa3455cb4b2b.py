"""Improved covering design solver using randomized greedy with local search."""
import time
import random
import itertools
from collections import defaultdict


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
    # point_mask[p] = 1 << p
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
    # (t-1)-subset -> list of t-subset indices
    # This is useful for gain computation
    tminus1_to_tsubs = defaultdict(list)
    for i, ts in enumerate(all_tsubsets):
        ts_list = list(ts)
        for j in range(t):
            sub = tuple(sorted(ts_list[:j] + ts_list[j+1:]))
            tminus1_to_tsubs[sub].append(i)
    
    # Greedy with randomization
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = bytearray(num_tsubsets)  # 1 = uncovered
        for i in range(num_tsubsets):
            uncovered[i] = 1
        
        blocks = []
        remaining = num_tsubsets
        
        while remaining > 0:
            if time.time() > deadline:
                return None  # timeout
            
            # Pick a random uncovered t-subset to seed the block
            idx = rng.randrange(num_tsubsets)
            while uncovered[idx] == 0:
                idx = (idx + 1) % num_tsubsets
            
            seed_tsub = all_tsubsets[idx]
            block = list(seed_tsub)
            rng.shuffle(block)
            
            # Greedily add points to maximize gain
            while len(block) < k:
                if time.time() > deadline:
                    return None
                
                best_gain = -1
                best_point = -1
                block_set = set(block)
                block_mask = 0
                for p in block:
                    block_mask |= point_mask[p]
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Calculate gain: number of uncovered t-subsets S 
                    # where x ∈ S and S\{x} ⊆ block
                    gain = 0
                    for ts_idx in point_tsubs[x]:
                        if uncovered[ts_idx] == 0:
                            continue
                        # Check if ts\{x} ⊆ block using bitmask
                        ts_m = tsub_mask[ts_idx]
                        if (ts_m & ~point_mask[x]) & ~block_mask == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and best_point != -1:
                        # Tie-breaking: random
                        if rng.random() < 0.5:
                            best_point = x
                
                if best_point == -1 or best_gain <= 0:
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
            for ts_idx in range(num_tsubsets):
                if uncovered[ts_idx] == 0:
                    continue
                if tsub_mask[ts_idx] & ~block_mask == 0:
                    uncovered[ts_idx] = 0
                    remaining -= 1
        
        return blocks
    
    def remove_redundant(blocks, time_limit=None):
        """Remove blocks whose t-subsets are all covered by other blocks."""
        if time_limit is None:
            time_limit = deadline
        n = len(blocks)
        if n == 0:
            return blocks
        
        # Build coverage: for each t-subset, which blocks cover it
        # tsub_covered_by[i] = set of block indices covering t-subset i
        tsub_covered_by = [set() for _ in range(num_tsubsets)]
        
        block_masks = []
        for b in blocks:
            m = 0
            for p in b:
                m |= point_mask[p]
            block_masks.append(m)
        
        for bi, bm in enumerate(block_masks):
            for ts_idx in range(num_tsubsets):
                if tsub_mask[ts_idx] & ~bm == 0:
                    tsub_covered_by[ts_idx].add(bi)
        
        # A block is redundant if every t-subset it covers is also covered by another block
        keep = [True] * n
        for bi in range(n):
            if not keep[bi]:
                continue
            if time.time() > time_limit:
                break
            bm = block_masks[bi]
            redundant = True
            for ts_idx in range(num_tsubsets):
                if tsub_mask[ts_idx] & ~bm == 0:
                    # This block covers ts_idx
                    if len(tsub_covered_by[ts_idx]) == 1:
                        # Only this block covers it, so not redundant
                        redundant = False
                        break
            if redundant:
                keep[bi] = False
        
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
        cover_count = [0] * num_tsubsets
        block_masks = []
        for b in blocks:
            m = 0
            for p in b:
                m |= point_mask[p]
            block_masks.append(m)
        
        for bm in block_masks:
            for ts_idx in range(num_tsubsets):
                if tsub_mask[ts_idx] & ~bm == 0:
                    cover_count[ts_idx] += 1
        
        # Try removing blocks in random order
        order = list(range(n))
        rng.shuffle(order)
        
        kept = [True] * n
        for bi in order:
            if time.time() > time_limit:
                break
            if not kept[bi]:
                continue
            bm = block_masks[bi]
            # Check if removing this block leaves any t-subset uncovered
            can_remove = True
            for ts_idx in range(num_tsubsets):
                if tsub_mask[ts_idx] & ~bm == 0:
                    if cover_count[ts_idx] == 1:
                        can_remove = False
                        break
            if can_remove:
                kept[bi] = False
                for ts_idx in range(num_tsubsets):
                    if tsub_mask[ts_idx] & ~bm == 0:
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
