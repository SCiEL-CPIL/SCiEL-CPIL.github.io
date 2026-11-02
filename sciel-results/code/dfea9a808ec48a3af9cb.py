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
    # point_tsubs[p] = list of t-subset indices containing p
    point_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        for p in ts:
            point_tsubs[p].append(i)
    
    # Precompute for each t-subset, its bitmask representation for fast subset checking
    # tsub_mask[i] = bitmask of points in t-subset i
    tsub_mask = [0] * num_tsubsets
    for i, ts in enumerate(all_tsubsets):
        mask = 0
        for p in ts:
            mask |= (1 << p)
        tsub_mask[i] = mask
    
    # For each point x, precompute the list of (tsub_idx, required_mask) pairs
    # where required_mask = tsub_mask[idx] & ~(1 << x)
    # gain(x, block_mask) = count of uncovered i where tsub_mask[i] & ~block_mask == 0
    # and x is in tsub i
    point_gain_data = [[] for _ in range(v)]
    for x in range(v):
        for ts_idx in point_tsubs[x]:
            req_mask = tsub_mask[ts_idx] & ~(1 << x)
            point_gain_data[x].append((ts_idx, req_mask))
    
    # Greedy with randomization and local search
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
            # Find an uncovered one
            idx = rng.randrange(num_tsubsets)
            while uncovered[idx] == 0:
                idx = (idx + 1) % num_tsubsets
            
            seed_tsub = all_tsubsets[idx]
            block = list(seed_tsub)
            rng.shuffle(block)
            
            # Greedily add points to maximize gain
            block_mask = 0
            for p in block:
                block_mask |= (1 << p)
            
            while len(block) < k:
                if time.time() > deadline:
                    return None
                
                best_gain = -1
                best_point = -1
                
                for x in range(v):
                    if (block_mask >> x) & 1:
                        continue
                    
                    # Calculate gain: number of uncovered t-subsets S 
                    # where x ∈ S and S\{x} ⊆ block
                    gain = 0
                    for ts_idx, req_mask in point_gain_data[x]:
                        if uncovered[ts_idx] == 0:
                            continue
                        # Check if req_mask ⊆ block_mask
                        if (req_mask & ~block_mask) == 0:
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
                    candidates = [x for x in range(v) if not ((block_mask >> x) & 1)]
                    if candidates:
                        best_point = rng.choice(candidates)
                    else:
                        break
                
                block.append(best_point)
                block_mask |= (1 << best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered using bitmask
            for ts_idx in range(num_tsubsets):
                if uncovered[ts_idx] == 0:
                    continue
                if (tsub_mask[ts_idx] & ~block_mask) == 0:
                    uncovered[ts_idx] = 0
                    remaining -= 1
        
        return blocks
    
    # Redundancy removal: remove blocks whose t-subsets are all covered by other blocks
    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # For each block, compute which t-subsets it covers
        block_tsub_sets = []
        for block in blocks:
            block_set = set(block)
            block_mask = 0
            for p in block:
                block_mask |= (1 << p)
            covered = []
            for ts_idx in range(num_tsubsets):
                if (tsub_mask[ts_idx] & ~block_mask) == 0:
                    covered.append(ts_idx)
            block_tsub_sets.append(covered)
        
        # For each t-subset, track which blocks cover it
        tsub_to_blocks = defaultdict(set)
        for i, covered in enumerate(block_tsub_sets):
            for ts_idx in covered:
                tsub_to_blocks[ts_idx].add(i)
        
        # Remove redundant blocks
        new_blocks = []
        for i, block in enumerate(blocks):
            # Check if all t-subsets in this block are covered by other blocks
            redundant = True
            for ts_idx in block_tsub_sets[i]:
                if tsub_to_blocks[ts_idx] == {i}:
                    redundant = False
                    break
            if not redundant:
                new_blocks.append(block)
        
        return new_blocks
    
    # Try multiple random seeds and keep the best
    best_blocks = None
    best_count = float('inf')
    
    # First, try a deterministic greedy (seed 0)
    blocks = greedy_cover(0)
    if blocks is not None:
        blocks = remove_redundant(blocks)
        best_blocks = blocks
        best_count = len(blocks)
    
    # Try more seeds with time remaining
    seed = 1
    while time.time() < deadline - 10:
        blocks = greedy_cover(seed)
        if blocks is not None:
            blocks = remove_redundant(blocks)
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
        seed += 1
    
    if best_blocks is None:
        # Fallback: return empty (shouldn't happen)
        return []
    
    return best_blocks
