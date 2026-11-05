"""Optimized covering design C(24,7,3) solver using vectorized greedy + local search."""
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin within 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute coverage matrix: for each possible block, which t-subsets it covers
    # A block covers a t-subset if the t-subset is a subset of the block
    # For k=7, t=3: each block covers C(7,3) = 35 t-subsets
    
    # Precompute for each block (represented as a bitmask or frozenset), the list of t-subset indices it covers
    # Since v=24, k=7, there are C(24,7) = 346104 possible blocks - too many to precompute all
    
    # Instead, use a smarter approach:
    # 1. Vectorized greedy: at each step, pick the block that covers the most uncovered t-subsets
    # 2. Use bitsets for efficiency
    
    # Represent t-subsets as bitmasks for fast subset checking
    # For v=24, we can use a 24-bit integer to represent a subset
    
    # Precompute bitmask for each t-subset
    tsubset_bitmasks = np.zeros(n_tsubsets, dtype=np.int32)
    for i, ts in enumerate(all_tsubsets):
        mask = 0
        for p in ts:
            mask |= (1 << p)
        tsubset_bitmasks[i] = mask
    
    # Greedy set cover with vectorized scoring
    uncovered = np.ones(n_tsubsets, dtype=np.uint8)
    blocks = []
    block_bitmasks = []  # Store bitmask for each selected block
    
    def time_remaining():
        return deadline - time.time()
    
    # Number of t-subsets per block: C(k, t)
    ts_per_block = 1
    for i in range(t):
        ts_per_block *= (k - i)
        ts_per_block //= (i + 1)
    
    # For greedy, we need to evaluate which block to add next.
    # Key insight: a block B covers t-subset S iff S ⊆ B.
    # For a candidate block B, the gain = number of uncovered t-subsets S with S ⊆ B.
    
    # Efficient approach: for each point, precompute which t-subsets contain it.
    # Then for a block B = {b1, b2, ..., bk}, a t-subset S is covered by B iff S ⊆ B.
    
    # Let's use a different strategy: 
    # At each step, find the most uncovered t-subset, then find the best block containing it.
    
    while np.any(uncovered) and time_remaining() > 5:
        # Find the first uncovered t-subset
        uncov_indices = np.where(uncovered == 1)[0]
        if len(uncov_indices) == 0:
            break
        
        # Pick a random uncovered t-subset to anchor
        anchor_idx = uncov_indices[random.randint(0, len(uncov_indices) - 1)]
        anchor = all_tsubsets[anchor_idx]
        anchor_mask = tsubset_bitmasks[anchor_idx]
        
        # Build a block containing the anchor points
        # Start with anchor, greedily add points that maximize coverage
        block = set(anchor)
        block_mask = anchor_mask
        
        # Precompute: for each point x, which t-subsets are "completable" by adding x
        # i.e., t-subsets that contain x and whose other t-1 points are in the current block
        
        while len(block) < k and time_remaining() > 5:
            best_point = None
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                
                # Gain = number of uncovered t-subsets S such that S ⊆ (block ∪ {x}) and x ∈ S
                # Equivalently: S contains x, and S \ {x} ⊆ block
                gain = 0
                x_mask = 1 << x
                for ts_i in range(n_tsubsets):
                    if uncovered[ts_i] == 0:
                        continue
                    ts_mask = tsubset_bitmasks[ts_i]
                    if ts_mask & x_mask == 0:
                        continue  # x not in this t-subset
                    # Check if ts_mask \ {x} ⊆ block_mask
                    if (ts_mask & ~x_mask) & ~block_mask == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None or best_gain <= 0:
                # Pick a random point
                candidates = [x for x in range(v) if x not in block]
                if candidates:
                    best_point = random.choice(candidates)
                else:
                    break
            
            block.add(best_point)
            block_mask |= (1 << best_point)
        
        block = sorted(block)
        
        # Check if this block is already in our list
        if block in blocks:
            continue
            
        blocks.append(block)
        block_bitmasks.append(block_mask)
        
        # Update uncovered: mark all t-subsets contained in this block as covered
        for ts_i in range(n_tsubsets):
            if uncovered[ts_i] == 1:
                if tsubset_bitmasks[ts_i] & ~block_mask == 0:
                    uncovered[ts_i] = 0
    
    # Local search: remove redundant blocks
    if time_remaining() > 10:
        changed = True
        while changed and time_remaining() > 10:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time_remaining() < 10:
                    break
                
                # Check if removing block i still covers all its t-subsets
                removable = True
                block_i_mask = block_bitmasks[i]
                
                for ts_i in range(n_tsubsets):
                    # Check if this t-subset is covered by block i
                    if tsubset_bitmasks[ts_i] & ~block_i_mask != 0:
                        continue  # Not covered by block i
                    
                    # Check if covered by another block
                    found_in_other = False
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if tsubset_bitmasks[ts_i] & ~block_bitmasks[j] == 0:
                            found_in_other = True
                            break
                    
                    if not found_in_other:
                        removable = False
                        break
                
                if removable:
                    blocks.pop(i)
                    block_bitmasks.pop(i)
                    changed = True
    
    # Second pass of redundancy removal
    if time_remaining() > 15:
        changed = True
        while changed and time_remaining() > 15:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time_remaining() < 15:
                    break
                
                removable = True
                block_i_mask = block_bitmasks[i]
                
                for ts_i in range(n_tsubsets):
                    if tsubset_bitmasks[ts_i] & ~block_i_mask != 0:
                        continue
                    
                    found_in_other = False
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if tsubset_bitmasks[ts_i] & ~block_bitmasks[j] == 0:
                            found_in_other = True
                            break
                    
                    if not found_in_other:
                        removable = False
                        break
                
                if removable:
                    blocks.pop(i)
                    block_bitmasks.pop(i)
                    changed = True
    
    return blocks
