import time
import random
from itertools import combinations

def _solve_S4511():
    """
    Solves the Steiner system S(4,5,11) on points 0..10.
    Returns a list of 66 tuples, each being a 5-subset.
    """
    # 4-subsets of 0..10
    all_4subs = list(combinations(range(11), 4))
    subset_to_idx = {s: i for i, s in enumerate(all_4subs)}
    
    # All 5-subsets
    all_5subs = list(combinations(range(11), 5))
    
    # Precompute the 4-subsets for each 5-subset
    block_4subs = []
    for b in all_5subs:
        subs = [subset_to_idx[s] for s in combinations(b, 4)]
        block_4subs.append(subs)
    
    covered = [0] * 330 # 330 4-subsets
    blocks = []
    
    def backtrack():
        if len(blocks) == 66:
            return True
        
        # Find the first uncovered 4-subset
        # Optimization: We can just iterate through blocks and pick the first valid one
        # But to speed up, we should pick a block that covers the "most constrained" uncovered 4-subset.
        # Simple heuristic: Find the first uncovered 4-subset, and only consider blocks containing it.
        
        # Find first uncovered 4-subset
        target_idx = -1
        for i in range(330):
            if covered[i] == 0:
                target_idx = i
                break
        
        if target_idx == -1:
            return False # Should not happen if we have 66 blocks
            
        target_4sub = all_4subs[target_idx]
        
        # Find all blocks containing target_4sub
        # We can precompute a map from 4-subset to list of 5-subsets containing it
        # But doing it on the fly is fine for 11 points.
        
        candidates = []
        for i, b in enumerate(all_5subs):
            if target_4sub in b:
                # Check if b is valid (all its 4-subsets are uncovered)
                valid = True
                for idx in block_4subs[i]:
                    if covered[idx] != 0:
                        valid = False
                        break
                if valid:
                    candidates.append(i)
                    
        # Shuffle candidates for diversity? No, deterministic is fine for a unique solution structure.
        # Actually, S(4,5,11) is unique up to isomorphism.
        
        for idx in candidates:
            b = all_5subs[idx]
            # Add block
            blocks.append(b)
            for idx_4 in block_4subs[idx]:
                covered[idx_4] = 1
                
            if backtrack():
                return True
                
            # Remove block
            blocks.pop()
            for idx_4 in block_4subs[idx]:
                covered[idx_4] = 0
                
        return False
        
    if backtrack():
        return blocks
    else:
        return []

def _construct_witt_design():
    """
    Construct the Steiner system S(5,6,12) using the S(4,5,11) subdesign.
    """
    start_time = time.time()
    deadline = start_time + 100 # Safety margin
    
    # 1. Get S(4,5,11) blocks
    m11_blocks = _solve_S4511()
    
    if not m11_blocks:
        # Fallback to greedy if solver fails (should not happen)
        return _fallback_greedy()
        
    # 2. Construct S(5,6,12) blocks
    # Points 0..10 are from M11, point 11 is the new point.
    witt_blocks = []
    
    # Type 1: B U {11}
    for b in m11_blocks:
        witt_blocks.append(tuple(list(b) + [11]))
        
    # Type 2: {0..10} \ B
    points_11 = set(range(11))
    for b in m11_blocks:
        b_set = set(b)
        comp = points_11 - b_set
        witt_blocks.append(tuple(sorted(comp)))
        
    # Verify count
    if len(witt_blocks) != 132:
        return _fallback_greedy()
        
    # Verify no repeated 5-subsets
    # This is guaranteed by the construction, but a quick check is good.
    seen_5subs = set()
    for b in witt_blocks:
        for sub in combinations(b, 5):
            if sub in seen_5subs:
                # Conflict found, fallback
                return _fallback_greedy()
            seen_5subs.add(sub)
            
    if len(seen_5subs) != 792:
        return _fallback_greedy()
        
    if time.time() > deadline:
        return _fallback_greedy()
        
    return [list(b) for b in witt_blocks]

def _fallback_greedy():
    """
    Fallback greedy search if algebraic construction fails.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    all_5subsets = list(combinations(range(12), 5))
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    
    all_6subsets = list(combinations(range(12), 6))
    block_masks = []
    for b in all_6subsets:
        mask = 0
        for sub in combinations(b, 5):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    def greedy_packing(order):
        covered = 0
        blocks = []
        for idx in order:
            mask = block_masks[idx]
            if (covered & mask) == 0:
                blocks.append(all_6subsets[idx])
                covered |= mask
                if len(blocks) == 132:
                    break
        return blocks

    rng = random.Random(42)
    num_blocks = len(all_6subsets)
    best_blocks = []
    
    for attempt in range(2000):
        if time.time() > deadline:
            break
        order = list(range(num_blocks))
        rng.shuffle(order)
        blocks = greedy_packing(order)
        if len(blocks) > len(best_blocks):
            best_blocks = blocks
            if len(best_blocks) == 132:
                break
                
    return [list(b) for b in best_blocks]

def construct():
    """
    Construct a packing of 6-subsets on 12 points with no repeated 5-subset.
    Returns a list of distinct 6-subsets of 0..11.
    """
    return _construct_witt_design()
