import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 120  # seconds, keep a 5s margin
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while True:
        current_time = time.time()
        if current_time - start_time > deadline - 5:
            break
            
        blocks = _greedy_cover(v, k, t, seed)
        blocks = _remove_redundant_blocks(blocks, v, k, t)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        seed += 1
        
    if best_blocks is None:
        best_blocks = _greedy_cover(v, k, t, 0)
        best_blocks = _remove_redundant_blocks(best_blocks, v, k, t)
        
    return best_blocks

def _greedy_cover(v, k, t, seed):
    rng = np.random.default_rng(seed)
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        # Convert to list for random choice. 
        # To avoid repeated conversion, we could maintain a list, but set removal is O(1) average.
        # Converting 2000 items is fast.
        start_subset = tuple(rng.choice(list(uncovered), size=1)[0])
        block = list(start_subset)
        
        while len(block) < k:
            best_gain = -1
            best_candidates = []
            
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                # Check subsets formed by x and (t-1) elements from block
                for S in itertools.combinations(block, t-1):
                    candidate_subset = tuple(sorted(S + (x,)))
                    if candidate_subset in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(x)
            
            if best_gain == 0:
                candidates = [x for x in range(v) if x not in block]
                if not candidates: break
                x = rng.choice(candidates)
            else:
                x = rng.choice(best_candidates)
                
            block.append(x)
            
            # Update uncovered
            block_old = block[:-1]
            for S in itertools.combinations(block_old, t-1):
                subset_to_remove = tuple(sorted(S + (x,)))
                if subset_to_remove in uncovered:
                    uncovered.remove(subset_to_remove)
                    
        block.sort()
        blocks.append(block)
        
    return blocks

def _remove_redundant_blocks(blocks, v, k, t):
    if not blocks:
        return []
        
    subset_counts = {}
    for block in blocks:
        for subset in itertools.combinations(block, t):
            subset_counts[subset] = subset_counts.get(subset, 0) + 1
            
    remaining_blocks = list(blocks)
    changed = True
    while changed:
        changed = False
        for i in range(len(remaining_blocks) - 1, -1, -1):
            block = remaining_blocks[i]
            is_redundant = True
            for subset in itertools.combinations(block, t):
                if subset_counts[subset] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                for subset in itertools.combinations(block, t):
                    subset_counts[subset] -= 1
                remaining_blocks.pop(i)
                changed = True
                
    return remaining_blocks
