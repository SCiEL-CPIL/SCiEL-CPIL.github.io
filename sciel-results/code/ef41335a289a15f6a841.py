import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    best_blocks = None
    best_count = float('inf')
    
    # Precompute index mapping for t-subsets for faster lookup
    # Using a set of tuples for membership testing
    
    # Multiple random trials
    trial = 0
    while time.time() < deadline:
        trial += 1
        blocks, valid = _greedy_cover(v, k, t, all_tsubsets, deadline)
        if valid:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                # Early stop if we hit known best
                if best_count <= 76:
                    break
        
        # If we've done many trials and no improvement, could stop
        if trial > 50 and best_count <= 78:
            break
    
    if best_blocks is None:
        # Fallback deterministic greedy
        blocks, valid = _greedy_cover_deterministic(v, k, t, all_tsubsets, deadline)
        if valid:
            best_blocks = blocks
    
    return best_blocks if best_blocks else []

def _greedy_cover(v, k, t, all_tsubsets, deadline):
    """Randomized greedy: seed with random uncovered subset, then greedily add points."""
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            return None, False
        
        # Pick a random uncovered subset to seed
        seed = random.choice(tuple(uncovered))
        block = list(seed)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() > deadline:
                return None, False
            
            # Compute gain for each candidate x not in block
            # Gain(x) = number of uncovered t-subsets s such that x in s and (s \ {x}) is a subset of block
            # Equivalently: for each (t-1)-subset p of block, count how many x make (p + (x,)) an uncovered t-subset
            
            t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            
            best_x = -1
            best_gain = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for p in t_minus_1_subsets:
                    s = tuple(sorted(p + (x,)))
                    if s in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain == 0:
                # No point helps; try a different approach or fail
                return None, False
            
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Remove covered t-subsets
        for s in itertools.combinations(block, t):
            uncovered.discard(s)
    
    return blocks, True

def _greedy_cover_deterministic(v, k, t, all_tsubsets, deadline):
    """Deterministic greedy: always seed with the smallest uncovered subset."""
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            return None, False
        
        # Pick the smallest uncovered subset (lexicographic)
        seed = min(uncovered)
        block = list(seed)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() > deadline:
                return None, False
            
            t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            
            best_x = -1
            best_gain = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for p in t_minus_1_subsets:
                    s = tuple(sorted(p + (x,)))
                    if s in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain == 0:
                return None, False
            
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(tuple(block))
        
        for s in itertools.combinations(block, t):
            uncovered.discard(s)
    
    return blocks, True
