"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _all_t_subsets(v, t):
    """Generate all t-subsets as tuples."""
    return itertools.combinations(range(v), t)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s kill

    # Precompute all t-subsets
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    
    # Map each t-subset to an index for fast lookup
    ts_index = {ts: i for i, ts in enumerate(all_ts)}
    
    # Coverage array: for each t-subset, how many blocks cover it
    covered = [0] * n_ts
    
    # Precompute for each point x, the list of t-subset indices that contain x
    # This is expensive, so we compute lazily or use a different approach
    
    blocks = []
    
    # Greedy construction with local improvement
    # Phase 1: Greedy covering
    while True:
        if time.time() > deadline - 30:
            break
        
        # Find uncovered t-subsets
        uncovered = [i for i in range(n_ts) if covered[i] == 0]
        if not uncovered:
            break
        
        # Pick the first uncovered subset as seed
        seed_ts = all_ts[uncovered[0]]
        block = list(seed_ts)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        remaining = [x for x in range(v) if x not in block]
        
        while len(block) < k and remaining:
            if time.time() > deadline - 30:
                break
            
            best_gain = -1
            best_x = remaining[0]
            
            for x in remaining:
                # Count how many currently uncovered t-subsets would be covered
                # by adding x to the current partial block
                gain = 0
                partial = block + [x]
                # A t-subset of partial is covered if it's a subset of the final block
                # For now, count t-subsets of the current partial block that are uncovered
                for ts in itertools.combinations(partial, t):
                    idx = ts_index.get(ts)
                    if idx is not None and covered[idx] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_gain <= 0:
                # No improvement possible; just pick a random remaining point
                best_x = random.choice(remaining)
            
            block.append(best_x)
            remaining.remove(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update coverage
        for ts in itertools.combinations(block, t):
            idx = ts_index.get(ts)
            if idx is not None:
                covered[idx] += 1
    
    # Phase 2: Local search - try to remove redundant blocks
    if time.time() < deadline - 10:
        # Try removing each block if it doesn't break coverage
        i = 0
        while i < len(blocks) and time.time() < deadline - 10:
            # Check if block i is redundant
            block_i = blocks[i]
            is_redundant = True
            for ts in itertools.combinations(block_i, t):
                idx = ts_index.get(ts)
                if idx is not None and covered[idx] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                for ts in itertools.combinations(block_i, t):
                    idx = ts_index.get(ts)
                    if idx is not None:
                        covered[idx] -= 1
                blocks.pop(i)
            else:
                i += 1
    
    # Phase 3: Local search - try replacing blocks with better ones
    if time.time() < deadline - 5:
        # For each block, try to find a replacement that covers the same t-subsets
        # but also covers additional uncovered ones
        for i in range(len(blocks)):
            if time.time() > deadline - 5:
                break
            # Skip for now due to time constraints
    
    return blocks


# EVOLVE-BLOCK-END
