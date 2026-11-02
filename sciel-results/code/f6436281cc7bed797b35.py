import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Covering design C(v,k,t) using optimized greedy + local search.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Precompute all t-subsets as tuples
    all_tsubs = list(itertools.combinations(range(v), t))
    total_tsubs = len(all_tsubs)
    
    # Coverage count: how many blocks cover each t-subset
    coverage = [0] * total_tsubs
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    blocks = []
    
    def get_gain(block, x):
        """Count how many currently uncovered t-subsets would be covered by adding x."""
        gain = 0
        for rest in itertools.combinations(block, t - 1):
            ts = tuple(sorted(rest + (x,)))
            idx = tsub_index[ts]
            if coverage[idx] == 0:
                gain += 1
        return gain
    
    def add_block(block):
        """Add a block and update coverage counts."""
        blocks.append(block)
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            coverage[idx] += 1
    
    def remove_block(block_idx):
        """Remove a block and update coverage counts."""
        block = blocks.pop(block_idx)
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            coverage[idx] -= 1
    
    # Greedy phase
    while time.time() < deadline:
        # Find first uncovered t-subset
        first_uncovered_idx = -1
        for i in range(total_tsubs):
            if coverage[i] == 0:
                first_uncovered_idx = i
                break
        
        if first_uncovered_idx == -1:
            break  # All covered
        
        first_uncovered = all_tsubs[first_uncovered_idx]
        block = list(first_uncovered)
        
        # Add points greedily
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x not in block:
                    g = get_gain(block, x)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
            block.append(best_x)
        
        block.sort()
        add_block(block)
    
    # Local search: try to remove redundant blocks
    # A block is redundant if removing it doesn't leave any t-subset uncovered
    def is_redundant(block_idx):
        block = blocks[block_idx]
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            if coverage[idx] <= 1:
                return False
        return True
    
    # Try removing blocks (multiple passes)
    for _ in range(5):  # Limited passes
        if time.time() >= deadline:
            break
        removed_any = True
        while removed_any and time.time() < deadline:
            removed_any = False
            for i in range(len(blocks) - 1, -1, -1):
                if is_redundant(i):
                    remove_block(i)
                    removed_any = True
    
    return blocks
