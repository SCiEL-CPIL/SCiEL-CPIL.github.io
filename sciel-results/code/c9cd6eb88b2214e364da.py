"""Covering design C(19,7,4) - greedy with local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    rng = random.Random(42)
    
    # Precompute all t-subsets
    all_tsubsets = set(itertools.combinations(range(v), t))
    
    # Greedy construction
    uncovered = set(all_tsubsets)
    blocks = []
    used_blocks = set()
    
    while uncovered and time.time() < deadline:
        # Pick the uncovered t-subset with highest "priority" (first in sorted order for determinism)
        # But we can try to pick one that allows a better block
        # For speed, just pick the first uncovered one
        target = min(uncovered)
        
        # Build a block containing this target
        block_set = set(target)
        remaining_needed = k - t
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        for _ in range(remaining_needed):
            if time.time() > deadline:
                break
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many new t-subsets would be covered
                gain = 0
                # A new t-subset containing x and t-1 elements from current block
                for rest in itertools.combinations(block_set, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is not None:
                block_set.add(best_x)
        
        block = tuple(sorted(block_set))
        if len(block) < k:
            # Pad with arbitrary points
            for x in range(v):
                if len(block_set) >= k:
                    break
                if x not in block_set:
                    block_set.add(x)
            block = tuple(sorted(block_set))
        
        if block in used_blocks:
            continue
        used_blocks.add(block)
        blocks.append(list(block))
        
        # Remove covered t-subsets
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    
    # Local search: remove redundant blocks
    if time.time() < deadline:
        changed = True
        while changed and time.time() < deadline:
            changed = False
            indices = list(range(len(blocks)))
            rng.shuffle(indices)
            for i in indices:
                if time.time() > deadline:
                    break
                # Check if block i is redundant
                block_i = blocks[i]
                # Build coverage count for t-subsets in block_i
                tsubs_in_block = list(itertools.combinations(block_i, t))
                
                # Count how many other blocks cover each of these t-subsets
                redundant = True
                for ts in tsubs_in_block:
                    count = 0
                    for j, b in enumerate(blocks):
                        if j == i:
                            continue
                        if ts in itertools.combinations(b, t):
                            count += 1
                            if count > 0:
                                break
                    if count == 0:
                        redundant = False
                        break
                
                if redundant:
                    del blocks[i]
                    changed = True
                    break  # restart scan
    
    return blocks
# EVOLVE-BLOCK-END
