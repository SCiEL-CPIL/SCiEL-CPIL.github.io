"""Improved covering design C(19,8,5) using randomized greedy with deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

# Precompute all 5-subsets of 19 points as frozensets for fast membership
_ALL_5SETS = list(itertools.combinations(range(19), 5))
_ALL_5SETS_SET = set(_ALL_5SETS)
_NUM_5SETS = len(_ALL_5SETS_SET)  # 11628


def _count_new_tsubs(block, uncovered_set):
    """Count how many 5-subsets of block are in uncovered_set."""
    cnt = 0
    for sub in itertools.combinations(block, 5):
        if sub in uncovered_set:
            cnt += 1
    return cnt


def _greedy_cover(uncovered_set, k=8, v=19, rng=None):
    """Greedy covering: repeatedly pick block covering most uncovered 5-sets."""
    if rng is None:
        rng = random.Random()
    blocks = []
    uncovered = set(uncovered_set)
    while uncovered:
        # Find the 5-subset with the fewest candidate blocks (heuristic for harder ones)
        # Actually, let's just pick the first uncovered one and build a block around it
        # Better: pick a random uncovered 5-set to diversify
        target = rng.choice(list(uncovered)) if len(uncovered) > 1 else next(iter(uncovered))
        
        # Start with the target 5-set
        block = list(target)
        
        # Add points one at a time, choosing the one that covers the most new uncovered 5-sets
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered 5-sets would be covered by adding x
                gain = 0
                for sub in itertools.combinations(block, 4):
                    tsub = tuple(sorted(sub + (x,)))
                    if tsub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        # Remove all 5-subsets of this block from uncovered
        for sub in itertools.combinations(block, 5):
            uncovered.discard(sub)
    
    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t) with minimal blocks using randomized greedy."""
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Precompute all 5-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    
    best_blocks = None
    best_count = float('inf')
    
    # Try multiple random seeds for diversification
    for seed in range(200):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        blocks = _greedy_cover(uncovered, k=k, v=v, rng=rng)
        
        # Verify coverage
        covered = set()
        for b in blocks:
            for sub in itertools.combinations(b, t):
                covered.add(sub)
        
        if covered == set(all_tsubs):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
        # If we've found a good solution, we can stop early
        if best_count <= 340:
            break
    
    if best_blocks is not None:
        return best_blocks
    
    # Fallback: single deterministic greedy
    rng = random.Random(0)
    return _greedy_cover(uncovered, k=k, v=v, rng=rng)
# EVOLVE-BLOCK-END
