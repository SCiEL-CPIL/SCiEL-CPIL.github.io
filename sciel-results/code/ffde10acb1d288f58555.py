"""Stronger covering design C(19,7,4) solver.

Strategy:
1. Greedy construction with a fast gain computation (count of uncovered t-subsets
   that would be covered by adding point x to the current partial block).
2. Local search / block replacement: try to replace a block with a better one,
   or remove redundant blocks.
3. Deadline guard to return best result before timeout.
"""

import itertools
import time
import random
from collections import defaultdict

def _count_covered(block, t):
    """Return the set of t-subsets covered by a block."""
    return set(itertools.combinations(block, t))

def _all_t_subsets(v, t):
    """Generate all t-subsets of {0,...,v-1}."""
    return set(itertools.combinations(range(v), t))

def _greedy_construct(v, k, t, deadline):
    """Greedy covering algorithm with fast gain computation."""
    # Precompute all t-subsets
    all_subs = _all_t_subsets(v, t)
    uncovered = set(all_subs)
    blocks = []
    
    # For speed, we track uncovered as a set of tuples
    # Gain for adding point x to current partial block B:
    # number of uncovered t-subsets that contain x and (t-1) points from B
    
    # Precompute for each (t-1)-subset, the list of t-subsets extending it
    # This is expensive for large v, but v=19 is small
    
    # Actually, let's use a different approach:
    # For a partial block B of size s, adding x covers t-subsets that contain x
    # and (t-1) points from B. So we need to count how many of those are in uncovered.
    
    # Precompute: for each point x and each (t-1)-subset S, the t-subset S∪{x}
    # We'll just compute on the fly since v is small.
    
    while uncovered and time.time() < deadline:
        # Start with the lexicographically smallest uncovered t-subset
        min_sub = min(uncovered)
        block = list(min_sub)
        
        # Greedily add points
        while len(block) < k and time.time() < deadline:
            best_x = -1
            best_gain = -1
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                # Compute gain: number of uncovered t-subsets containing x and (t-1) from block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    if sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
    
    return blocks, uncovered

def _remove_redundant_blocks(blocks, v, k, t, deadline):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks
    
    # Count how many blocks cover each t-subset
    sub_count = defaultdict(int)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            sub_count[sub] += 1
    
    # A block is redundant if every t-subset it covers is also covered by another block
    # i.e., for every sub in block, sub_count[sub] > 1
    kept = []
    for block in blocks:
        redundant = True
        for sub in itertools.combinations(block, t):
            if sub_count[sub] == 1:
                redundant = False
                break
        if not redundant:
            kept.append(block)
    
    return kept

def _local_search(blocks, v, k, t, deadline):
    """Try to improve the covering by replacing or removing blocks."""
    all_subs = _all_t_subsets(v, t)
    
    # Count coverage
    sub_count = defaultdict(int)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            sub_count[sub] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try removing each block and see if coverage is still complete
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            # Check if block i is needed
            block = blocks[i]
            needed = False
            for sub in itertools.combinations(block, t):
                if sub_count[sub] == 1:
                    needed = True
                    break
            if not needed:
                # Remove block i
                for sub in itertools.combinations(block, t):
                    sub_count[sub] -= 1
                blocks.pop(i)
                improved = True
                break
        
        # Try replacing a block with a better one
        # (This is complex; skip for now)
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for safety
    
    # Phase 1: Greedy construction
    blocks, uncovered = _greedy_construct(v, k, t, deadline)
    
    if time.time() > deadline:
        return blocks
    
    # Phase 2: Remove redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, k, t, deadline)
    
    if time.time() > deadline:
        return blocks
    
    # Phase 3: Local search - try to find better blocks
    # We'll try a simple hill-climbing: for each block, try to find a replacement
    # that covers more uncovered subsets (if any remain) or allows removal of others
    
    # Actually, let's try a different local search:
    # Randomly perturb blocks and see if we can reduce the count
    
    random.seed(42)
    best_blocks = blocks[:]
    best_len = len(blocks)
    
    while time.time() < deadline:
        # Try to remove a random block and see if coverage is still complete
        if len(best_blocks) > 0:
            idx = random.randint(0, len(best_blocks) - 1)
            test_blocks = best_blocks[:]
            test_blocks.pop(idx)
            
            # Check if all t-subsets are still covered
            covered = set()
            for block in test_blocks:
                covered.update(itertools.combinations(block, t))
            
            if len(covered) == len(_all_t_subsets(v, t)):
                best_blocks = test_blocks
                best_len = len(best_blocks)
                continue
        
        # Try to replace a block with a random one
        if len(best_blocks) > 0 and time.time() < deadline:
            idx = random.randint(0, len(best_blocks) - 1)
            old_block = best_blocks[idx]
            
            # Generate a random k-subset
            new_block = sorted(random.sample(range(v), k))
            
            # Check if replacing improves things (covers same or more)
            old_covered = set(itertools.combinations(old_block, t))
            new_covered = set(itertools.combinations(new_block, t))
            
            # Count how many subsets in old_covered are NOT in new_covered and not covered by other blocks
            other_covered = set()
            for i, block in enumerate(best_blocks):
                if i != idx:
                    other_covered.update(itertools.combinations(block, t))
            
            lost = old_covered - new_covered - other_covered
            gained = new_covered - old_covered - other_covered
            
            if len(lost) <= len(gained) and len(lost) == 0:
                best_blocks[idx] = new_block
        
        # Occasionally try greedy repair if uncovered subsets exist
        if time.time() > deadline - 20:
            break
    
    return best_blocks
