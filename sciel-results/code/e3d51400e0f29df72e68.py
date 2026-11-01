"""
Covering design C(24,7,3) via randomized greedy with efficient gain computation
and local search (deletion + swap) refinement.
"""
import itertools
import time
import random
from collections import defaultdict

DEADLINE = 145.0


def _all_tsubsets(v, t):
    """Return list of all t-subsets of {0,...,v-1} as tuples (sorted)."""
    return [tuple(c) for c in itertools.combinations(range(v), t)]


def _block_tsubsets(block, t):
    """Return list of t-subsets (as tuples) contained in a block."""
    return [tuple(c) for c in itertools.combinations(block, t)]


def _build_greedy(v, k, t, uncovered_set, rng):
    """
    Build a single block greedily: start from a random uncovered t-subset,
    then repeatedly add the point that covers the most uncovered t-subsets.
    Returns the block as a sorted list.
    """
    # Pick a random uncovered t-subset to seed the block
    seed_ts = rng.choice(tuple(uncovered_set))
    block = list(seed_ts)
    
    while len(block) < k:
        best_x = None
        best_gain = -1
        for x in range(v):
            if x in block:
                continue
            gain = 0
            # For t=3: count uncovered pairs in block that with x form an uncovered triple
            for rest in itertools.combinations(block, t - 1):
                sub = tuple(sorted(rest + (x,)))
                if sub in uncovered_set:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x is None:
            break
        block.append(best_x)
    
    # Fill remaining with arbitrary points
    if len(block) < k:
        for x in range(v):
            if len(block) >= k:
                break
            if x not in block:
                block.append(x)
    
    block.sort()
    return block


def _local_search_delete(blocks, v, k, t, uncovered_set, rng):
    """
    Try to remove blocks that don't cover any unique t-subset.
    Returns new list of blocks and updated uncovered_set.
    """
    # Count how many blocks cover each t-subset
    ts_count = defaultdict(int)
    for b in blocks:
        for ts in _block_tsubsets(b, t):
            ts_count[ts] += 1
    
    new_blocks = []
    for b in blocks:
        # Check if any t-subset in this block is covered only by this block
        unique = False
        for ts in _block_tsubsets(b, t):
            if ts_count[ts] == 1:
                unique = True
                break
        if not unique:
            # Safe to remove
            for ts in _block_tsubsets(b, t):
                ts_count[ts] -= 1
        else:
            new_blocks.append(b)
    
    return new_blocks


def _local_search_swap(blocks, v, k, t, uncovered_set, rng):
    """
    Try to replace a block with a better one (covers more uncovered t-subsets).
    This is a simplified swap: for a random block, try to find a replacement
    that covers at least as many currently-covered t-subsets while also
    covering some new ones.
    """
    if len(blocks) < 2:
        return blocks
    
    # Compute coverage counts
    ts_count = defaultdict(int)
    for b in blocks:
        for ts in _block_tsubsets(b, t):
            ts_count[ts] += 1
    
    # Try to improve by replacing one block
    idx = rng.randrange(len(blocks))
    old_block = blocks[idx]
    old_ts = _block_tsubsets(old_block, t)
    
    # Build a candidate block that covers the same critical t-subsets
    # Find t-subsets in old_block that are covered only by old_block
    critical = [ts for ts in old_ts if ts_count[ts] == 1]
    
    if not critical:
        # No critical t-subsets, try to replace with a block that covers more
        # Build a new block greedily aiming to cover uncovered t-subsets
        new_block = _build_greedy(v, k, t, uncovered_set, rng)
        if tuple(sorted(new_block)) != tuple(old_block):
            # Check validity: all critical t-subsets must be in new_block
            # (they're empty, so always valid)
            blocks[idx] = new_block
            return blocks
    
    # If there are critical t-subsets, we must keep them covered
    # Try to build a block containing all critical t-subsets
    # This is hard if there are many; skip for now
    return blocks


def construct(v, k, t):
    start_time = time.time()
    rng = random.Random(42)
    
    def time_check():
        return time.time() - start_time > DEADLINE - 5.0
    
    # All t-subsets
    all_ts = _all_tsubsets(v, t)
    uncovered = set(all_ts)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized greedy passes with local search
    num_restarts = 50  # will be limited by time
    
    for restart in range(num_restarts):
        if time_check():
            break
        
        # Reset for this restart
        uncovered = set(all_ts)
        blocks = []
        
        # Greedy phase
        while uncovered and not time_check():
            block = _build_greedy(v, k, t, uncovered, rng)
            blocks.append(block)
            # Remove covered t-subsets
            for ts in _block_tsubsets(block, t):
                uncovered.discard(ts)
        
        # Local search: deletion
        if not time_check():
            blocks = _local_search_delete(blocks, v, k, t, uncovered, rng)
        
        # Local search: swap (a few iterations)
        if not time_check():
            for _ in range(10):
                if time_check():
                    break
                new_blocks = _local_search_swap(blocks, v, k, t, uncovered, rng)
                if len(new_blocks) < len(blocks):
                    blocks = new_blocks
                    # Recompute uncovered
                    covered = set()
                    for b in blocks:
                        for ts in _block_tsubsets(b, t):
                            covered.add(ts)
                    uncovered = set(all_ts) - covered
        
        # Track best
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
        
        # If we've achieved a good count, we can stop early
        if best_count <= 76:
            break
    
    # Final safety: if best_blocks is None or doesn't cover everything, do a simple greedy
    if best_blocks is None:
        uncovered = set(all_ts)
        best_blocks = []
        while uncovered and not time_check():
            block = _build_greedy(v, k, t, uncovered, rng)
            best_blocks.append(block)
            for ts in _block_tsubsets(block, t):
                uncovered.discard(ts)
    
    # Verify coverage; if incomplete, add blocks for remaining uncovered t-subsets
    covered = set()
    for b in best_blocks:
        for ts in _block_tsubsets(b, t):
            covered.add(ts)
    uncovered = set(all_ts) - covered
    
    if uncovered:
        # Add blocks to cover remaining
        for ts in list(uncovered):
            block = list(ts)
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block:
                    block.append(x)
            block.sort()
            if block not in best_blocks:
                best_blocks.append(block)
            for ts2 in _block_tsubsets(block, t):
                uncovered.discard(ts2)
    
    return best_blocks
