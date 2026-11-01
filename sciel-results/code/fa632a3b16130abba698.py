"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

_global_deadline = None


def _build_coverage(blocks, v, k, t):
    """Build coverage count dict: t-subset tuple -> number of blocks covering it."""
    coverage_count = {}
    for block in blocks:
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            coverage_count[ts] = coverage_count.get(ts, 0) + 1
    return coverage_count


def _generate_block_weighted(seed_points, v, k, t, weights, rng):
    """Generate a k-block starting from seed_points, choosing additional points
    to maximize weighted coverage of t-subsets. weights maps t-subset tuple -> float."""
    block = list(seed_points)
    block_set = set(block)
    
    while len(block) < k:
        best_x = -1
        best_gain = -1.0
        for x in range(v):
            if x in block_set:
                continue
            # Compute weighted gain of adding x
            gain = 0.0
            for rest in itertools.combinations(block, t - 1):
                ts = tuple(sorted(rest + (x,)))
                w = weights.get(ts, 0.0)
                if w > 0:
                    gain += w
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x < 0:
            # No positive-gain point; pick a random one
            remaining = [x for x in range(v) if x not in block_set]
            if not remaining:
                break
            best_x = rng.choice(remaining)
        block.append(best_x)
        block_set.add(best_x)
    block.sort()
    return block


def _column_generation(v, k, t, deadline, rng):
    """Column generation for minimum set cover.
    Maintains a set of blocks and dual prices for uncovered t-subsets.
    Each iteration: generate a new block that maximizes reduced cost
    (weighted coverage of uncovered/low-coverage t-subsets)."""
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    
    # Initialize with a greedy solution
    weights = {ts: 1.0 for ts in all_ts}
    blocks = []
    covered = set()
    
    # Initial greedy phase to get a valid cover
    uncovered_set = set(all_ts)
    while uncovered_set and time.time() < deadline:
        seed = rng.choice(list(uncovered_set))
        block = _generate_block_weighted(seed, v, k, t, weights, rng)
        blocks.append(block)
        for ts in itertools.combinations(block, t):
            uncovered_set.discard(tuple(sorted(ts)))
    
    if not uncovered_set:
        return blocks
    
    # Column generation refinement
    coverage_count = _build_coverage(blocks, v, k, t)
    
    # Dual prices: higher for less-covered t-subsets
    # Start with price proportional to 1/count (or large if uncovered)
    prices = {}
    for ts in all_ts:
        cnt = coverage_count.get(ts, 0)
        if cnt == 0:
            prices[ts] = 100.0
        elif cnt == 1:
            prices[ts] = 10.0
        elif cnt == 2:
            prices[ts] = 3.0
        else:
            prices[ts] = 1.0 / cnt
    
    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    
    improved = True
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        iteration += 1
        
        # Find the most expensive uncovered/low-coverage t-subset as seed
        # Sample from high-price t-subsets
        high_price_ts = [(ts, p) for ts, p in prices.items() if p >= 3.0]
        if not high_price_ts:
            high_price_ts = [(ts, p) for ts, p in prices.items() if p >= 1.0]
        if not high_price_ts:
            break
        
        # Pick a random high-price t-subset as seed
        seed_ts = rng.choice(high_price_ts)[0]
        
        # Generate candidate block
        new_block = _generate_block_weighted(seed_ts, v, k, t, prices, rng)
        
        # Check if this block is already in our set
        if new_block in blocks:
            continue
        
        # Evaluate: does adding this block and removing some others reduce total count?
        # Try: add new block, then remove redundant blocks
        test_blocks = [list(b) for b in blocks] + [new_block]
        test_cov = _build_coverage(test_blocks, v, k, t)
        
        # Remove redundant blocks from test set
        removed_any = True
        while removed_any and time.time() < deadline:
            removed_any = False
            for bi in range(len(test_blocks)):
                if time.time() >= deadline:
                    break
                redundant = True
                for ts in itertools.combinations(test_blocks[bi], t):
                    ts = tuple(sorted(ts))
                    if test_cov.get(ts, 0) <= 1:
                        redundant = False
                        break
                if redundant:
                    removed_block = test_blocks.pop(bi)
                    for ts in itertools.combinations(removed_block, t):
                        ts = tuple(sorted(ts))
                        test_cov[ts] -= 1
                        if test_cov[ts] <= 0:
                            del test_cov[ts]
                    removed_any = True
                    break
        
        if len(test_blocks) < len(blocks):
            blocks = test_blocks
            coverage_count = test_cov
            best_blocks = [list(b) for b in blocks]
            best_count = len(blocks)
            improved = True
            
            # Update prices based on new coverage
            for ts in all_ts:
                cnt = coverage_count.get(ts, 0)
                if cnt == 0:
                    prices[ts] = 100.0
                elif cnt == 1:
                    prices[ts] = 10.0
                elif cnt == 2:
                    prices[ts] = 3.0
                else:
                    prices[ts] = 1.0 / cnt
        else:
            # No improvement; try a few more random seeds
            for _ in range(3):
                if time.time() >= deadline:
                    break
                seed_ts2 = rng.choice(high_price_ts)[0]
                new_block2 = _generate_block_weighted(seed_ts2, v, k, t, prices, rng)
                if new_block2 in blocks:
                    continue
                test_blocks2 = [list(b) for b in blocks] + [new_block2]
                test_cov2 = _build_coverage(test_blocks2, v, k, t)
                removed_any2 = True
                while removed_any2 and time.time() < deadline:
                    removed_any2 = False
                    for bi in range(len(test_blocks2)):
                        if time.time() >= deadline:
                            break
                        redundant2 = True
                        for ts in itertools.combinations(test_blocks2[bi], t):
                            ts = tuple(sorted(ts))
                            if test_cov2.get(ts, 0) <= 1:
                                redundant2 = False
                                break
                        if redundant2:
                            removed_block2 = test_blocks2.pop(bi)
                            for ts in itertools.combinations(removed_block2, t):
                                ts = tuple(sorted(ts))
                                test_cov2[ts] -= 1
                                if test_cov2[ts] <= 0:
                                    del test_cov2[ts]
                            removed_any2 = True
                            break
                if len(test_blocks2) < len(blocks):
                    blocks = test_blocks2
                    coverage_count = test_cov2
                    best_blocks = [list(b) for b in blocks]
                    best_count = len(blocks)
                    improved = True
                    for ts in all_ts:
                        cnt = coverage_count.get(ts, 0)
                        if cnt == 0:
                            prices[ts] = 100.0
                        elif cnt == 1:
                            prices[ts] = 10.0
                        elif cnt == 2:
                            prices[ts] = 3.0
                        else:
                            prices[ts] = 1.0 / cnt
                    break
    
    return best_blocks


def _local_search_refine(blocks, v, k, t, deadline, rng):
    """Local search: try to swap blocks to reduce count.
    For each block, try replacing it with a better one that covers
    more low-coverage t-subsets without breaking validity."""
    blocks = [list(b) for b in blocks]
    coverage_count = _build_coverage(blocks, v, k, t)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to remove redundant blocks
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            redundant = True
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                removed_block = blocks.pop(bi)
                for ts in itertools.combinations(removed_block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                improved = True
                break
        if improved:
            continue
        
        # Try block swaps: remove block i, add a new block that covers its unique t-subsets
        # Prioritize blocks with fewest unique t-subsets
        block_unique_counts = []
        for block in blocks:
            unique = 0
            for ts in itertools.combinations(block, t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) == 1:
                    unique += 1
            block_unique_counts.append(unique)
        
        candidates = sorted(range(len(blocks)), key=lambda i: block_unique_counts[i])
        tried = 0
        for bi in candidates:
            if time.time() >= deadline or tried >= 5:
                break
            tried += 1
            if block_unique_counts[bi] > 5:
                break
            
            # Find t-subsets that would become uncovered
            test_cov = dict(coverage_count)
            removed = blocks[bi]
            newly_uncovered = []
            for ts in itertools.combinations(removed, t):
                ts = tuple(sorted(ts))
                test_cov[ts] -= 1
                if test_cov[ts] <= 0:
                    del test_cov[ts]
                    newly_uncovered.append(ts)
            
            if not newly_uncovered:
                continue
            
            # Try to cover newly_uncovered by modifying an existing block
            success = True
            test_blocks = [list(b) for b in blocks if b is not removed]
            for ts in newly_uncovered:
                ts_set = set(ts)
                found = False
                for j, block in enumerate(test_blocks):
                    if time.time() >= deadline:
                        success = False
                        break
                    block_set = set(block)
                    overlap = len(ts_set & block_set)
                    if overlap >= t - 1:
                        missing = ts_set - block_set
                        if len(missing) == 1 and len(block) < k:
                            new_point = missing.pop()
                            test_blocks[j].append(new_point)
                            test_blocks[j].sort()
                            for new_ts in itertools.combinations(test_blocks[j], t):
                                new_ts = tuple(sorted(new_ts))
                                test_cov[new_ts] = test_cov.get(new_ts, 0) + 1
                            found = True
                            break
                if not found:
                    success = False
                    break
            
            if success:
                blocks = test_blocks
                coverage_count = test_cov
                improved = True
                break
    
    return blocks


def construct(v, k, t):
    global _global_deadline
    _global_deadline = time.time() + 140  # Safety margin before 150s limit
    rng = random.Random(42)
    
    # Phase 1: Column generation
    blocks = _column_generation(v, k, t, _global_deadline - 40, rng)
    
    # Phase 2: Local search refinement
    if time.time() < _global_deadline - 10:
        blocks = _local_search_refine(blocks, v, k, t, _global_deadline - 5, rng)
    
    return blocks
# EVOLVE-BLOCK-END
