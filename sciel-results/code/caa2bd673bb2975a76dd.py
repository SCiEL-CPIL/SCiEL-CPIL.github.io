"""Bitmask-based greedy set cover with local search optimization.
Uses Python big integers as bitmasks for fast coverage tracking.
"""
import time
import itertools
import random
from collections import defaultdict

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s evaluator limit


def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _precompute(v, k, t):
    """Precompute all t-subsets and their indices, plus block-to-tsubset mappings."""
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # For each possible k-block, compute which t-subsets it covers (as bitmask)
    # We'll compute this on-demand to save memory
    return all_tsubs, tsub_index, n_tsubs


def _block_to_tsubmask(block, t, tsub_index):
    """Convert a block to a bitmask of t-subsets it covers."""
    mask = 0
    for ts in itertools.combinations(block, t):
        idx = tsub_index.get(ts)
        if idx is not None:
            mask |= (1 << idx)
    return mask


def _greedy_cover(v, k, t, rng, time_limit):
    """Greedy set cover using bitmask operations."""
    all_tsubs, tsub_index, n_tsubs = _precompute(v, k, t)
    
    # Generate all possible k-subsets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    
    # Precompute tsubmask for each k-subset
    ksub_masks = []
    for ksub in all_ksubs:
        mask = 0
        for ts in itertools.combinations(ksub, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                mask |= (1 << idx)
        ksub_masks.append(mask)
    
    # Full mask: all t-subsets
    full_mask = (1 << n_tsubs) - 1
    
    # Uncovered mask
    uncovered = full_mask
    
    selected = []
    start = time.time()
    
    while uncovered and (time.time() - start) < time_limit:
        best_idx = -1
        best_gain = 0
        
        # Find block with maximum gain (most newly covered t-subsets)
        for i in range(n_ksubs):
            gain = bin(ksub_masks[i] & uncovered).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            break
        
        selected.append(best_idx)
        uncovered &= ~ksub_masks[best_idx]
    
    # Convert indices to blocks
    blocks = [list(all_ksubs[i]) for i in selected]
    return blocks


def _is_valid(blocks, v, t, tsub_index, n_tsubs):
    """Check if blocks form a valid cover."""
    covered = 0
    for block in blocks:
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                covered |= (1 << idx)
    return covered == (1 << n_tsubs) - 1


def _local_search(blocks, v, k, t, tsub_index, n_tsubs, rng, time_limit):
    """Local search to reduce number of blocks."""
    blocks = [list(b) for b in blocks]
    full_mask = (1 << n_tsubs) - 1
    
    # Precompute masks for current blocks
    block_masks = [_block_to_tsubmask(b, t, tsub_index) for b in blocks]
    
    # Coverage count: how many blocks cover each t-subset
    # We'll track this implicitly
    
    start = time.time()
    improved = True
    
    while improved and (time.time() - start) < time_limit:
        improved = False
        
        # Try to remove redundant blocks
        for i in range(len(blocks)):
            if (time.time() - start) > time_limit:
                break
            
            # Check if block i is redundant: all its t-subsets are covered by other blocks
            redundant = True
            for ts in itertools.combinations(blocks[i], t):
                idx = tsub_index.get(ts)
                if idx is None:
                    continue
                bit = 1 << idx
                # Check if any other block covers this t-subset
                covered_by_other = False
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    if block_masks[j] & bit:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            
            if redundant:
                blocks.pop(i)
                block_masks.pop(i)
                improved = True
                break
        
        if improved:
            continue
        
        # Try to merge or replace blocks
        # For each block, try replacing one point to see if we can still maintain coverage
        for i in range(len(blocks)):
            if (time.time() - start) > time_limit:
                break
            
            block = blocks[i]
            original_mask = block_masks[i]
            
            # Try each position
            for pi in range(k):
                old_point = block[pi]
                for new_point in range(v):
                    if new_point == old_point or new_point in block:
                        continue
                    
                    new_block = block[:]
                    new_block[pi] = new_point
                    new_block.sort()
                    new_mask = _block_to_tsubmask(new_block, t, tsub_index)
                    
                    # Check if new_mask covers all t-subsets that original_mask uniquely covers
                    # i.e., for each t-subset in original_mask, is it covered by new_mask OR by some other block?
                    ok = True
                    for ts in itertools.combinations(block, t):
                        idx = tsub_index.get(ts)
                        if idx is None:
                            continue
                        bit = 1 << idx
                        if not (new_mask & bit):
                            # Not covered by new block, check if covered by another
                            covered_by_other = False
                            for j in range(len(blocks)):
                                if j == i:
                                    continue
                                if block_masks[j] & bit:
                                    covered_by_other = True
                                    break
                            if not covered_by_other:
                                ok = False
                                break
                    
                    if ok:
                        blocks[i] = new_block
                        block_masks[i] = new_mask
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return blocks


def construct(v, k, t):
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(42)
    
    all_tsubs, tsub_index, n_tsubs = _precompute(v, k, t)
    
    # Phase 1: Greedy construction
    best_blocks = _greedy_cover(v, k, t, rng, min(60, _time_left() * 0.5))
    
    # Ensure validity
    if not _is_valid(best_blocks, v, t, tsub_index, n_tsubs):
        # Emergency fill
        covered = 0
        for block in best_blocks:
            for ts in itertools.combinations(block, t):
                idx = tsub_index.get(ts)
                if idx is not None:
                    covered |= (1 << idx)
        
        full_mask = (1 << n_tsubs) - 1
        uncovered = full_mask & ~covered
        
        while uncovered and _time_left() > 5:
            # Find an uncovered t-subset
            idx = uncovered.bit_length() - 1
            if idx >= n_tsubs:
                break
            ts = all_tsubs[idx]
            
            # Create a block containing this t-subset
            block = list(ts)
            remaining = [p for p in range(v) if p not in block]
            rng.shuffle(remaining)
            block.extend(remaining[:k - t])
            block.sort()
            best_blocks.append(block)
            
            # Update covered
            for ts2 in itertools.combinations(block, t):
                idx2 = tsub_index.get(ts2)
                if idx2 is not None:
                    covered |= (1 << idx2)
            uncovered = full_mask & ~covered
    
    # Phase 2: Local search
    if _time_left() > 10:
        best_blocks = _local_search(best_blocks, v, k, t, tsub_index, n_tsubs, rng, min(60, _time_left() * 0.5))
    
    # Final validation
    if not _is_valid(best_blocks, v, t, tsub_index, n_tsubs):
        # Emergency fill again
        covered = 0
        for block in best_blocks:
            for ts in itertools.combinations(block, t):
                idx = tsub_index.get(ts)
                if idx is not None:
                    covered |= (1 << idx)
        
        full_mask = (1 << n_tsubs) - 1
        uncovered = full_mask & ~covered
        
        while uncovered and _time_left() > 2:
            idx = uncovered.bit_length() - 1
            if idx >= n_tsubs:
                break
            ts = all_tsubs[idx]
            
            block = list(ts)
            remaining = [p for p in range(v) if p not in block]
            rng.shuffle(remaining)
            block.extend(remaining[:k - t])
            block.sort()
            best_blocks.append(block)
            
            for ts2 in itertools.combinations(block, t):
                idx2 = tsub_index.get(ts2)
                if idx2 is not None:
                    covered |= (1 << idx2)
            uncovered = full_mask & ~covered
    
    # Remove duplicates
    unique_blocks = []
    seen = set()
    for block in best_blocks:
        key = tuple(sorted(block))
        if key not in seen:
            seen.add(key)
            unique_blocks.append(list(block))
    
    return unique_blocks
