"""Covering design C(19,8,5): randomized greedy with ILP-style block removal."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import Counter
import numpy as np


def _precompute_tsub_indices(v, t):
    """Precompute index mapping for all t-subsets."""
    tsub_to_idx = {}
    idx_to_tsub = []
    for i, tsub in enumerate(itertools.combinations(range(v), t)):
        tsub_to_idx[tsub] = i
        idx_to_tsub.append(tsub)
    return tsub_to_idx, idx_to_tsub


def _greedy_pass(v, k, t, rng, time_limit=120.0, start_time=None, tsub_to_idx=None, idx_to_tsub=None):
    """One greedy pass: repeatedly pick a t-subset to cover, build a block greedily."""
    if start_time is None:
        start_time = time.time()
    
    if tsub_to_idx is None or idx_to_tsub is None:
        tsub_to_idx, idx_to_tsub = _precompute_tsub_indices(v, t)
    
    num_tsubs = len(idx_to_tsub)
    uncovered = np.ones(num_tsubs, dtype=bool)
    blocks = []
    all_points = set(range(v))
    
    while uncovered.any():
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Pick a random uncovered t-subset
        uncov_indices = np.where(uncovered)[0]
        target_idx = uncov_indices[rng.randint(0, len(uncov_indices) - 1)]
        target = idx_to_tsub[target_idx]
        block = list(target)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Find the point that covers the most uncovered t-subsets
            best_x = None
            best_gain = -1
            candidates = all_points - block_set
            for x in candidates:
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_tsub = tuple(sorted(rest + (x,)))
                    if new_tsub in tsub_to_idx:
                        idx = tsub_to_idx[new_tsub]
                        if uncovered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain <= 0:
                # No point adds coverage; just pick a random one
                available = list(all_points - block_set)
                if available:
                    best_x = rng.choice(available)
                else:
                    break
            
            block.append(best_x)
            block_set.add(best_x)
        
        if len(block) == k:
            block.sort()
            blocks.append(tuple(block))
            # Remove all t-subsets of this block from uncovered
            for tsub in itertools.combinations(block, t):
                if tsub in tsub_to_idx:
                    uncovered[tsub_to_idx[tsub]] = False
    
    return blocks, uncovered


def _remove_redundant_blocks(v, k, t, blocks, time_limit=120.0, start_time=None, tsub_to_idx=None):
    """Remove blocks that don't cover any t-subset uniquely."""
    if start_time is None:
        start_time = time.time()
    
    if not blocks:
        return blocks
    
    if tsub_to_idx is None:
        tsub_to_idx, _ = _precompute_tsub_indices(v, t)
    
    # Count how many blocks cover each t-subset
    num_tsubs = len(tsub_to_idx)
    tsub_counts = np.zeros(num_tsubs, dtype=np.int32)
    for block in blocks:
        for tsub in itertools.combinations(block, t):
            if tsub in tsub_to_idx:
                tsub_counts[tsub_to_idx[tsub]] += 1
    
    # Keep only blocks that cover at least one t-subset with count 1
    kept = []
    for block in blocks:
        if time.time() - start_time > time_limit - 5.0:
            kept.append(block)
            continue
        for tsub in itertools.combinations(block, t):
            if tsub in tsub_to_idx and tsub_counts[tsub_to_idx[tsub]] == 1:
                kept.append(block)
                break
    
    return kept


def _ilp_block_removal(v, k, t, blocks, time_limit=120.0, start_time=None, tsub_to_idx=None, idx_to_tsub=None):
    """ILP-style: greedily remove blocks that don't leave any t-subset uncovered.
    Maximizes number of excluded blocks while maintaining full coverage."""
    if start_time is None:
        start_time = time.time()
    
    if not blocks:
        return blocks
    
    if tsub_to_idx is None or idx_to_tsub is None:
        tsub_to_idx, idx_to_tsub = _precompute_tsub_indices(v, t)
    
    num_tsubs = len(idx_to_tsub)
    n_blocks = len(blocks)
    
    # Build coverage matrix: cov[tsub_idx] = list of block indices covering it
    cov = [[] for _ in range(num_tsubs)]
    for i, block in enumerate(blocks):
        for tsub in itertools.combinations(block, t):
            if tsub in tsub_to_idx:
                cov[tsub_to_idx[tsub]].append(i)
    
    # Coverage count per t-subset
    cov_count = np.array([len(c) for c in cov], dtype=np.int32)
    
    # Greedy removal: repeatedly find a block whose removal doesn't break coverage
    # Priority: remove blocks that cover the fewest critical (count==1) t-subsets
    removed = np.zeros(n_blocks, dtype=bool)
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # For each non-removed block, count how many t-subsets it uniquely covers
        best_block = -1
        best_unique = float('inf')
        for i in range(n_blocks):
            if removed[i]:
                continue
            unique_count = 0
            for tsub in itertools.combinations(blocks[i], t):
                if tsub in tsub_to_idx:
                    idx = tsub_to_idx[tsub]
                    if cov_count[idx] == 1:
                        unique_count += 1
            if unique_count < best_unique:
                best_unique = unique_count
                best_block = i
            if unique_count == 0:
                break
        
        if best_block == -1 or best_unique > 0:
            break
        
        # Remove best_block
        removed[best_block] = True
        for tsub in itertools.combinations(blocks[best_block], t):
            if tsub in tsub_to_idx:
                idx = tsub_to_idx[tsub]
                cov_count[idx] -= 1
                cov[idx].remove(best_block)
        changed = True
    
    # Return non-removed blocks
    return [blocks[i] for i in range(n_blocks) if not removed[i]]


def _local_search_polish(v, k, t, blocks, rng, time_limit=120.0, start_time=None, tsub_to_idx=None):
    """Try to reduce block count by replacing blocks with better ones via local search."""
    if start_time is None:
        start_time = time.time()
    
    if not blocks:
        return blocks
    
    if tsub_to_idx is None:
        tsub_to_idx, _ = _precompute_tsub_indices(v, t)
    
    # Build coverage count per t-subset
    num_tsubs = len(tsub_to_idx)
    tsub_cov_count = np.zeros(num_tsubs, dtype=np.int32)
    for block in blocks:
        for tsub in itertools.combinations(block, t):
            if tsub in tsub_to_idx:
                tsub_cov_count[tsub_to_idx[tsub]] += 1
    
    blocks_list = list(blocks)
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > time_limit - 5.0:
            break
        
        for i in range(len(blocks_list) - 1, -1, -1):
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Check if removing block i leaves any t-subset uncovered
            can_remove = True
            for tsub in itertools.combinations(blocks_list[i], t):
                if tsub in tsub_to_idx:
                    if tsub_cov_count[tsub_to_idx[tsub]] == 1:
                        can_remove = False
                        break
            
            if can_remove:
                # Remove block i
                removed_block = blocks_list.pop(i)
                for tsub in itertools.combinations(removed_block, t):
                    if tsub in tsub_to_idx:
                        tsub_cov_count[tsub_to_idx[tsub]] -= 1
                changed = True
                break
    
    return blocks_list


def _try_replace_block(v, k, t, blocks, block_idx, rng, time_limit=120.0, start_time=None, tsub_to_idx=None):
    """Try to replace block at block_idx with a different block that covers the same critical t-subsets."""
    if start_time is None:
        start_time = time.time()
    
    if block_idx < 0 or block_idx >= len(blocks):
        return blocks
    
    if tsub_to_idx is None:
        tsub_to_idx, _ = _precompute_tsub_indices(v, t)
    
    # Find t-subsets that are ONLY covered by this block
    num_tsubs = len(tsub_to_idx)
    tsub_counts = np.zeros(num_tsubs, dtype=np.int32)
    for i, block in enumerate(blocks):
        if i == block_idx:
            continue
        for tsub in itertools.combinations(block, t):
            if tsub in tsub_to_idx:
                tsub_counts[tsub_to_idx[tsub]] += 1
    
    # T-subsets that must be covered by the new block
    must_cover = set()
    for tsub in itertools.combinations(blocks[block_idx], t):
        if tsub in tsub_to_idx and tsub_counts[tsub_to_idx[tsub]] == 0:
            must_cover.add(tsub)
    
    if not must_cover:
        # This block is fully redundant; just remove it
        return blocks[:block_idx] + blocks[block_idx+1:]
    
    # Try to find a new block that covers all must_cover t-subsets
    # The new block must contain the union of all must_cover t-subsets
    required_points = set()
    for tsub in must_cover:
        required_points.update(tsub)
    
    if len(required_points) > k:
        return blocks  # Can't fit all required points in one block
    
    # Fill remaining slots greedily to cover more t-subsets
    new_block = list(required_points)
    new_block_set = set(new_block)
    
    # Count uncovered t-subsets (those not covered by other blocks)
    uncovered_mask = (tsub_counts == 0)
    
    while len(new_block) < k:
        if time.time() - start_time > time_limit - 5.0:
            break
        
        best_x = None
        best_gain = -1
        for x in range(v):
            if x in new_block_set:
                continue
            gain = 0
            for rest in itertools.combinations(new_block, t - 1):
                new_tsub = tuple(sorted(rest + (x,)))
                if new_tsub in tsub_to_idx:
                    idx = tsub_to_idx[new_tsub]
                    if uncovered_mask[idx]:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None:
            available = [x for x in range(v) if x not in new_block_set]
            if available:
                best_x = rng.choice(available)
            else:
                break
        
        new_block.append(best_x)
        new_block_set.add(best_x)
    
    if len(new_block) != k:
        return blocks
    
    new_block.sort()
    new_block = tuple(new_block)
    
    # Verify the new block covers all must_cover t-subsets
    new_block_tsubs = set(itertools.combinations(new_block, t))
    for tsub in must_cover:
        if tsub not in new_block_tsubs:
            return blocks  # New block doesn't cover required t-subsets
    
    # Replace the block
    return blocks[:block_idx] + [new_block] + blocks[block_idx+1:]


def construct(v, k, t):
    start_time = time.time()
    time_limit = 120.0
    
    # Precompute t-subset indexing
    tsub_to_idx, idx_to_tsub = _precompute_tsub_indices(v, t)
    
    # Try multiple randomized restarts, keep the best result
    best_blocks = None
    best_len = float('inf')
    
    # Use different seeds for diversity
    num_restarts = 8
    
    for restart in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 15.0:
            break
        
        rng = random.Random(42 + restart)
        
        # Greedy pass
        blocks, remaining = _greedy_pass(v, k, t, rng, time_limit, start_time, tsub_to_idx, idx_to_tsub)
        
        if not blocks:
            continue
        
        # If we didn't finish (remaining uncovered), try to complete or skip
        if remaining.any():
            # Try to finish the remaining with a simple greedy
            elapsed = time.time() - start_time
            if elapsed < time_limit - 10.0:
                uncov_indices = np.where(remaining)[0]
                for tsub_idx in uncov_indices:
                    if time.time() - start_time > time_limit - 5.0:
                        break
                    tsub = idx_to_tsub[tsub_idx]
                    block = list(tsub)
                    block_set = set(block)
                    while len(block) < k:
                        if time.time() - start_time > time_limit - 5.0:
                            break
                        best_x = None
                        best_gain = -1
                        for x in range(v):
                            if x in block_set:
                                continue
                            gain = 0
                            for rest in itertools.combinations(block, t - 1):
                                new_tsub = tuple(sorted(rest + (x,)))
                                if new_tsub in tsub_to_idx:
                                    idx = tsub_to_idx[new_tsub]
                                    if remaining[idx]:
                                        gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            available = [x for x in range(v) if x not in block_set]
                            if available:
                                best_x = rng.choice(available)
                            else:
                                break
                        block.append(best_x)
                        block_set.add(best_x)
                    if len(block) == k:
                        block.sort()
                        blocks.append(tuple(block))
                        for ts in itertools.combinations(block, t):
                            if ts in tsub_to_idx:
                                remaining[tsub_to_idx[ts]] = False
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(v, k, t, blocks, time_limit, start_time, tsub_to_idx)
        
        # ILP-style block removal: maximize excluded blocks
        elapsed = time.time() - start_time
        if elapsed < time_limit - 20.0 and len(blocks) > 1:
            blocks = _ilp_block_removal(v, k, t, blocks, time_limit, start_time, tsub_to_idx, idx_to_tsub)
        
        # Local search polish: try to replace blocks to reduce count
        elapsed = time.time() - start_time
        if elapsed < time_limit - 20.0 and len(blocks) > 1:
            # Try replacing some blocks
            for attempt in range(min(8, len(blocks))):
                if time.time() - start_time > time_limit - 12.0:
                    break
                idx = rng.randint(0, len(blocks) - 1)
                new_blocks = _try_replace_block(v, k, t, blocks, idx, rng, time_limit, start_time, tsub_to_idx)
                if len(new_blocks) < len(blocks):
                    blocks = new_blocks
                    # Re-remove redundant blocks
                    blocks = _remove_redundant_blocks(v, k, t, blocks, time_limit, start_time, tsub_to_idx)
                    # Try ILP removal again
                    if time.time() - start_time < time_limit - 15.0:
                        blocks = _ilp_block_removal(v, k, t, blocks, time_limit, start_time, tsub_to_idx, idx_to_tsub)
        
        # Verify coverage
        covered_mask = np.zeros(len(idx_to_tsub), dtype=bool)
        for block in blocks:
            for tsub in itertools.combinations(block, t):
                if tsub in tsub_to_idx:
                    covered_mask[tsub_to_idx[tsub]] = True
        
        if covered_mask.all() and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
    
    if best_blocks is None:
        # Fallback: simple greedy
        rng = random.Random(0)
        best_blocks, _ = _greedy_pass(v, k, t, rng, time_limit, start_time, tsub_to_idx, idx_to_tsub)
        if best_blocks:
            best_blocks = list(best_blocks)
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
