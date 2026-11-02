"""Covering design C(19,8,5): randomized greedy with redundancy removal."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import Counter


def _greedy_pass(v, k, t, rng, time_limit=120.0, start_time=None):
    """One greedy pass: repeatedly pick a t-subset to cover, build a block greedily."""
    if start_time is None:
        start_time = time.time()
    
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    all_points = set(range(v))
    
    while uncovered:
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Pick a random uncovered t-subset
        target = rng.choice(list(uncovered))
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
                    if new_tsub in uncovered:
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
                uncovered.discard(tsub)
    
    return blocks, uncovered


def _remove_redundant_blocks(v, k, t, blocks, time_limit=120.0, start_time=None):
    """Remove blocks that don't cover any t-subset uniquely."""
    if start_time is None:
        start_time = time.time()
    
    if not blocks:
        return blocks
    
    # Count how many blocks cover each t-subset
    tsub_counts = Counter()
    for block in blocks:
        for tsub in itertools.combinations(block, t):
            tsub_counts[tsub] += 1
    
    # Keep only blocks that cover at least one t-subset with count 1
    kept = []
    for block in blocks:
        if time.time() - start_time > time_limit - 5.0:
            kept.append(block)
            continue
        for tsub in itertools.combinations(block, t):
            if tsub_counts[tsub] == 1:
                kept.append(block)
                break
    
    return kept


def _local_search_polish(v, k, t, blocks, rng, time_limit=120.0, start_time=None):
    """Try to reduce block count by replacing blocks with better ones via local search."""
    if start_time is None:
        start_time = time.time()
    
    if not blocks:
        return blocks
    
    # Build coverage map: for each t-subset, which blocks cover it
    all_tsubs = set(itertools.combinations(range(v), t))
    tsub_to_blocks = {}
    for i, block in enumerate(blocks):
        for tsub in itertools.combinations(block, t):
            if tsub not in tsub_to_blocks:
                tsub_to_blocks[tsub] = set()
            tsub_to_blocks[tsub].add(i)
    
    # Find t-subsets covered by exactly one block (critical coverage)
    critical = {tsub: list(bs)[0] for tsub, bs in tsub_to_blocks.items() if len(bs) == 1}
    
    # Try to remove blocks that are not critical, then re-cover if needed
    # First pass: greedily try removing non-critical blocks
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
                if tsub in tsub_to_blocks:
                    if len(tsub_to_blocks[tsub]) == 1:
                        can_remove = False
                        break
            
            if can_remove:
                # Remove block i
                removed_block = blocks_list.pop(i)
                for tsub in itertools.combinations(removed_block, t):
                    if tsub in tsub_to_blocks:
                        tsub_to_blocks[tsub].discard(i)
                        # Shift indices for blocks after i
                        for j in range(i, len(blocks_list)):
                            pass  # We'll rebuild if needed
                changed = True
                break
    
    return blocks_list


def _try_replace_block(v, k, t, blocks, block_idx, rng, time_limit=120.0, start_time=None):
    """Try to replace block at block_idx with a different block that covers the same critical t-subsets."""
    if start_time is None:
        start_time = time.time()
    
    if block_idx < 0 or block_idx >= len(blocks):
        return blocks
    
    # Find t-subsets that are ONLY covered by this block
    all_tsubs = set(itertools.combinations(range(v), t))
    tsub_counts = Counter()
    for i, block in enumerate(blocks):
        if i == block_idx:
            continue
        for tsub in itertools.combinations(block, t):
            tsub_counts[tsub] += 1
    
    # T-subsets that must be covered by the new block
    must_cover = set()
    for tsub in itertools.combinations(blocks[block_idx], t):
        if tsub_counts.get(tsub, 0) == 0:
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
    remaining = k - len(new_block)
    
    # Count uncovered t-subsets (those not covered by other blocks)
    uncovered_tsubs = set()
    for tsub in all_tsubs:
        if tsub_counts.get(tsub, 0) == 0:
            uncovered_tsubs.add(tsub)
    
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
                if new_tsub in uncovered_tsubs:
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
    
    # Try multiple randomized restarts, keep the best result
    best_blocks = None
    best_len = float('inf')
    
    # Use different seeds for diversity
    num_restarts = 10
    
    for restart in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10.0:
            break
        
        rng = random.Random(42 + restart)
        
        # Greedy pass
        blocks, remaining = _greedy_pass(v, k, t, rng, time_limit, start_time)
        
        if not blocks:
            continue
        
        # If we didn't finish (remaining uncovered), try to complete or skip
        if remaining:
            # Try to finish the remaining with a simple greedy
            elapsed = time.time() - start_time
            if elapsed < time_limit - 8.0:
                for tsub in remaining:
                    if time.time() - start_time > time_limit - 5.0:
                        break
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
                                if new_tsub in remaining:
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
                            remaining.discard(ts)
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(v, k, t, blocks, time_limit, start_time)
        
        # Local search polish: try to replace blocks to reduce count
        elapsed = time.time() - start_time
        if elapsed < time_limit - 15.0 and len(blocks) > 1:
            # Try replacing some blocks
            for attempt in range(min(5, len(blocks))):
                if time.time() - start_time > time_limit - 10.0:
                    break
                idx = rng.randint(0, len(blocks) - 1)
                new_blocks = _try_replace_block(v, k, t, blocks, idx, rng, time_limit, start_time)
                if len(new_blocks) < len(blocks):
                    blocks = new_blocks
                    # Re-remove redundant blocks
                    blocks = _remove_redundant_blocks(v, k, t, blocks, time_limit, start_time)
        
        # Verify coverage
        all_tsubs = set(itertools.combinations(range(v), t))
        covered = set()
        for block in blocks:
            for tsub in itertools.combinations(block, t):
                covered.add(tsub)
        
        if covered >= all_tsubs and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
    
    if best_blocks is None:
        # Fallback: simple greedy
        rng = random.Random(0)
        best_blocks, _ = _greedy_pass(v, k, t, rng, time_limit, start_time)
        if best_blocks:
            best_blocks = list(best_blocks)
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
