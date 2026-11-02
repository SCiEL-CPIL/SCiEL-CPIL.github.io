"""
Combined approach: multi-start randomized greedy with bitmask-accelerated gain
calculation, redundancy removal, and a local-search polish pass.
"""
import itertools
import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(v, k, t):
    """
    Multi-start randomized greedy covering with bitmask acceleration,
    redundancy removal, and a local-search polish pass.
    
    Strategy:
    1. Precompute all t-subsets and index them.
    2. For each restart, run a greedy construction that picks blocks to
       maximize marginal gain (number of newly covered t-subsets).
    3. After construction, remove redundant blocks.
    4. Apply a local-search polish: try swapping points in blocks to reduce
       total block count or improve coverage.
    5. Keep the best (smallest) valid covering found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin below 150s kill

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    
    # Fast lookup: for any (a, b, c) with a < b < c, get index
    # For general t, we use the triple_to_idx dict
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        seed_val = restart_count * 12345 + 1
        rng = random.Random(seed_val)
        
        blocks = _greedy_construct(v, k, t, all_t_subsets, triple_to_idx, deadline, rng)
        
        if blocks is None:
            # Timeout during construction
            continue
        
        # Try to remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, v, k, t, deadline)
        
        # Try local search polish
        if time.time() < deadline - 10.0:
            blocks = _local_search_polish(blocks, v, k, t, all_t_subsets, triple_to_idx, deadline, rng)
        
        # Validate
        if _is_valid_covering(blocks, v, k, t, all_t_subsets):
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = [list(b) for b in blocks]
                
                # If we reached a very good score, we can stop early
                if best_len <= 76:
                    break

    if best_blocks is None:
        # Fallback: run a deterministic greedy pass
        if time.time() < deadline - 5.0:
            rng = random.Random(0)
            blocks = _greedy_construct(v, k, t, all_t_subsets, triple_to_idx, deadline, rng)
            if blocks and _is_valid_covering(blocks, v, k, t, all_t_subsets):
                best_blocks = blocks
                best_len = len(blocks)
        
        if best_blocks is None:
            return []

    return best_blocks


def _greedy_construct(v, k, t, all_t_subsets, triple_to_idx, deadline, rng):
    """
    Greedy construction of a covering using bitmask for fast gain calculation.
    Returns a list of blocks (tuples) or None if timeout.
    """
    uncovered = set(all_t_subsets)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            return None
        
        # Pick a random uncovered t-subset to be the seed of the new block
        seed_triple = rng.choice(list(uncovered))
        current_block = set(seed_triple)
        
        # Fill the block to size k
        while len(current_block) < k:
            if time.time() > deadline:
                return None
            
            best_point = None
            best_gain = -1
            best_gain_ties = []
            
            # Precompute (t-1) subsets of current_block for gain calculation
            t_minus_1_subsets = list(itertools.combinations(sorted(current_block), t - 1))
            
            for x in range(v):
                if x in current_block:
                    continue
                
                gain = 0
                for rest in t_minus_1_subsets:
                    # Construct the t-subset containing x and (t-1) elements from current_block
                    t_subset = tuple(sorted(rest + (x,)))
                    if t_subset in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    best_gain_ties = [x]
                elif gain == best_gain:
                    best_gain_ties.append(x)
            
            if best_gain <= 0:
                # No point improves coverage. Pick a random point to fill the block.
                candidates = [x for x in range(v) if x not in current_block]
                if not candidates:
                    break
                best_point = rng.choice(candidates)
            else:
                # Pick a random point from the best gain ties for diversity
                best_point = rng.choice(best_gain_ties)
            
            current_block.add(best_point)
        
        # Convert to sorted tuple
        block_tuple = tuple(sorted(current_block))
        blocks.append(block_tuple)
        
        # Update uncovered set
        for sub in itertools.combinations(block_tuple, t):
            uncovered.discard(sub)
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t, deadline):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks

    n = len(blocks)
    if n > 300:
        return blocks

    # Map each t-subset to the list of block indices that cover it
    subset_to_blocks = defaultdict(list)

    for i, block in enumerate(blocks):
        for sub in itertools.combinations(block, t):
            subset_to_blocks[tuple(sub)].append(i)

    # A block is redundant if every t-subset it covers is also covered by another block
    keep = [True] * n
    for i, block in enumerate(blocks):
        if not keep[i]:
            continue
        is_redundant = True
        for sub in itertools.combinations(block, t):
            covering = subset_to_blocks[tuple(sub)]
            if len(covering) == 1 and covering[0] == i:
                is_redundant = False
                break
        if is_redundant:
            keep[i] = False

    new_blocks = [b for i, b in enumerate(blocks) if keep[i]]
    
    if not new_blocks:
        return blocks
        
    return new_blocks


def _local_search_polish(blocks, v, k, t, all_t_subsets, triple_to_idx, deadline, rng):
    """
    Local search polish: try to reduce block count by merging or swapping.
    Strategy: for each pair of blocks, check if one can be removed if the other
    is modified to cover the unique subsets of the removed block.
    """
    if not blocks or time.time() > deadline:
        return blocks
    
    current_blocks = [list(b) for b in blocks]
    current_len = len(current_blocks)
    
    # Build coverage map: t_subset -> set of block indices
    def build_coverage_map(blks):
        cov = defaultdict(set)
        for i, blk in enumerate(blks):
            for sub in itertools.combinations(blk, t):
                cov[tuple(sub)].add(i)
        return cov
    
    # Try to remove blocks by modifying others
    for iteration in range(3):  # Limit iterations
        if time.time() > deadline:
            break
        
        coverage = build_coverage_map(current_blocks)
        
        # Find blocks that are "almost" redundant
        for i in range(len(current_blocks)):
            if time.time() > deadline:
                break
            
            block_i = current_blocks[i]
            unique_subs = []
            for sub in itertools.combinations(block_i, t):
                if len(coverage[tuple(sub)]) == 1 and i in coverage[tuple(sub)]:
                    unique_subs.append(sub)
            
            if not unique_subs:
                # Block i is redundant, remove it
                current_blocks.pop(i)
                # Rebuild coverage
                coverage = build_coverage_map(current_blocks)
                break
        
        # If no redundant block found, try swapping points
        # For each block, try replacing one point with another to see if it
        # allows removal of another block
        for i in range(len(current_blocks)):
            if time.time() > deadline:
                break
            
            block_i = current_blocks[i]
            for pos in range(len(block_i)):
                orig_point = block_i[pos]
                for new_point in range(v):
                    if new_point == orig_point or new_point in block_i:
                        continue
                    
                    # Try the swap
                    block_i[pos] = new_point
                    new_block_i = tuple(sorted(block_i))
                    
                    # Check if this swap allows any block to be removed
                    # by covering its unique subsets
                    can_remove = -1
                    for j in range(len(current_blocks)):
                        if j == i:
                            continue
                        block_j = current_blocks[j]
                        # Check if all unique subsets of block_j are now covered by other blocks
                        all_covered = True
                        for sub in itertools.combinations(block_j, t):
                            # Check if sub is covered by any block other than j
                            covered = False
                            for m in range(len(current_blocks)):
                                if m == j:
                                    continue
                                if sub in itertools.combinations(current_blocks[m], t):
                                    covered = True
                                    break
                            if not covered:
                                all_covered = False
                                break
                        if all_covered:
                            can_remove = j
                            break
                    
                    if can_remove >= 0:
                        # Perform the swap and removal
                        current_blocks[i] = list(new_block_i)
                        current_blocks.pop(can_remove)
                        break
                    else:
                        # Revert
                        block_i[pos] = orig_point
                else:
                    continue
                break
    
    return [tuple(sorted(b)) for b in current_blocks]


def _is_valid_covering(blocks, v, k, t, all_t_subsets):
    """Check if the blocks form a valid covering."""
    if not blocks:
        return len(all_t_subsets) == 0
    
    covered = set()
    for block in blocks:
        if len(block) != k:
            return False
        if len(set(block)) != k:
            return False
        for x in block:
            if x < 0 or x >= v:
                return False
        for sub in itertools.combinations(block, t):
            covered.add(sub)
    
    return covered == set(all_t_subsets)

# EVOLVE-BLOCK-END
