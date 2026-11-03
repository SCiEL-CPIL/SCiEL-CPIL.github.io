"""Improved covering design C(24,7,3) solver using vectorized greedy + local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from collections import Counter

def construct(v, k, t):
    """Construct a covering design C(v,k,t) using vectorized greedy and local search."""
    deadline = time.time() + 115  # safety margin before 120s limit
    
    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # Precompute for each point and (t-1)-subset, the list of t-subset indices
    # that would be covered by adding this point
    # For efficiency, precompute for each point x, the set of t-subsets containing x
    point_tsubs = {}
    for x in points:
        point_tsubs[x] = set(i for i, sub in enumerate(all_tsubs) if x in sub)
    
    # Greedy covering with vectorized gain computation
    uncovered = np.ones(num_tsubs, dtype=np.int32)  # 1 = uncovered
    blocks = []
    
    def compute_gains_for_block(current_block):
        """Compute gain of adding each point to current_block."""
        if len(current_block) < t - 1:
            # Need at least t-1 points to form any (t-1)-subset
            # All gains are 0
            return np.zeros(v, dtype=np.int32)
        
        # Get all (t-1)-subsets of current_block
        current_set = set(current_block)
        t_minus_1 = t - 1
        
        # For each candidate point x not in current_block,
        # gain = number of uncovered t-subsets that contain x and (t-1) points from current_block
        gains = np.zeros(v, dtype=np.int32)
        
        # Precompute: for each (t-1)-subset of current_block, which t-subsets extend it
        # A t-subset extends (t-1)-subset S if it contains S and one more point
        # So for (t-1)-subset S, the t-subsets extending it are S ∪ {x} for x not in S
        
        # Build a mapping: for each (t-1)-subset of current_block, store the t-subset indices
        # that extend it
        # Then for point x, gain = sum over (t-1)-subsets S of current_block where x not in S
        #   of uncovered[S ∪ {x}]
        
        # More efficient: iterate over (t-1)-subsets of current_block
        for rest in itertools.combinations(current_block, t_minus_1):
            rest_set = set(rest)
            # For each x not in rest_set and not in current_block,
            # the t-subset rest + (x,) has index tsub_to_idx[tuple(sorted(rest + (x,)))]
            for x in points:
                if x not in rest_set:
                    tsub = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx[tsub]
                    if uncovered[idx] == 1:
                        gains[x] += 1
        
        # Zero out gains for points already in current_block
        for x in current_block:
            gains[x] = 0
        
        return gains
    
    while np.any(uncovered == 1) and time.time() < deadline - 5:
        # Start a new block: pick the first uncovered t-subset
        first_uncovered = np.where(uncovered == 1)[0][0]
        first_tsub = all_tsubs[first_uncovered]
        block = list(first_tsub)
        
        while len(block) < k and time.time() < deadline - 5:
            gains = compute_gains_for_block(block)
            if np.max(gains) <= 0:
                # No point improves coverage; just add a random unused point
                unused = [x for x in points if x not in set(block)]
                if unused:
                    block.append(unused[0])
                else:
                    break
            else:
                best_x = int(np.argmax(gains))
                block.append(best_x)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update uncovered
        block_set = set(block)
        for i, tsub in enumerate(all_tsubs):
            if uncovered[i] == 1 and all(x in block_set for x in tsub):
                uncovered[i] = 0
    
    # Local search: try to remove or swap points in blocks to reduce total block count
    # Strategy: try removing a block if all its t-subsets are still covered
    # Also try replacing a block with a better one
    
    def is_covered_without_block(blocks_list, block_to_remove):
        """Check if all t-subsets in block_to_remove are covered by other blocks."""
        block_set = set(block_to_remove)
        # Find all t-subsets that are only covered by this block
        # A t-subset is "only covered" by this block if it's contained in this block
        # and not contained in any other block
        for tsub in itertools.combinations(block_to_remove, t):
            tsub_set = set(tsub)
            covered = False
            for b in blocks_list:
                if b == block_to_remove:
                    continue
                if tsub_set.issubset(set(b)):
                    covered = True
                    break
            if not covered:
                return False
        return True
    
    # Try to remove redundant blocks
    changed = True
    while changed and time.time() < deadline - 3:
        changed = False
        for i in range(len(blocks)):
            if is_covered_without_block(blocks, blocks[i]):
                blocks.pop(i)
                changed = True
                break
    
    # Try local search: for each block, try swapping one point
    # to see if we can reduce the number of blocks needed
    # Simpler: try to find a block that can be replaced to cover more uncovered t-subsets
    # But since we already have a full covering, try to merge or replace
    
    # Try a simple local search: for each pair of blocks, see if one can be removed
    # after adjusting the other
    for _ in range(100):  # limited iterations
        if time.time() > deadline - 3:
            break
        improved = False
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                # Try removing block i
                if is_covered_without_block(blocks, blocks[i]):
                    blocks.pop(i)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
