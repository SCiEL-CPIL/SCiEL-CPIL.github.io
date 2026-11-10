"""Covering design C(19,8,5) with greedy construction and local improvement."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using greedy block construction
    followed by local improvement (block replacement and deletion).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Track coverage: for each t-subset, how many blocks cover it
    coverage = [0] * n_t_subsets
    # For each t-subset, store the block index that covers it (or -1)
    covering_block = [-1] * n_t_subsets
    
    blocks = []
    
    def time_check():
        return time.time() >= deadline
    
    # Phase 1: Greedy construction
    uncovered_count = n_t_subsets
    
    while uncovered_count > 0:
        if time_check():
            break
        
        # Find the first uncovered t-subset
        first_uncovered = None
        for i in range(n_t_subsets):
            if coverage[i] == 0:
                first_uncovered = i
                break
        if first_uncovered is None:
            break
        
        base_subset = all_t_subsets[first_uncovered]
        block = set(base_subset)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Calculate gain: number of uncovered t-subsets that would be covered
                # by adding x to the current block
                gain = 0
                # A t-subset is covered by adding x if it contains x and t-1 elements from block
                # and is currently uncovered
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    idx = subset_to_idx.get(candidate)
                    if idx is not None and coverage[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is not None:
                block.add(best_point)
        
        block_list = sorted(block)
        blocks.append(block_list)
        block_idx = len(blocks) - 1
        
        # Update coverage
        for subset in itertools.combinations(block_list, t):
            idx = subset_to_idx[subset]
            coverage[idx] += 1
            if covering_block[idx] == -1:
                covering_block[idx] = block_idx
        
        # Recalculate uncovered count
        uncovered_count = sum(1 for c in coverage if c == 0)
    
    # Phase 2: Local improvement - try to remove redundant blocks
    def try_remove_block(block_idx):
        """Try removing a block if all its t-subsets are covered by other blocks."""
        if block_idx >= len(blocks):
            return False
        block = blocks[block_idx]
        # Check if all t-subsets in this block are covered by at least one other block
        for subset in itertools.combinations(block, t):
            idx = subset_to_idx[subset]
            if coverage[idx] <= 1:
                return False
        # Remove the block
        del blocks[block_idx]
        # Update coverage and reindex covering_block
        for subset in itertools.combinations(block, t):
            idx = subset_to_idx[subset]
            coverage[idx] -= 1
        # Reindex covering_block
        for i in range(n_t_subsets):
            if covering_block[i] > block_idx:
                covering_block[i] -= 1
            elif covering_block[i] == block_idx:
                # Find another block that covers this subset
                found = False
                for bi in range(len(blocks)):
                    if subset in itertools.combinations(blocks[bi], t):
                        covering_block[i] = bi
                        found = True
                        break
                if not found:
                    covering_block[i] = -1
        return True
    
    # Try removing blocks multiple times
    improved = True
    while improved and not time_check():
        improved = False
        for bi in range(len(blocks) - 1, -1, -1):
            if try_remove_block(bi):
                improved = True
                break
    
    # Phase 3: Try replacing blocks with better ones
    # For each block, try to find a replacement that covers more uncovered subsets
    for bi in range(len(blocks)):
        if time_check():
            break
        if bi >= len(blocks):
            continue
        old_block = blocks[bi]
        
        # Find t-subsets only covered by this block
        exclusive_subsets = []
        for subset in itertools.combinations(old_block, t):
            idx = subset_to_idx[subset]
            if coverage[idx] == 1:
                exclusive_subsets.append(idx)
        
        if not exclusive_subsets:
            continue
        
        # Try to construct a new block that covers these exclusive subsets
        # and more uncovered subsets
        new_block = None
        best_new_gain = 0
        
        # Start with a random subset of points from old block plus some new points
        for _ in range(20):  # Try multiple random starts
            if time_check():
                break
            
            # Start with some points from old block
            candidate = set()
            # Add points that appear in exclusive subsets
            for idx in exclusive_subsets:
                for p in all_t_subsets[idx]:
                    if len(candidate) < k:
                        candidate.add(p)
            
            # Fill remaining slots greedily
            while len(candidate) < k:
                best_x = None
                best_g = -1
                for x in range(v):
                    if x in candidate:
                        continue
                    g = 0
                    for rest in itertools.combinations(candidate, t - 1):
                        cand_subset = tuple(sorted(rest + (x,)))
                        idx = subset_to_idx.get(cand_subset)
                        if idx is not None and coverage[idx] == 0:
                            g += 1
                    if g > best_g:
                        best_g = g
                        best_x = x
                if best_x is not None:
                    candidate.add(best_x)
                else:
                    break
            
            if len(candidate) != k:
                continue
            
            # Check if this candidate covers all exclusive subsets
            covers_all = True
            for idx in exclusive_subsets:
                subset = all_t_subsets[idx]
                if not set(subset).issubset(candidate):
                    covers_all = False
                    break
            
            if not covers_all:
                continue
            
            # Calculate total gain: uncovered subsets covered by new block
            gain = 0
            for subset in itertools.combinations(sorted(candidate), t):
                idx = subset_to_idx[subset]
                if coverage[idx] == 0:
                    gain += 1
            
            if gain > best_new_gain:
                best_new_gain = gain
                new_block = sorted(candidate)
        
        if new_block is not None and best_new_gain > 0:
            # Replace the block
            old_block_set = set(old_block)
            new_block_set = set(new_block)
            
            # Update coverage for subsets in old block but not in new
            for subset in itertools.combinations(old_block, t):
                idx = subset_to_idx[subset]
                if not set(subset).issubset(new_block_set):
                    coverage[idx] -= 1
                    if covering_block[idx] == bi:
                        covering_block[idx] = -1
            
            # Update coverage for subsets in new block but not in old
            for subset in itertools.combinations(new_block, t):
                idx = subset_to_idx[subset]
                if not set(subset).issubset(old_block_set):
                    coverage[idx] += 1
                    if covering_block[idx] == -1:
                        covering_block[idx] = bi
            
            blocks[bi] = new_block
    
    # Final cleanup: remove any redundant blocks
    for bi in range(len(blocks) - 1, -1, -1):
        if time_check():
            break
        if try_remove_block(bi):
            pass  # Block removed
    
    return blocks
# EVOLVE-BLOCK-END
