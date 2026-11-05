"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 115s deadline to be safe within 120s limit
    
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized restarts with different strategies
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        blocks = _greedy_cover_ratio(v, k, t, subset_to_idx, all_t_subsets, deadline)
        if blocks is not None and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
        # Try local search improvement
        if best_blocks is not None:
            improved = _local_search(best_blocks, v, k, t, subset_to_idx, all_t_subsets, deadline)
            if improved is not None and len(improved) < best_count:
                best_count = len(improved)
                best_blocks = [b[:] for b in improved]
        
        # If we have plenty of time left, do another full restart
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
    
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = _greedy_cover_abs(v, k, t, subset_to_idx, all_t_subsets, deadline)
    
    return best_blocks

def _greedy_cover_ratio(v, k, t, subset_to_idx, all_t_subsets, deadline):
    """Greedy set cover using ratio of newly covered triplets to block size."""
    coverage = np.zeros(len(all_t_subsets), dtype=np.int32)
    uncovered_count = len(all_t_subsets)
    blocks = []
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        # Find an uncovered t-subset to seed the block
        uncovered_indices = np.where(coverage == 0)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Randomly pick an uncovered subset to start
        seed_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
        seed_subset = all_t_subsets[seed_idx]
        
        # Build block greedily using ratio criterion
        block = list(seed_subset)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_x = -1
            best_ratio = -1
            
            for x in range(v):
                if x in block:
                    continue
                
                # Calculate gain: number of new t-subsets covered by adding x
                gain = 0
                for S in itertools.combinations(block, t-1):
                    new_subset = tuple(sorted(S + (x,)))
                    idx = subset_to_idx[new_subset]
                    if coverage[idx] == 0:
                        gain += 1
                
                # Ratio: gain / (size of new block)
                # This penalizes blocks that don't add much coverage per element
                new_block_size = len(block) + 1
                ratio = gain / new_block_size if new_block_size > 0 else 0
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_x = x
                    
            if best_x == -1:
                break
                
            block.append(best_x)
        
        if len(block) < k:
            # Pad with random elements if needed
            remaining = [x for x in range(v) if x not in block]
            random.shuffle(remaining)
            for x in remaining[:k - len(block)]:
                block.append(x)
        
        block.sort()
        
        # Update coverage
        newly_covered = 0
        for S in itertools.combinations(block, t):
            idx = subset_to_idx[S]
            if coverage[idx] == 0:
                coverage[idx] = 1
                newly_covered += 1
            else:
                coverage[idx] += 1
                
        uncovered_count -= newly_covered
        blocks.append(block)
        
    return blocks if len(blocks) > 0 else None

def _greedy_cover_abs(v, k, t, subset_to_idx, all_t_subsets, deadline):
    """Fallback: absolute gain greedy."""
    coverage = np.zeros(len(all_t_subsets), dtype=np.int32)
    uncovered_count = len(all_t_subsets)
    blocks = []
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        uncovered_indices = np.where(coverage == 0)[0]
        if len(uncovered_indices) == 0:
            break
        
        seed_idx = uncovered_indices[0]
        seed_subset = all_t_subsets[seed_idx]
        block = list(seed_subset)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_x = -1
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                for S in itertools.combinations(block, t-1):
                    new_subset = tuple(sorted(S + (x,)))
                    idx = subset_to_idx[new_subset]
                    if coverage[idx] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
            if best_x == -1:
                break
                
            block.append(best_x)
        
        if len(block) < k:
            remaining = [x for x in range(v) if x not in block]
            random.shuffle(remaining)
            for x in remaining[:k - len(block)]:
                block.append(x)
        
        block.sort()
        
        newly_covered = 0
        for S in itertools.combinations(block, t):
            idx = subset_to_idx[S]
            if coverage[idx] == 0:
                coverage[idx] = 1
                newly_covered += 1
            else:
                coverage[idx] += 1
                
        uncovered_count -= newly_covered
        blocks.append(block)
        
    return blocks

def _local_search(blocks, v, k, t, subset_to_idx, all_t_subsets, deadline):
    """Try to improve coverage by removing redundant blocks or swapping elements."""
    if blocks is None or len(blocks) == 0:
        return None
    
    # First pass: remove redundant blocks
    blocks = _remove_redundant(blocks, v, k, t, subset_to_idx, all_t_subsets, deadline)
    
    # Second pass: try swapping elements in blocks to improve coverage
    for i in range(len(blocks)):
        if time.time() > deadline:
            break
        
        block = blocks[i][:]
        for j in range(len(block)):
            if time.time() > deadline:
                break
            
            orig_elem = block[j]
            best_swap = None
            best_improvement = 0
            
            for x in range(v):
                if x in block:
                    continue
                
                # Try swapping orig_elem with x
                new_block = block[:]
                new_block[j] = x
                new_block.sort()
                
                # Calculate improvement
                improvement = _calculate_improvement(new_block, blocks, i, v, k, t, subset_to_idx, all_t_subsets)
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_swap = x
            
            if best_swap is not None:
                block[j] = best_swap
                block.sort()
                blocks[i] = block
    
    return blocks

def _calculate_improvement(new_block, blocks, block_idx, v, k, t, subset_to_idx, all_t_subsets):
    """Calculate how much improvement a new block provides over the current one."""
    # Count coverage before and after
    coverage_before = np.zeros(len(all_t_subsets), dtype=np.int32)
    for i, b in enumerate(blocks):
        if i == block_idx:
            continue
        for S in itertools.combinations(b, t):
            coverage_before[subset_to_idx[S]] += 1
    
    # Add new block
    coverage_after = coverage_before.copy()
    for S in itertools.combinations(new_block, t):
        coverage_after[subset_to_idx[S]] += 1
    
    # Count newly covered subsets
    newly_covered = 0
    for S in itertools.combinations(new_block, t):
        idx = subset_to_idx[S]
        if coverage_before[idx] == 0 and coverage_after[idx] > 0:
            newly_covered += 1
    
    # Count lost coverage (subsets only covered by old block)
    old_block = blocks[block_idx]
    lost = 0
    for S in itertools.combinations(old_block, t):
        idx = subset_to_idx[S]
        if coverage_before[idx] == 1:  # Only covered by this block
            lost += 1
    
    return newly_covered - lost

def _remove_redundant(blocks, v, k, t, subset_to_idx, all_t_subsets, deadline):
    """Remove blocks that don't contribute unique coverage."""
    if not blocks:
        return blocks
    
    # Build coverage map
    coverage = Counter()
    for block in blocks:
        for S in itertools.combinations(block, t):
            coverage[S] += 1
    
    # Try removing each block
    new_blocks = []
    for i, block in enumerate(blocks):
        if time.time() > deadline:
            new_blocks.append(block)
            continue
        
        # Check if this block is redundant
        is_redundant = True
        for S in itertools.combinations(block, t):
            if coverage[S] == 1:  # Only covered by this block
                is_redundant = False
                break
        
        if not is_redundant:
            new_blocks.append(block)
    
    return new_blocks
# EVOLVE-BLOCK-END
