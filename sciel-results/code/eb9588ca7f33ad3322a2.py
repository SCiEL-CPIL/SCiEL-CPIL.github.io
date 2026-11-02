import itertools
import time
import random
import sys
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def _prune_redundant(selected, block_covers, n_t, deadline):
    """Remove redundant blocks from selected set."""
    if not selected:
        return selected
    
    final_counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            final_counts[s] += 1
            
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        for i in list(selected):
            redundant = True
            for s in block_covers[i]:
                if final_counts[s] <= 1:
                    redundant = False
                    break
            if redundant:
                selected.remove(i)
                for s in block_covers[i]:
                    final_counts[s] -= 1
                changed = True
                if not _deadline_ok(deadline):
                    break
    return selected

def _local_search(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=200):
    """Try to improve the solution by swapping blocks."""
    if not selected or len(selected) < 2:
        return selected
    
    # Build coverage counts
    counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            counts[s] += 1
    
    best_selected = set(selected)
    best_size = len(best_selected)
    
    for iteration in range(max_iterations):
        if not _deadline_ok(deadline):
            break
        
        if not best_selected:
            break
        
        # Try to remove a random block
        block_to_remove = random.choice(list(best_selected))
        
        # Find uncovered subsets if we remove this block
        uncovered_if_removed = []
        for s in block_covers[block_to_remove]:
            if counts[s] <= 1:
                uncovered_if_removed.append(s)
        
        if not uncovered_if_removed:
            # Block is redundant, remove it
            best_selected.remove(block_to_remove)
            for s in block_covers[block_to_remove]:
                counts[s] -= 1
            best_size -= 1
            continue
        
        # Try to find a replacement block that covers all uncovered_if_removed
        best_replacement = None
        
        # Get candidate blocks from the most constrained uncovered subset
        min_count = float('inf')
        target_sub = -1
        for s in uncovered_if_removed:
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target_sub = s
                if c == 1:
                    break
        
        if target_sub == -1:
            continue
        
        candidates = sub_to_blocks[target_sub]
        
        # Try candidates, prefer those that cover more uncovered subsets
        best_gain = -1
        for b in candidates:
            if b in best_selected:
                continue
            
            # Check if this block covers all uncovered_if_removed
            covers_all = True
            for s in uncovered_if_removed:
                if s not in block_covers[b]:
                    covers_all = False
                    break
            
            if not covers_all:
                continue
            
            # Count how many currently covered subsets this block would add
            gain = 0
            for s in block_covers[b]:
                if counts[s] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_replacement = b
        
        if best_replacement is not None:
            # Perform the swap
            best_selected.remove(block_to_remove)
            for s in block_covers[block_to_remove]:
                counts[s] -= 1
            best_selected.add(best_replacement)
            for s in block_covers[best_replacement]:
                counts[s] += 1
    
    return best_selected

def _greedy_cover(block_covers, n_t, sub_to_blocks, deadline, randomize=False, top_k=10):
    """Greedy set cover with optional randomization."""
    uncovered = set(range(n_t))
    selected = set()
    
    while uncovered:
        if not _deadline_ok(deadline):
            break
        
        # Find the uncovered subset with fewest candidate blocks
        min_count = float('inf')
        target_sub = -1
        for s in uncovered:
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target_sub = s
                if c == 1:
                    break
        
        if target_sub == -1:
            break
        
        candidates = sub_to_blocks[target_sub]
        
        # Calculate gains for all candidates
        gains = []
        for b in candidates:
            gain = 0
            for s in block_covers[b]:
                if s in uncovered:
                    gain += 1
            gains.append((gain, b))
        
        # Sort by gain descending
        gains.sort(reverse=True)
        
        if not randomize:
            best_block = gains[0][1]
        else:
            top_k_actual = min(top_k, len(gains))
            best_block = random.choice(gains[:top_k_actual])[1]
        
        selected.add(best_block)
        
        for s in block_covers[best_block]:
            uncovered.discard(s)
    
    return selected

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their covered t-subsets
    blocks = []
    block_covers = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        block_covers.append(covers)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # Build sub_to_blocks
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []

    # 3. Greedy Set Cover with multiple random restarts
    best_selected = None
    best_size = float('inf')
    
    num_restarts = 8
    
    for restart in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        
        # Randomized greedy
        selected = _greedy_cover(block_covers, n_t, sub_to_blocks, deadline, 
                                  randomize=(restart > 0), top_k=10)
        
        # Prune redundant blocks
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        
        # Local search
        selected = _local_search(selected, block_covers, n_t, sub_to_blocks, deadline)
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
        
        # If we have time, do more local search on the best solution
        if _deadline_ok(deadline) and best_selected is not None:
            best_selected = _local_search(best_selected, block_covers, n_t, sub_to_blocks, deadline)
            if len(best_selected) < best_size:
                best_size = len(best_selected)
    
    # Final pruning
    if best_selected is not None:
        best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
    
    # Convert to result
    result = [list(blocks[i]) for i in best_selected] if best_selected else []
    
    return result
