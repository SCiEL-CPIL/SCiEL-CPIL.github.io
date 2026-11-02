import itertools
import time
import random
import sys

def _deadline_ok(deadline):
    return time.time() < deadline

def _prune_redundant(selected, block_covers, n_t, deadline):
    """Remove redundant blocks from selected set."""
    if not selected:
        return set()
    
    # Calculate coverage counts
    final_counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            final_counts[s] += 1
    
    # Try to remove blocks iteratively
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        for i in list(selected):
            # Check if block i is redundant
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

def _local_search(selected, blocks, block_covers, sub_to_blocks, n_t, deadline):
    """Local search: try to replace pairs of blocks with single blocks that cover same subsets."""
    if not selected:
        return selected
    
    # Calculate coverage counts
    final_counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            final_counts[s] += 1
    
    # Find blocks that could potentially be removed if we add another block
    # A block b is "removable" if all its subsets are covered by other selected blocks
    # OR if we can replace two blocks with one that covers all their subsets
    
    best_selected = set(selected)
    best_size = len(selected)
    
    # Try removing one block and adding another
    # For each block in selected, try to remove it and see if we can cover its unique subsets
    for b_remove in list(selected):
        if not _deadline_ok(deadline):
            break
        
        # Find subsets only covered by b_remove
        unique_subs = []
        for s in block_covers[b_remove]:
            if final_counts[s] == 1:
                unique_subs.append(s)
        
        if not unique_subs:
            # Block is redundant, remove it
            new_selected = set(selected)
            new_selected.discard(b_remove)
            if len(new_selected) < best_size:
                best_selected = new_selected
                best_size = len(new_selected)
            continue
        
        # Try to find a single block that covers all unique_subs and doesn't introduce new uncovered subsets
        # This is expensive, so we limit the search
        if len(unique_subs) > 3:
            continue
        
        # Find candidate blocks that cover all unique_subs
        candidate_blocks = None
        for s in unique_subs:
            if candidate_blocks is None:
                candidate_blocks = set(sub_to_blocks[s])
            else:
                candidate_blocks &= set(sub_to_blocks[s])
        
        if not candidate_blocks:
            continue
        
        # Try each candidate
        for b_add in candidate_blocks:
            if b_add in selected:
                continue
            
            # Check if adding b_add and removing b_remove keeps all subsets covered
            # New coverage = (old coverage - b_remove's contribution) + b_add's contribution
            # We need: for all s, (final_counts[s] - [s in block_covers[b_remove]] + [s in block_covers[b_add]]) >= 1
            
            valid = True
            for s in block_covers[b_remove]:
                if final_counts[s] == 1 and s not in block_covers[b_add]:
                    valid = False
                    break
            
            if valid:
                new_selected = set(selected)
                new_selected.discard(b_remove)
                new_selected.add(b_add)
                if len(new_selected) < best_size:
                    best_selected = new_selected
                    best_size = len(new_selected)
    
    return best_selected

def _greedy_cover(blocks, block_covers, sub_to_blocks, n_t, deadline, rng=None):
    """Greedy set cover with randomized tie-breaking."""
    if rng is None:
        rng = random.Random()
    
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
            
        # Get candidates for target_sub
        candidates = sub_to_blocks[target_sub]
        
        # Pick the candidate that maximizes coverage of other uncovered subsets
        best_block = -1
        max_gain = -1
        
        for b in candidates:
            gain = 0
            for s in block_covers[b]:
                if s in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = b
            elif gain == max_gain and gain > 0:
                # Random tie-breaking
                if rng.random() < 0.5:
                    best_block = b
                    max_gain = gain
        
        # Select best_block
        selected.add(best_block)
        
        # Update uncovered
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
    
    # 3. Multiple randomized greedy restarts with local search
    best_result = None
    best_size = float('inf')
    
    # Number of restarts depends on available time
    elapsed = time.time() - start
    remaining = deadline - time.time()
    # Estimate time per restart: greedy is O(n_blocks * n_t) roughly, local search adds more
    # Let's try at least 3 restarts if time permits
    n_restarts = max(3, int(remaining / 20))  # ~20s per restart estimate
    n_restarts = min(n_restarts, 10)  # Cap at 10
    
    rng = random.Random(42)
    
    for restart in range(n_restarts):
        if not _deadline_ok(deadline):
            break
        
        # Randomized greedy
        selected = _greedy_cover(blocks, block_covers, sub_to_blocks, n_t, deadline, rng)
        
        if not selected:
            continue
        
        # Prune redundant blocks
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        
        # Local search
        if _deadline_ok(deadline):
            selected = _local_search(selected, blocks, block_covers, sub_to_blocks, n_t, deadline)
        
        # Check if this is better
        if len(selected) < best_size:
            best_size = len(selected)
            best_result = set(selected)
    
    # If no result found, fall back to a single greedy
    if best_result is None:
        selected = _greedy_cover(blocks, block_covers, sub_to_blocks, n_t, deadline)
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        best_result = selected
    
    result = [list(blocks[i]) for i in best_result]
    
    # Verify validity (just in case)
    if _deadline_ok(deadline - 2.0):
        check_counts = [0] * n_t
        for i in best_result:
            for s in block_covers[i]:
                check_counts[s] += 1
        if any(c == 0 for c in check_counts):
            # Invalid, try again with a fresh greedy
            selected = _greedy_cover(blocks, block_covers, sub_to_blocks, n_t, deadline)
            selected = _prune_redundant(selected, block_covers, n_t, deadline)
            result = [list(blocks[i]) for i in selected]
            
    return result
