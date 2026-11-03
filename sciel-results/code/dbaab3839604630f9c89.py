import itertools
import time
import random
import sys
import numpy as np
from heapq import heappush, heappop, heapify

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start_time = time.time()
    limit = 115.0
    deadline = start_time + limit

    # 1. Precomputation
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # Map each block to the indices of the t-subsets it covers
    block_covers = []
    for block in blocks:
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        block_covers.append(np.array(covers, dtype=np.int32))
        
    if not _deadline_ok(deadline):
        return []

    # Build reverse mapping: for each t-subset, which blocks cover it?
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
    
    # Pre-compute subset degrees (number of blocks covering each subset)
    subset_degrees = np.array([len(lst) for lst in sub_to_blocks], dtype=np.int32)
    
    # 2. Fast Greedy Initialization with Dual Variables
    # Use a primal-dual approach: maintain dual variables for uncovered subsets
    # and select blocks that maximize the ratio of dual weight to cost
    
    uncovered_mask = np.ones(n_t, dtype=bool)
    selected = set()
    
    # Initialize dual variables (start at 1.0 for all uncovered)
    dual = np.ones(n_t, dtype=np.float64)
    
    # Priority queue for most-constrained uncovered subsets
    # (degree, subset_index) - we want low degree first
    pq = [(int(subset_degrees[i]), i) for i in range(n_t)]
    heapify(pq)
    
    # Track which subsets are still in the heap (to avoid stale entries)
    in_heap = np.ones(n_t, dtype=bool)
    
    while np.any(uncovered_mask):
        if not _deadline_ok(deadline):
            break
            
        # Find the most constrained uncovered subset
        target_idx = -1
        while pq:
            deg, idx = heappop(pq)
            if in_heap[idx] and uncovered_mask[idx]:
                target_idx = idx
                break
        
        if target_idx == -1:
            break
            
        candidates = sub_to_blocks[target_idx]
        if not candidates:
            break
            
        # Among candidates, pick the one that maximizes "dual gain"
        # i.e., sum of dual values of uncovered subsets it covers
        best_cand = -1
        max_gain = 0
        
        # Limit candidates checked for speed
        # If too many, sample
        if len(candidates) > 50:
            sample = random.sample(candidates, 50)
        else:
            sample = candidates
            
        for b in sample:
            gain = np.sum(dual[block_covers[b]] * uncovered_mask[block_covers[b]])
            if gain > max_gain:
                max_gain = gain
                best_cand = b
                
        if best_cand != -1:
            selected.add(best_cand)
            covers = block_covers[best_cand]
            uncovered_mask[covers] = False
            
            # Update dual variables: increase dual for newly covered subsets
            # This makes them less attractive for future selections
            # and increases the value of blocks covering remaining uncovered subsets
            dual[covers] *= 0.5  # Discount covered subsets
            
            # Update priority queue for newly covered subsets
            for s in covers:
                if in_heap[s] and not uncovered_mask[s]:
                    in_heap[s] = False
    
    if not selected:
        return []

    # 3. Local Search Optimization
    # Initialize counts
    counts = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        counts[block_covers[b]] += 1
    
    # Helper: Check if a block is redundant given current counts
    def is_redundant(b, counts):
        return np.all(counts[block_covers[b]] > 1)
        
    # Helper: Try to remove a block. If it causes gaps, try to fix.
    def try_remove_and_fix(b, selected, counts, deadline):
        """
        Attempts to remove block b.
        Returns True if successful (and updates selected/counts), False otherwise.
        """
        if not _deadline_ok(deadline):
            return False
            
        # Check if b is redundant
        if is_redundant(b, counts):
            selected.remove(b)
            counts[block_covers[b]] -= 1
            return True
            
        # If not redundant, it covers some subsets with count == 1.
        covers_b = block_covers[b]
        critical_subsets = covers_b[counts[covers_b] == 1]
        
        if len(critical_subsets) == 0:
            selected.remove(b)
            counts[covers_b] -= 1
            return True
            
        # Find candidate blocks that cover ALL critical_subsets
        if len(critical_subsets) == 1:
            candidates = sub_to_blocks[critical_subsets[0]]
        else:
            # Find intersection
            min_count = float('inf')
            target = -1
            for s in critical_subsets:
                c = len(sub_to_blocks[s])
                if c < min_count:
                    min_count = c
                    target = s
            
            candidates = set(sub_to_blocks[target])
            for s in critical_subsets:
                if s == target: continue
                candidates.intersection_update(sub_to_blocks[s])
                if not candidates:
                    return False
            
            candidates = list(candidates)
            
        # Filter candidates: must not be in selected
        valid_candidates = [c for c in candidates if c not in selected]
        
        if not valid_candidates:
            return False
            
        # Pick the candidate that minimizes "waste" - covers fewest already-well-covered subsets
        best_b_new = -1
        min_waste = float('inf')
        
        # Limit candidates checked
        if len(valid_candidates) > 30:
            sample = random.sample(valid_candidates, 30)
        else:
            sample = valid_candidates
            
        for b_new in sample:
            # Waste = number of subsets covered by b_new that have count > 1
            # (excluding the critical subsets which will have count 1 after removal)
            covers_new = block_covers[b_new]
            # Count subsets with count > 1 (these are "wasted" coverage)
            waste = np.sum(counts[covers_new] > 1)
            if waste < min_waste:
                min_waste = waste
                best_b_new = b_new
        
        if best_b_new == -1:
            return False
            
        # Perform swap
        selected.remove(b)
        counts[covers_b] -= 1
        selected.add(best_b_new)
        counts[block_covers[best_b_new]] += 1
        
        return True

    # Main Local Search Loop
    best_size = len(selected)
    best_selected = set(selected)
    best_counts = counts.copy()
    
    # Keep a list for fast random access
    selected_list = list(selected)
    
    # Track consecutive failures
    consecutive_failures = 0
    max_consecutive_failures = 50
    
    while _deadline_ok(deadline):
        if not selected_list:
            break
            
        # Pick a random block to try to remove
        idx = random.randint(0, len(selected_list) - 1)
        b = selected_list[idx]
        
        success = try_remove_and_fix(b, selected, counts, deadline)
        
        if success:
            # Update the list representation
            selected_list = list(selected)
            consecutive_failures = 0
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = set(selected)
                best_counts = counts.copy()
        else:
            consecutive_failures += 1
            
            # If too many consecutive failures, try a different strategy
            if consecutive_failures >= max_consecutive_failures:
                # Try removing multiple blocks at once
                # Find a set of blocks that can be removed together
                # This is complex, so instead restart local search with different random seed
                consecutive_failures = 0
                # Shuffle the order to break symmetry
                random.shuffle(selected_list)
    
    # Final Result
    if len(best_selected) < len(selected):
        selected = best_selected
        
    # Convert to list of lists
    result = [list(blocks[i]) for i in selected]
    return result
