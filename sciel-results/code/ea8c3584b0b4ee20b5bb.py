import itertools
import time
import random
import sys
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start_time = time.time()
    # Evaluator limit is 150s, but we should leave a safety margin for overhead/return
    # Current program used 115s. Let's use 140s to allow more search time.
    limit = 140.0 
    deadline = start_time + limit

    # 1. Precomputation
    # Generate all t-subsets
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # Generate all k-blocks
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # Map each block to the indices of the t-subsets it covers
    # block_covers[i] is a list of subset indices covered by block i
    block_covers = []
    for block in blocks:
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        block_covers.append(np.array(covers, dtype=np.int32))
        
    if not _deadline_ok(deadline):
        return []

    # Build reverse mapping: for each t-subset, which blocks cover it?
    # sub_to_blocks[j] is a list of block indices covering subset j
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
    
    # 2. Fast Greedy Initialization
    # We need a valid solution to start local search.
    # Greedy: repeatedly pick the block that covers the most uncovered subsets.
    
    uncovered_mask = np.ones(n_t, dtype=bool)
    selected = set()
    
    # Simple Greedy Loop
    # We will use a "most constrained" heuristic to speed up convergence
    # and get a tighter initial bound than pure random.
    
    # Precompute static counts of blocks per subset for heuristic
    subset_counts = [len(lst) for lst in sub_to_blocks]
    
    while np.any(uncovered_mask):
        if not _deadline_ok(deadline):
            break
            
        # Find uncovered subset with minimum count (most constrained)
        # To avoid O(n_t) scan every time which might be slow if many steps,
        # we can sample or use a heap. 
        # For v=19, k=8, t=5, n_t=11628.
        # Number of greedy steps is roughly n_blocks / avg_gain.
        # Let's try to be smart: pick the uncovered subset with fewest options.
        
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
            
        # Sample a few uncovered subsets to find a good candidate to reduce overhead
        # Or just pick the global min if n_t is small enough.
        # n_t is ~10k. Scanning 10k items is fast in Python/Numpy.
        # Let's scan to find the min count among uncovered.
        
        min_count = float('inf')
        target_idx = -1
        
        # Vectorized min finding is tricky with dynamic subset.
        # Looping over uncovered indices might be slow if there are many.
        # However, as we cover, the number of uncovered decreases.
        # Early on, there are many.
        # Let's just pick a random uncovered subset to keep it O(1) roughly for speed,
        # but add a small check: if the random one has very few options, use it.
        
        # Actually, the "most constrained" heuristic is key for good greedy.
        # Let's optimize the scan.
        # Convert subset_counts to numpy for fast indexing?
        # subset_counts_np = np.array(subset_counts)
        # min_val = np.min(subset_counts_np[uncovered_indices])
        # target_idx = uncovered_indices[np.argmin(subset_counts_np[uncovered_indices])]
        
        # This is O(n_uncovered) per step.
        # Let's try it.
        subset_counts_np = np.array(subset_counts, dtype=np.int32)
        
        if len(uncovered_indices) > 0:
             min_count_val = np.min(subset_counts_np[uncovered_indices])
             # Find one index with this min count
             candidates_for_target = uncovered_indices[subset_counts_np[uncovered_indices] == min_count_val]
             target_idx = random.choice(candidates_for_target)
        else:
             break
             
        candidates = sub_to_blocks[target_idx]
        if not candidates:
            break # Should not happen
            
        # Pick the candidate that covers the most other uncovered subsets
        best_cand = -1
        max_gain = 0
        for b in candidates:
            gain = np.sum(uncovered_mask[block_covers[b]])
            if gain > max_gain:
                max_gain = gain
                best_cand = b
                
        if best_cand != -1:
            selected.add(best_cand)
            uncovered_mask[block_covers[best_cand]] = False

    if not selected:
        return []

    # 3. Local Search Optimization
    # We now have a valid cover. We want to minimize |selected|.
    
    # Initialize counts
    counts = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        counts[block_covers[b]] += 1
    
    # Helper: Check if a block is redundant given current counts
    def is_redundant(b, counts):
        return np.all(counts[block_covers[b]] > 1)
        
    # Helper: Try to remove a block. If it causes gaps, try to fix.
    def try_remove_and_fix(b, selected, counts, deadline):
        if not _deadline_ok(deadline):
            return False
            
        # Check if b is redundant
        if is_redundant(b, counts):
            selected.remove(b)
            counts[block_covers[b]] -= 1
            return True
            
        covers_b = block_covers[b]
        critical_subsets = covers_b[counts[covers_b] == 1]
        
        if len(critical_subsets) == 0:
            selected.remove(b)
            counts[covers_b] -= 1
            return True
            
        # Find candidate blocks that cover ALL critical_subsets.
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
            
        valid_candidates = [c for c in candidates if c not in selected]
        
        if not valid_candidates:
            return False
            
        # Heuristic for picking b_new:
        # We want to minimize the "waste" or increase in future criticality.
        # A good metric: Pick the block that covers the most subsets with count > 1 
        # (making them more robust) or covers fewer subsets with count == 1 
        # (avoiding creating new single-coverage points).
        
        # Let's score candidates:
        # Score = - (number of subsets covered by b_new that are currently count==1 and NOT in critical_subsets)
        # Because adding b_new to a count==1 subset (not critical) makes it count==2, which is good.
        # Wait, if it's count==1 and NOT critical, it means it's covered by some other block.
        # Adding b_new makes it count==2. This is safe and good.
        # If it's count==0, it's uncovered. But we are in a valid cover, so no count==0.
        # So all subsets covered by b_new have count >= 1.
        # The critical ones become count==1 (after removing b, before adding b_new? No, we remove b first? 
        # The logic in try_remove_and_fix removes b, then adds b_new.
        # So during evaluation, counts still include b.
        # If we remove b, critical_subsets become count 0.
        # Other subsets covered by b become count-1.
        # Subsets covered by b_new (not in b) become count+1.
        # Subsets in intersection become count unchanged (remove b, add b_new).
        
        # Let's just pick the one that maximizes the number of subsets it covers that are currently "safe" (count > 1)?
        # Or simply random for diversity. 
        # Let's try to pick the one that minimizes the number of subsets it covers that are currently count==1 
        # (excluding the critical ones which MUST be covered).
        # If b_new covers a subset s with count==1 (and s is not critical), then after swap, s has count==2. 
        # This is an improvement in robustness.
        # If b_new covers a subset s with count==1 (and s IS critical), it becomes count==1. No change in robustness for s.
        
        # So we want to maximize the number of "non-critical" count==1 subsets covered by b_new.
        
        best_b_new = -1
        best_score = -float('inf')
        
        # Limit candidates if too many to save time?
        # Usually intersection is small.
        for b_new in valid_candidates:
            covers_new = block_covers[b_new]
            # Count how many subsets in covers_new have count == 1
            # And are NOT in critical_subsets
            # critical_subsets is an array of indices.
            # We can check membership.
            
            # Vectorized approach:
            # counts == 1 mask
            is_one = (counts == 1)
            # is_critical mask
            is_critical = np.zeros(n_t, dtype=bool)
            is_critical[critical_subsets] = True
            
            # We want subsets that are count==1 AND NOT critical
            good_targets = is_one & ~is_critical
            
            score = np.sum(good_targets[covers_new])
            
            if score > best_score:
                best_score = score
                best_b_new = b_new
                
        if best_b_new == -1:
            # Fallback to random
            best_b_new = random.choice(valid_candidates)

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
    
    selected_list = list(selected)
    
    # Add a "redundancy sweep" phase periodically to quickly remove obvious redundancies
    # before doing swaps.
    
    last_sweep_time = start_time
    sweep_interval = 1.0 # every 1 second
    
    while _deadline_ok(deadline):
        # Periodic redundancy sweep
        if time.time() - last_sweep_time > sweep_interval:
            last_sweep_time = time.time()
            # Try to remove any redundant block found in a random scan
            for _ in range(10): # 10 attempts per sweep
                if not selected_list: break
                idx = random.randint(0, len(selected_list) - 1)
                b = selected_list[idx]
                if is_redundant(b, counts):
                    selected.remove(b)
                    counts[block_covers[b]] -= 1
                    selected_list = list(selected)
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
                        best_counts = counts.copy()
        
        if not selected_list:
            break
            
        # Pick a random block to try to remove/swap
        idx = random.randint(0, len(selected_list) - 1)
        b = selected_list[idx]
        
        success = try_remove_and_fix(b, selected, counts, deadline)
        
        if success:
            selected_list = list(selected)
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = set(selected)
                best_counts = counts.copy()
                
    # Final Result
    if len(best_selected) < len(selected):
        selected = best_selected
        
    # Convert to list of lists
    result = [list(blocks[i]) for i in selected]
    return result
