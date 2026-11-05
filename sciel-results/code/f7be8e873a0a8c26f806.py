import itertools
import random
import time

def construct(v, k, t):
    # Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    id_to_subset = {i: s for s, i in subset_to_id.items()}
    
    # subsets_by_point[x] = list of subset IDs containing x
    subsets_by_point = [[] for _ in range(v)]
    for i, s in enumerate(all_subsets):
        for x in s:
            subsets_by_point[x].append(i)
            
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts
    # We have 120s. Let's estimate time per run.
    # If one run takes ~1s, we can do ~50 runs.
    # If one run takes ~10s, we can do ~5 runs.
    # Let's do a dynamic number of restarts or a fixed reasonable number like 20-50.
    
    def greedy_once(seed_val):
        random.seed(seed_val)
        uncovered = set(range(len(all_subsets)))
        blocks = []
        
        while uncovered:
            # Pick a starting point
            # Strategy: Pick a point involved in many uncovered subsets?
            # Or just pick the min element of a random uncovered subset?
            # Let's pick a random uncovered subset, and pick a random element from it.
            # This ensures we cover that subset.
            start_subset_id = random.choice(list(uncovered))
            start_subset = id_to_subset[start_subset_id]
            first_point = random.choice(start_subset)
            
            block = [first_point]
            block_set = {first_point}
            
            while len(block) < k:
                best_gain = -1
                best_candidates = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Calculate gain
                    gain = 0
                    # Only check subsets in uncovered that contain x
                    for sid in subsets_by_point[x]:
                        if sid not in uncovered:
                            continue
                        # Check if subset \ {x} is in block
                        # subset has t elements. x is one. t-1 others.
                        # If t-1 is small (3), this is fast.
                        s = id_to_subset[sid]
                        # Optimization: check if all elements in s except x are in block_set
                        # Since s is a tuple, we can iterate.
                        ok = True
                        for elem in s:
                            if elem != x and elem not in block_set:
                                ok = False
                                break
                        if ok:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if not best_candidates:
                    # Should not happen if uncovered is not empty and k>1
                    break
                
                # Random tie-breaking
                next_point = random.choice(best_candidates)
                block.append(next_point)
                block_set.add(next_point)
                
                # Update uncovered
                # Remove all subsets covered by the new block
                # A subset is covered if it is a subset of the current block
                # We only need to check subsets that contain the new point?
                # No, adding a point might cover subsets that don't contain the new point?
                # No, a subset T is covered by Block B if T is subset of B.
                # If T was not covered before, it means T was not subset of B_old.
                # If T is subset of B_new = B_old U {x}, then T must contain x?
                # Yes, if T does not contain x, then T subset of B_old, so it was already covered.
                # So we only need to check subsets containing x.
                
                to_remove = []
                for sid in subsets_by_point[next_point]:
                    if sid in uncovered:
                        s = id_to_subset[sid]
                        # Check if s is subset of block_set
                        # Since we know x is in s and x is in block_set,
                        # we just need to check if s \ {x} is subset of block_set
                        # But block_set contains x, so just check if s is subset of block_set
                        if all(elem in block_set for elem in s):
                            to_remove.append(sid)
                
                uncovered.difference_update(to_remove)
            
            block.sort()
            blocks.append(block)
        
        # Post-processing: Remove redundant blocks
        # Create a counter for how many times each subset is covered
        cover_count = [0] * len(all_subsets)
        block_subsets = []
        for b in blocks:
            b_set = set(b)
            subs_in_b = []
            for i, s in enumerate(all_subsets):
                if all(elem in b_set for elem in s):
                    cover_count[i] += 1
                    subs_in_b.append(i)
            block_subsets.append(subs_in_b)
            
        # Try to remove blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                # Check if block i is redundant
                # It is redundant if all its subsets are covered by other blocks
                # i.e., cover_count[sid] > 1 for all sid in block_subsets[i]
                redundant = True
                for sid in block_subsets[i]:
                    if cover_count[sid] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    for sid in block_subsets[i]:
                        cover_count[sid] -= 1
                    blocks.pop(i)
                    block_subsets.pop(i)
                    changed = True
                    break # Restart loop as indices changed
        
        return blocks

    # Run multiple times
    # Estimate time for one run
    start_time = time.time()
    # Do a few runs to estimate
    num_runs = 10
    for i in range(num_runs):
        blocks = greedy_once(i)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    elapsed = time.time() - start_time
    avg_time = elapsed / num_runs
    
    # Do more runs if time permits
    remaining_time = 110 - elapsed # Leave 10s buffer
    if remaining_time > 0:
        additional_runs = int(remaining_time / avg_time)
        for i in range(num_runs, num_runs + additional_runs):
            blocks = greedy_once(i)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
            
    return best_blocks
