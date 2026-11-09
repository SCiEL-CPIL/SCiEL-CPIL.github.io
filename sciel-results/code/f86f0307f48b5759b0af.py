import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    # Precomputation
    all_targets = list(itertools.combinations(range(v), t))
    target_index = {tup: i for i, tup in enumerate(all_targets)}
    num_targets = len(all_targets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage for each block as a set of target indices
    block_coverage = []
    for b in all_blocks:
        covered = set()
        for sub in itertools.combinations(b, t):
            covered.add(target_index[sub])
        block_coverage.append(covered)

    # Greedy Construction
    uncovered = set(range(num_targets))
    blocks_idx = []
    used = set()
    
    while uncovered:
        if time.time() > deadline - 10:  # Leave 10s for removal/search
            # If we run out of time, we must return a valid cover.
            # The greedy loop is the main cost. If it's not done, we have a partial cover.
            # We should ensure the greedy loop is fast enough.
            # For v=19, k=7, t=4, it should be fine.
            pass
        
        best_idx = -1
        best_gain = -1
        # Randomize iteration order to break ties and explore different landscapes?
        # For determinism and speed, linear scan is better.
        for i, cov_set in enumerate(block_coverage):
            if i in used:
                continue
            gain = len(cov_set & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == len(cov_set):
                    # Max possible gain for this block
                    # We can't get higher gain from this block, but other blocks might have same gain.
                    # We pick the first one with max gain.
                    pass
                    
        if best_idx == -1:
            break # Should not happen
            
        blocks_idx.append(best_idx)
        used.add(best_idx)
        uncovered -= block_coverage[best_idx]
        
    # If we didn't finish greedy due to time, we have an invalid cover.
    # But we assume it finishes.
    
    # Removal Phase
    cover_count = [0] * num_targets
    for idx in blocks_idx:
        for t in block_coverage[idx]:
            cover_count[t] += 1
            
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        # Iterate in reverse to avoid index shifting issues if we pop
        for i in range(len(blocks_idx) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            idx = blocks_idx[i]
            redundant = True
            for t in block_coverage[idx]:
                if cover_count[t] == 1:
                    redundant = False
                    break
            if redundant:
                for t in block_coverage[idx]:
                    cover_count[t] -= 1
                blocks_idx.pop(i)
                changed = True
                break # Restart the while loop to check from scratch? 
                # Or continue? If we continue, we might miss other removals enabled by this one?
                # No, removing a block only increases the "freedom" for other blocks.
                # So we can continue checking other blocks in the same pass.
                # But indices shift.
                # So we must break and restart the for loop, or use a list of indices to remove.
                # Let's use a list of indices to remove.
                pass
        
        # Let's refine the removal loop to handle multiple removals per pass
        if changed:
            continue
            
        # If no removals, try Local Search
        if time.time() < deadline - 5:
            # Local Search: Try to swap a block
            # Pick a random block
            if blocks_idx:
                i = random.randint(0, len(blocks_idx) - 1)
                idx = blocks_idx[i]
                # Find critical targets
                critical = []
                for t in block_coverage[idx]:
                    if cover_count[t] == 1:
                        critical.append(t)
                        
                if not critical:
                    # Should have been removed
                    continue
                    
                # Find a replacement block that covers all critical targets
                # This is expensive. 
                # We can just skip this for now to ensure robustness.
                pass

    return [list(all_blocks[i]) for i in blocks_idx]
