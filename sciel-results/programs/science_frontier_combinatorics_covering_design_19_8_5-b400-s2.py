import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block to subsets mapping
    block_to_subsets = [[] for _ in range(num_blocks)]
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            block_to_subsets[b_idx].append(subset_to_idx[s])
            
    # Precompute subset to blocks mapping
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx in range(num_blocks):
        for s_idx in block_to_subsets[b_idx]:
            subset_to_blocks[s_idx].append(b_idx)

    def get_greedy_solution():
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(bs) for bs in block_to_subsets], dtype=np.int32)
        selected = []
        # Use a simple greedy with random tie-breaking or just max
        # To add diversity, we can shuffle gains slightly or pick random among top k
        # Here we use a randomized greedy: pick from top 10% with some noise
        for _ in range(num_blocks): # Max blocks needed
            if not uncovered.any():
                break
            # Add small noise to break ties and introduce randomness
            noise = np.random.uniform(0, 1, num_blocks) * 0.1
            scores = gains + noise
            best_idx = np.argmax(scores)
            if gains[best_idx] == 0:
                break
            selected.append(best_idx)
            b_subs = block_to_subsets[best_idx]
            for s in b_subs:
                if uncovered[s]:
                    uncovered[s] = False
                    for b in subset_to_blocks[s]:
                        gains[b] -= 1
        return selected

    def is_valid_swap(current_set, old_block, new_block, coverage_counts):
        # Check if replacing old_block with new_block maintains validity
        # We only need to check subsets that were uniquely covered by old_block
        unique_subs = []
        for s in block_to_subsets[old_block]:
            if coverage_counts[s] == 1:
                unique_subs.append(s)
        
        if not unique_subs:
            return True
            
        # New block must cover all unique_subs
        new_subs_set = set(block_to_subsets[new_block])
        for s in unique_subs:
            if s not in new_subs_set:
                return False
        return True

    def perform_swap(current_set, old_idx, new_block, coverage_counts):
        # old_idx is index in current_set list
        old_block = current_set[old_idx]
        # Remove old
        for s in block_to_subsets[old_block]:
            coverage_counts[s] -= 1
        # Add new
        for s in block_to_subsets[new_block]:
            coverage_counts[s] += 1
        current_set[old_idx] = new_block

    def minimize_and_swap(start_time, time_limit):
        # Initial greedy
        current_set = get_greedy_solution()
        if not current_set:
            return []
            
        # Calculate initial coverage counts
        coverage_counts = np.zeros(num_subsets, dtype=np.int32)
        for b in current_set:
            for s in block_to_subsets[b]:
                coverage_counts[s] += 1
                
        best_set = list(current_set)
        best_len = len(current_set)
        
        # Precompute sets for faster "in set" checks if needed, 
        # but list membership is O(N). N is ~300. 
        # We can use a set for current blocks to check if new_block is already in set
        current_set_list = list(current_set)
        
        while time.time() - start_time < time_limit:
            # 1. Deletion Pass
            changed = True
            while changed:
                changed = False
                # Random order for deletion
                random.shuffle(current_set_list)
                for i in range(len(current_set_list)):
                    b = current_set_list[i]
                    # Check if redundant
                    redundant = True
                    for s in block_to_subsets[b]:
                        if coverage_counts[s] == 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove
                        for s in block_to_subsets[b]:
                            coverage_counts[s] -= 1
                        current_set_list.pop(i)
                        changed = True
                        break # Restart deletion pass
                    
            if len(current_set_list) < best_len:
                best_len = len(current_set_list)
                best_set = list(current_set_list)
                
            # 2. Swap Pass
            # Try to swap blocks to improve structure
            # We iterate through blocks and try to swap them
            # We limit swaps per iteration to avoid getting stuck
            swaps_done = 0
            max_swaps = 50
            
            # Shuffle order
            indices = list(range(len(current_set_list)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() - start_time > time_limit:
                    break
                if swaps_done >= max_swaps:
                    break
                    
                old_block = current_set_list[i]
                
                # Find unique subsets
                unique_subs = []
                for s in block_to_subsets[old_block]:
                    if coverage_counts[s] == 1:
                        unique_subs.append(s)
                
                if not unique_subs:
                    # Block is not critical for any subset.
                    # We can swap with any random block to diversify.
                    # Pick a random block from all_blocks
                    new_block = random.choice(all_blocks)
                    # Avoid swapping with itself or a block already in the set (unless we allow duplicates, which we don't)
                    if new_block == old_block:
                        continue
                    if new_block in current_set_list:
                        continue
                        
                    # Perform swap
                    perform_swap(current_set_list, i, new_block, coverage_counts)
                    swaps_done += 1
                    # Check if this swap allowed a deletion? 
                    # We don't check immediately, we let the next deletion pass handle it.
                    # But to be aggressive, we can check if the NEW block makes some OTHER block redundant?
                    # That's complex. Let's just do deletion pass after swaps.
                else:
                    # Block is critical for unique_subs.
                    # Find candidates that cover all unique_subs
                    # Intersection of subset_to_blocks[s] for s in unique_subs
                    candidates = None
                    for s in unique_subs:
                        if candidates is None:
                            candidates = set(subset_to_blocks[s])
                        else:
                            candidates.intersection_update(subset_to_blocks[s])
                            if not candidates:
                                break
                    
                    if candidates:
                        # Remove self from candidates? 
                        # If old_block is in candidates, swapping with itself is useless.
                        candidates.discard(old_block)
                        if candidates:
                            # Pick a random candidate
                            new_block = random.choice(list(candidates))
                            # Check if new_block is already in current_set_list
                            if new_block in current_set_list:
                                # If it's already there, swapping is a no-op in terms of set content, 
                                # but effectively changes which instance we have? No, blocks are unique.
                                # So if new_block is in current_set_list, we can't swap to it (duplicate).
                                # We need a block NOT in current_set_list.
                                # Filter candidates
                                valid_candidates = [c for c in candidates if c not in current_set_list]
                                if valid_candidates:
                                    new_block = random.choice(valid_candidates)
                                else:
                                    continue # No valid swap
                
                            if new_block != old_block and new_block not in current_set_list:
                                perform_swap(current_set_list, i, new_block, coverage_counts)
                                swaps_done += 1
                                # After a critical swap, it's good to restart deletion immediately
                                # because the structure changed significantly
                                break # Break inner loop, restart deletion pass
            
            # If we did swaps, run deletion pass again immediately
            if swaps_done > 0:
                # The loop continues, so it will hit the "1. Deletion Pass" at the top of the while loop
                pass

        return best_set

    start_time = time.time()
    # Run for 110 seconds
    result = minimize_and_swap(start_time, 110)
    
    # Convert indices to blocks
    return [list(all_blocks[i]) for i in result]
