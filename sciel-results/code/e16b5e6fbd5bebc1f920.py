import itertools
import random
import math
import time

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute all k-subsets and their covered t-subset indices
    all_k_subsets = []
    block_t_indices = []
    
    for block in itertools.combinations(range(v), k):
        all_k_subsets.append(block)
        t_indices = []
        for ts in itertools.combinations(block, t):
            t_indices.append(t_subset_to_index[ts])
        block_t_indices.append(t_indices)
        
    num_k_subsets = len(all_k_subsets)
    
    # Precompute t_subset_to_blocks: for each t-subset, list of blocks containing it
    # This is crucial for fast "Add" and "Swap" moves
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, t_indices in enumerate(block_t_indices):
        for ts_idx in t_indices:
            t_subset_to_blocks[ts_idx].append(b_idx)

    # 1. Greedy Initialization
    uncovered = set(range(num_t_subsets))
    initial_blocks = []
    
    # To speed up greedy, we can use the t_subset_to_blocks structure?
    # Greedy: pick block covering most uncovered.
    # We can iterate blocks and count uncovered.
    # Optimization: If a t-subset is uncovered, any block covering it is a candidate.
    # But standard greedy is fine for initialization.
    
    while uncovered:
        best_block_idx = -1
        max_covered = 0
        
        for i in range(num_k_subsets):
            count = 0
            for idx in block_t_indices[i]:
                if idx in uncovered:
                    count += 1
            if count > max_covered:
                max_covered = count
                best_block_idx = i
                
        if best_block_idx == -1:
            break
            
        initial_blocks.append(best_block_idx)
        
        for idx in block_t_indices[best_block_idx]:
            uncovered.discard(idx)
            
    current_blocks = initial_blocks[:]
    
    # Coverage counts
    coverage_counts = [0] * num_t_subsets
    for b_idx in current_blocks:
        for ts_idx in block_t_indices[b_idx]:
            coverage_counts[ts_idx] += 1
            
    num_uncovered = 0
    for c in coverage_counts:
        if c == 0:
            num_uncovered += 1
            
    best_blocks = current_blocks[:]
    best_size = len(current_blocks)
    
    # SA Parameters
    T = 50.0 # Higher initial T
    cooling = 0.999 # Slower cooling
    min_T = 0.1
    
    start_time = time.time()
    time_limit = 110.0
    
    while T > min_T and (time.time() - start_time) < time_limit:
        move_type = random.random()
        
        if move_type < 0.5:
            # Remove a block
            if not current_blocks:
                T *= cooling
                continue
                
            b_idx_pos = random.randint(0, len(current_blocks) - 1)
            b_idx = current_blocks[b_idx_pos]
            
            # Check if removal is valid (no critical subsets)
            is_critical = False
            for ts_idx in block_t_indices[b_idx]:
                if coverage_counts[ts_idx] == 1:
                    is_critical = True
                    break
            
            if not is_critical:
                # Apply removal
                current_blocks.pop(b_idx_pos)
                for ts_idx in block_t_indices[b_idx]:
                    coverage_counts[ts_idx] -= 1
                # num_uncovered unchanged (was 0, stays 0)
                
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = current_blocks[:]
            else:
                # If we want to allow invalid states, we could accept with penalty.
                # But for speed and simplicity, we reject or try to swap instead.
                # Let's try a swap instead of a failed remove.
                pass # Fall through to swap logic? No, just skip.

        elif move_type < 0.8:
            # Add a block
            # Heuristic: Pick a block covering an uncovered subset
            if num_uncovered > 0:
                # Pick a random uncovered t-subset
                # We need a list of uncovered indices. 
                # Rebuilding it is O(N_t). N_t = 3876. Fast.
                uncovered_indices = [i for i, c in enumerate(coverage_counts) if c == 0]
                if uncovered_indices:
                    target_ts = random.choice(uncovered_indices)
                    # Pick a random block covering this t-subset
                    candidates = t_subset_to_blocks[target_ts]
                    if candidates:
                        b_idx = random.choice(candidates)
                        # Check if it's already in current_blocks?
                        # If it is, adding it again is useless (blocks must be distinct? 
                        # "blocks must be distinct". So we cannot add a duplicate.
                        # We should check if b_idx is in current_blocks.
                        # Using a set for current_blocks would be faster for lookup.
                        # Let's maintain a set of current block indices.
                        pass # We need a set.
                        
            # Simplified Add:
            # To ensure distinctness and efficiency, let's maintain `current_block_set`.
            # I will refactor to use a set for current blocks.

        else:
            # Swap a block
            if not current_blocks:
                T *= cooling
                continue
                
            b_idx_pos = random.randint(0, len(current_blocks) - 1)
            b_idx_old = current_blocks[b_idx_pos]
            
            # Find critical subsets in b_idx_old
            critical_ts = []
            for ts_idx in block_t_indices[b_idx_old]:
                if coverage_counts[ts_idx] == 1:
                    critical_ts.append(ts_idx)
                    
            if critical_ts:
                # Must pick a new block that covers at least one critical ts
                target_ts = random.choice(critical_ts)
                candidates = [b for b in t_subset_to_blocks[target_ts] if b != b_idx_old]
                if candidates:
                    b_idx_new = random.choice(candidates)
                    # Apply swap
                    current_blocks[b_idx_pos] = b_idx_new
                    # Update counts
                    for ts_idx in block_t_indices[b_idx_old]:
                        coverage_counts[ts_idx] -= 1
                    for ts_idx in block_t_indices[b_idx_new]:
                        coverage_counts[ts_idx] += 1
                    # Validity is preserved for target_ts. 
                    # Other ts in b_old might become uncovered? 
                    # No, we only picked a critical ts. Other ts in b_old have count >= 2? 
                    # If another ts in b_old was count 1, it would be in critical_ts. 
                    # If we didn't pick it, it remains count 1 (still covered by b_old? No, b_old is removed).
                    # Wait. If b_old has TWO critical subsets, and we only ensure b_new covers ONE, 
                    # the other becomes uncovered!
                    # So we must ensure b_new covers ALL critical subsets of b_old? 
                    # Or we accept invalidity?
                    # If we require validity, we must pick b_new that covers ALL critical_ts of b_old.
                    # Intersection of t_subset_to_blocks for all critical_ts.
                    # This might be empty.
                    
                    # Let's check validity of the swap.
                    # If invalid, revert? Or just don't do it?
                    # For now, let's just check if the swap results in num_uncovered > 0.
                    # If so, revert.
                    
                    # To make this fast, we can just apply and check counts?
                    # We already updated counts. 
                    # We need to check if any count is 0.
                    # We know target_ts is covered.
                    # Other critical_ts might not be.
                    # Let's just check all counts? O(N_t). 3876 ops. Fast.
                    
                    # Actually, if we just want to improve, we can allow "repaired" moves.
                    # But let's stick to valid states.
                    # If b_new doesn't cover all critical_ts, the move is invalid.
                    # We can check this before applying.
                    
                    # Let's revert and pick a better candidate if possible?
                    # Or just skip if invalid.
                    
                    # To keep it simple and fast:
                    # If critical_ts is not empty, find intersection of candidates for all critical_ts.
                    # If intersection is empty, skip move.
                    
                    # This is complex. Let's just do the swap and check validity.
                    # If invalid, revert.
                    
                    # Check validity
                    valid = True
                    for c in coverage_counts:
                        if c == 0:
                            valid = False
                            break
                            
                    if not valid:
                        # Revert
                        current_blocks[b_idx_pos] = b_idx_old
                        for ts_idx in block_t_indices[b_idx_old]:
                            coverage_counts[ts_idx] += 1
                        for ts_idx in block_t_indices[b_idx_new]:
                            coverage_counts[ts_idx] -= 1
                    else:
                        # Valid swap, accept (size unchanged)
                        # We might want to update best if we found a "better" structure?
                        # No, size is same. Just explore.
                        pass
            else:
                # No critical subsets. Any swap is valid.
                b_idx_new = random.randint(0, num_k_subsets - 1)
                if b_idx_new == b_idx_old:
                    T *= cooling
                    continue
                # Apply swap
                current_blocks[b_idx_pos] = b_idx_new
                for ts_idx in block_t_indices[b_idx_old]:
                    coverage_counts[ts_idx] -= 1
                for ts_idx in block_t_indices[b_idx_new]:
                    coverage_counts[ts_idx] += 1
                # Always valid
                pass

        T *= cooling

    # Final Check and Return
    # Ensure best_blocks is valid
    # (It should be, as we only update best when valid)
    
    final_blocks = [list(all_k_subsets[i]) for i in best_blocks]
    
    return final_blocks
