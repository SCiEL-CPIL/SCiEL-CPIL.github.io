import itertools
import time
import numpy as np
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def elapsed():
        return time.time() - start_time

    # Step 1: Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build incidence structures
    block_to_tsets = [[] for _ in range(num_blocks)]
    tset_to_blocks = [[] for _ in range(num_tsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)
    
    # Convert to numpy arrays for faster operations
    # Create a sparse-like representation using offsets
    block_tsets_offsets = np.zeros(num_blocks + 1, dtype=np.int32)
    for i in range(num_blocks):
        block_tsets_offsets[i+1] = block_tsets_offsets[i] + len(block_to_tsets[i])
    
    block_tsets_flat = np.zeros(block_tsets_offsets[-1], dtype=np.int32)
    for i in range(num_blocks):
        offset = block_tsets_offsets[i]
        block_tsets_flat[offset:offset+len(block_to_tsets[i])] = block_to_tsets[i]
    
    tset_blocks_offsets = np.zeros(num_tsets + 1, dtype=np.int32)
    for i in range(num_tsets):
        tset_blocks_offsets[i+1] = tset_blocks_offsets[i] + len(tset_to_blocks[i])
    
    tset_blocks_flat = np.zeros(tset_blocks_offsets[-1], dtype=np.int32)
    for i in range(num_tsets):
        offset = tset_blocks_offsets[i]
        tset_blocks_flat[offset:offset+len(tset_to_blocks[i])] = tset_to_blocks[i]
    
    # Helper to get solution from block indices
    def get_solution(indices):
        return [list(all_blocks[i]) for i in indices]
    
    def compute_cover_counts(indices):
        counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in indices:
            offset = block_tsets_offsets[b_idx]
            end = block_tsets_offsets[b_idx+1]
            for ts_idx in block_tsets_flat[offset:end]:
                counts[ts_idx] += 1
        return counts
    
    # Greedy set cover with numpy acceleration
    def greedy_cover():
        uncovered = np.ones(num_tsets, dtype=np.bool_)
        uncovered_count = num_tsets
        selected = []
        selected_mask = np.zeros(num_blocks, dtype=np.bool_)
        
        # Initial weights: 1 / (number of blocks covering each tset)
        safe_counts = np.maximum(tset_blocks_offsets[1:] - tset_blocks_offsets[:-1], 1).astype(np.float64)
        t_weights = 1.0 / safe_counts
        
        # Gain for each block
        gain = np.zeros(num_blocks, dtype=np.float64)
        for i in range(num_blocks):
            offset = block_tsets_offsets[i]
            end = block_tsets_offsets[i+1]
            if end > offset:
                gain[i] = np.sum(t_weights[block_tsets_flat[offset:end]])
        
        while uncovered_count > 0 and elapsed() < deadline - 20:
            # Find best block using numpy argmax
            masked_gain = gain.copy()
            masked_gain[selected_mask] = -1
            best_idx = int(np.argmax(masked_gain))
            
            if masked_gain[best_idx] <= 0:
                break
            
            selected.append(best_idx)
            selected_mask[best_idx] = True
            
            # Update uncovered and gains
            offset = block_tsets_offsets[best_idx]
            end = block_tsets_offsets[best_idx+1]
            for ts_idx in block_tsets_flat[offset:end]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_count -= 1
                    w = t_weights[ts_idx]
                    # Update gains for all blocks covering this tset
                    t_offset = tset_blocks_offsets[ts_idx]
                    t_end = tset_blocks_offsets[ts_idx+1]
                    for b in tset_blocks_flat[t_offset:t_end]:
                        if not selected_mask[b]:
                            gain[b] -= w
        
        return selected
    
    # Local search with multiple strategies
    def local_search(initial_indices, max_time=15.0):
        current = list(initial_indices)
        if not current:
            return []
        
        cover_counts = compute_cover_counts(current)
        best_size = len(current)
        best_solution = list(current)
        
        local_start = time.time()
        no_improvement_rounds = 0
        max_no_improvement = 3
        
        while (time.time() - local_start) < max_time and no_improvement_rounds < max_no_improvement:
            improved = False
            
            # Phase 1: Remove redundant blocks
            while True:
                if (time.time() - local_start) > max_time:
                    break
                found_redundant = False
                for i in range(len(current) - 1, -1, -1):
                    if (time.time() - local_start) > max_time:
                        break
                    b_idx = current[i]
                    is_redundant = True
                    offset = block_tsets_offsets[b_idx]
                    end = block_tsets_offsets[b_idx+1]
                    for ts_idx in block_tsets_flat[offset:end]:
                        if cover_counts[ts_idx] == 1:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        for ts_idx in block_tsets_flat[offset:end]:
                            cover_counts[ts_idx] -= 1
                        current.pop(i)
                        found_redundant = True
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                
                if not found_redundant or (time.time() - local_start) > max_time:
                    break
            
            if not current:
                break
                
            # Phase 2: 1-for-1 Swaps
            for strategy in range(3):
                if (time.time() - local_start) > max_time:
                    break
                
                random.shuffle(current)
                
                for i in range(len(current)):
                    if (time.time() - local_start) > max_time:
                        break
                    
                    b_out = current[i]
                    
                    # Find tsets uniquely covered by b_out
                    unique_tsets = []
                    offset = block_tsets_offsets[b_out]
                    end = block_tsets_offsets[b_out+1]
                    for ts_idx in block_tsets_flat[offset:end]:
                        if cover_counts[ts_idx] == 1:
                            unique_tsets.append(ts_idx)
                            
                    if not unique_tsets:
                        continue

                    # Find candidate blocks that cover all unique_tsets
                    candidates = set(tset_to_blocks[unique_tsets[0]])
                    candidates.discard(b_out)
                    
                    for ts_idx in unique_tsets[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(tset_to_blocks[ts_idx])
                    
                    if not candidates:
                        continue
                    
                    b_out_tsets_set = set(block_to_tsets[b_out])
                    
                    candidate_list = list(candidates)
                    limit = [100, 200, 400][strategy]
                    if len(candidate_list) > limit:
                        random.shuffle(candidate_list)
                        candidate_list = candidate_list[:limit]
                    
                    best_candidate = -1
                    best_score = -1
                    
                    for b_in in candidate_list:
                        score = 0
                        in_offset = block_tsets_offsets[b_in]
                        in_end = block_tsets_offsets[b_in+1]
                        for ts_idx in block_tsets_flat[in_offset:in_end]:
                            if ts_idx not in b_out_tsets_set:
                                if strategy == 0:
                                    score += 1
                                elif strategy == 1:
                                    if cover_counts[ts_idx] == 1:
                                        score += 15
                                    elif cover_counts[ts_idx] == 2:
                                        score += 5
                                    else:
                                        score += 1
                                else:
                                    if cover_counts[ts_idx] == 1:
                                        score += 20
                                    elif cover_counts[ts_idx] == 2:
                                        score += 8
                                    elif cover_counts[ts_idx] == 3:
                                        score += 3
                                    else:
                                        score += 1
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = b_in
                    
                    if best_candidate != -1:
                        # Perform the swap
                        out_offset = block_tsets_offsets[b_out]
                        out_end = block_tsets_offsets[b_out+1]
                        for ts_idx in block_tsets_flat[out_offset:out_end]:
                            cover_counts[ts_idx] -= 1
                        
                        in_offset = block_tsets_offsets[best_candidate]
                        in_end = block_tsets_offsets[best_candidate+1]
                        for ts_idx in block_tsets_flat[in_offset:in_end]:
                            cover_counts[ts_idx] += 1
                        
                        current[i] = best_candidate
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
            
            if not improved:
                no_improvement_rounds += 1
            else:
                no_improvement_rounds = 0
        
        return best_solution
    
    # Main algorithm
    # Start with greedy cover
    selected = greedy_cover()
    
    # Apply local search
    if elapsed() < deadline - 10:
        selected = local_search(selected, max_time=15.0)
    
    # Ensure validity
    cover_counts = compute_cover_counts(selected)
    if not np.all(cover_counts > 0):
        # Add blocks to cover remaining tsets
        uncovered = cover_counts == 0
        uncovered_indices = np.where(uncovered)[0]
        
        for ts_idx in uncovered_indices:
            if elapsed() > deadline - 5:
                break
            # Find a block covering this tset
            t_offset = tset_blocks_offsets[ts_idx]
            t_end = tset_blocks_offsets[ts_idx+1]
            for b in tset_blocks_flat[t_offset:t_end]:
                if b not in selected:
                    selected.append(b)
                    offset = block_tsets_offsets[b]
                    end = block_tsets_offsets[b+1]
                    for ts in block_tsets_flat[offset:end]:
                        cover_counts[ts] += 1
                    break
    
    return get_solution(selected)
