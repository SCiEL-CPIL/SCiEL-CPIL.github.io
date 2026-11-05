import itertools
import time
import numpy as np
import scipy.sparse
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    def time_ok():
        return time.time() < deadline

    # 1. Generate all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # 2. Generate all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_blocks)

    # 3. Build Sparse Matrix M (n_blocks x n_t) and auxiliary structures
    row_indices = []
    col_indices = []
    data = []
    
    # block_to_subsets: list of lists of t-subset indices for each block
    block_to_subsets = [[] for _ in range(n_blocks)]
    # subset_to_blocks: list of lists of block indices for each t-subset
    subset_to_blocks = [[] for _ in range(n_t)]
    
    for b_idx, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            s_idx = t_sub_to_idx.get(t_sub)
            if s_idx is not None:
                row_indices.append(b_idx)
                col_indices.append(s_idx)
                data.append(1)
                block_to_subsets[b_idx].append(s_idx)
                subset_to_blocks[s_idx].append(b_idx)
                
    M = scipy.sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_blocks, n_t), dtype=np.int32)
    
    def greedy_construct(randomize=False, noise_scale=0.0, use_weighted=False):
        uncovered = np.ones(n_t, dtype=np.int32)
        active_blocks = []
        n_uncovered = n_t
        
        # Maintain incremental gains
        gains = np.zeros(n_blocks, dtype=np.int32)
        for b_idx in range(n_blocks):
            start = M.indptr[b_idx]
            end = M.indptr[b_idx + 1]
            if start != end:
                gains[b_idx] = end - start  # all subsets initially uncovered
        
        # Weighted gains for tie-breaking
        if use_weighted:
            weights = np.ones(n_blocks, dtype=np.float64)
        
        while n_uncovered > 0:
            if not time_ok():
                break
            
            if use_weighted:
                # Weighted gain: gain / log(weight) to prefer blocks that haven't been "used" much
                # Actually, standard approach: gain / (1 + count_of_times_considered)
                # Simpler: just add small random noise for diversity
                if randomize and noise_scale > 0:
                    noise = np.random.uniform(0, 1, n_blocks) * noise_scale
                    weighted_gains = gains.astype(np.float64) / (1.0 + weights) + noise
                else:
                    weighted_gains = gains.astype(np.float64) / (1.0 + weights)
                best_b = int(np.argmax(weighted_gains))
            else:
                if randomize and noise_scale > 0:
                    noise = np.random.uniform(0, 1, n_blocks) * noise_scale
                    gains_with_noise = gains.astype(np.float64) + noise
                    best_b = int(np.argmax(gains_with_noise))
                else:
                    best_b = int(np.argmax(gains))
            
            if gains[best_b] == 0:
                break
            
            active_blocks.append(best_b)
            if use_weighted:
                weights[best_b] += 1.0
            
            # Update uncovered and gains incrementally
            start = M.indptr[best_b]
            end = M.indptr[best_b + 1]
            cols = M.indices[start:end]
            
            for s_idx in cols:
                if uncovered[s_idx] == 1:
                    uncovered[s_idx] = 0
                    n_uncovered -= 1
                    # This t-subset is now covered, so any block that covered it loses 1 gain
                    for b_idx in subset_to_blocks[s_idx]:
                        gains[b_idx] -= 1
        
        return active_blocks

    def redundancy_removal(active_blocks, time_limit=5.0):
        # active_blocks is a list of block indices
        if not active_blocks:
            return []
            
        current_set = list(active_blocks)
        
        # Compute coverage counts
        counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_set:
            start = M.indptr[b_idx]
            end = M.indptr[b_idx + 1]
            cols = M.indices[start:end]
            counts[cols] += 1
        
        # To allow multiple passes within time limit
        end_time = time.time() + time_limit
        improved = True
        
        while improved and time.time() < end_time:
            if len(current_set) == 0:
                break
            
            remove_mask = np.zeros(len(current_set), dtype=bool)
            
            # Iterate through blocks to check redundancy
            for i in range(len(current_set)):
                b_idx = current_set[i]
                start = M.indptr[b_idx]
                end = M.indptr[b_idx + 1]
                if start == end:
                    remove_mask[i] = True
                    continue
                
                cols = M.indices[start:end]
                # If min count > 1, block is redundant
                if np.min(counts[cols]) > 1:
                    remove_mask[i] = True
            
            if not remove_mask.any():
                break
            
            # Remove redundant blocks and update counts
            for i in range(len(current_set)):
                if remove_mask[i]:
                    b_idx = current_set[i]
                    start = M.indptr[b_idx]
                    end = M.indptr[b_idx + 1]
                    cols = M.indices[start:end]
                    counts[cols] -= 1
            
            current_set = [b for i, b in enumerate(current_set) if not remove_mask[i]]
            improved = True
            
        return current_set

    def swap_local_search(active_blocks, time_limit=10.0):
        # active_blocks is a list of block indices
        if not active_blocks:
            return []
            
        current_set = list(active_blocks)
        current_set_set = set(current_set)
        
        # Compute coverage counts
        coverage_counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_set:
            start = M.indptr[b_idx]
            end = M.indptr[b_idx + 1]
            cols = M.indices[start:end]
            coverage_counts[cols] += 1
        
        best_set = list(current_set)
        best_len = len(current_set)
        
        end_time = time.time() + time_limit
        no_improve_passes = 0
        max_no_improve = 3
        
        while time.time() < end_time and no_improve_passes < max_no_improve:
            changed = False
            # Shuffle order for diversity
            indices = list(range(len(current_set)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > end_time:
                    break
                
                old_block = current_set[i]
                
                # Find unique subsets covered by old_block
                unique_subs = []
                start = M.indptr[old_block]
                end = M.indptr[old_block + 1]
                if start != end:
                    cols = M.indices[start:end]
                    for s in cols:
                        if coverage_counts[s] == 1:
                            unique_subs.append(s)
                
                if not unique_subs:
                    # Block is redundant - try to replace with a block that helps others
                    # Score candidates by how many "critical" subsets they cover
                    new_block = None
                    best_score = -1
                    
                    # Try blocks from subset_to_blocks of subsets that are covered exactly once
                    # by other blocks (not old_block)
                    candidate_blocks = set()
                    for s in range(n_t):
                        if coverage_counts[s] == 1:
                            # This subset is covered by exactly one block
                            # Find which block
                            for b in subset_to_blocks[s]:
                                if b != old_block:
                                    candidate_blocks.add(b)
                                    break
                    
                    if not candidate_blocks:
                        # Just try random
                        for _ in range(3):
                            cand = random.randint(0, n_blocks - 1)
                            if cand not in current_set_set and cand != old_block:
                                new_block = cand
                                break
                    else:
                        # Score each candidate by how many subsets it covers that are
                        # currently covered exactly once (excluding those it would make double)
                        for cand in list(candidate_blocks)[:20]:  # Limit candidates
                            if cand in current_set_set or cand == old_block:
                                continue
                            # Calculate score: number of subsets where coverage would go from 1 to 2
                            # (bad) vs 0 to 1 (good, but we're swapping so 0->1 doesn't happen)
                            # Actually for a swap: old_block removed, new_block added
                            # Subsets in old_block: count decreases by 1
                            # Subsets in new_block: count increases by 1
                            # We want to minimize the number of subsets where count goes to 0
                            # (shouldn't happen if we check unique_subs) and maximize
                            # subsets where count goes from 2 to 3 (good for future removals)
                            
                            new_cols = M.indices[M.indptr[cand]:M.indptr[cand + 1]]
                            score = 0
                            for s in new_cols:
                                if coverage_counts[s] >= 2:
                                    score += 1  # Good: becomes more covered
                                elif coverage_counts[s] == 1:
                                    score -= 2  # Bad: was uniquely covered, now double
                            
                            if score > best_score:
                                best_score = score
                                new_block = cand
                    
                    if new_block is None:
                        continue
                    
                    # Perform swap
                    old_cols = M.indices[M.indptr[old_block]:M.indptr[old_block + 1]]
                    new_cols = M.indices[M.indptr[new_block]:M.indptr[new_block + 1]]
                    
                    coverage_counts[old_cols] -= 1
                    coverage_counts[new_cols] += 1
                    
                    current_set[i] = new_block
                    current_set_set.remove(old_block)
                    current_set_set.add(new_block)
                    
                else:
                    # Block is critical for unique_subs
                    # Find candidates that cover all unique_subs
                    candidates = None
                    for s in unique_subs:
                        if candidates is None:
                            candidates = set(subset_to_blocks[s])
                        else:
                            candidates.intersection_update(subset_to_blocks[s])
                            if not candidates:
                                break
                                
                    if not candidates:
                        continue
                    
                    candidates.discard(old_block)
                    if not candidates:
                        continue
                    
                    # Score candidates by how many additional subsets they cover well
                    best_cand = None
                    best_score = -float('inf')
                    
                    for cand in list(candidates)[:30]:  # Limit candidates
                        if cand in current_set_set:
                            continue
                        
                        new_cols = M.indices[M.indptr[cand]:M.indptr[cand + 1]]
                        score = 0
                        for s in new_cols:
                            if coverage_counts[s] >= 2:
                                score += 1
                            elif coverage_counts[s] == 1:
                                score -= 2
                        
                        if score > best_score:
                            best_score = score
                            best_cand = cand
                    
                    if best_cand is None:
                        continue
                    
                    # Perform swap
                    old_cols = M.indices[M.indptr[old_block]:M.indptr[old_block + 1]]
                    new_cols = M.indices[M.indptr[best_cand]:M.indptr[best_cand + 1]]
                    
                    coverage_counts[old_cols] -= 1
                    coverage_counts[new_cols] += 1
                    
                    current_set[i] = best_cand
                    current_set_set.remove(old_block)
                    current_set_set.add(best_cand)
            
            # Deletion pass after swaps
            if time.time() < end_time:
                remove_mask = np.zeros(len(current_set), dtype=bool)
                for i in range(len(current_set)):
                    b_idx = current_set[i]
                    start = M.indptr[b_idx]
                    end = M.indptr[b_idx + 1]
                    if start == end:
                        remove_mask[i] = True
                        continue
                    cols = M.indices[start:end]
                    if np.min(coverage_counts[cols]) > 1:
                        remove_mask[i] = True
                        
                if remove_mask.any():
                    for i in range(len(current_set)):
                        if remove_mask[i]:
                            b_idx = current_set[i]
                            start = M.indptr[b_idx]
                            end = M.indptr[b_idx + 1]
                            cols = M.indices[start:end]
                            coverage_counts[cols] -= 1
                            current_set_set.remove(b_idx)
                    
                    current_set = [b for i, b in enumerate(current_set) if not remove_mask[i]]
                    
                    if len(current_set) < best_len:
                        best_len = len(current_set)
                        best_set = list(current_set)
                        changed = True
                        no_improve_passes = 0
                    else:
                        no_improve_passes += 1
                else:
                    no_improve_passes += 1
            
            if not changed:
                no_improve_passes += 1
            
        return best_set

    # Strategy:
    # 1. Run standard greedy (fast baseline)
    # 2. Run weighted greedy
    # 3. Run randomized greedy with local search
    # 4. Return best result

    best_result = []
    best_len = float('inf')

    # Phase 1: Standard Greedy (fast)
    if time_ok():
        g1 = greedy_construct(randomize=False, use_weighted=False)
        r1 = redundancy_removal(g1, time_limit=5.0)
        if len(r1) < best_len:
            best_len = len(r1)
            best_result = r1

    # Phase 2: Weighted Greedy
    if time_ok():
        g2 = greedy_construct(randomize=False, use_weighted=True)
        r2 = redundancy_removal(g2, time_limit=5.0)
        if len(r2) < best_len:
            best_len = len(r2)
            best_result = r2

    # Phase 3: Randomized Greedy + Local Search
    iterations = 0
    while time.time() < deadline - 15.0:
        iterations += 1
        # Randomized greedy with varying noise
        noise = 0.3 + 0.3 * (iterations % 3) / 3.0
        g = greedy_construct(randomize=True, noise_scale=noise, use_weighted=True)
        
        # Redundancy removal
        r = redundancy_removal(g, time_limit=3.0)
        
        if len(r) < best_len:
            best_len = len(r)
            best_result = r
        
        # Swap local search
        remaining = deadline - time.time()
        if remaining > 8.0:
            s = swap_local_search(r, time_limit=remaining - 8.0)
            if len(s) < best_len:
                best_len = len(s)
                best_result = s

    # Final redundancy check on best_result
    if best_result and time.time() < deadline - 2.0:
        final_r = redundancy_removal(best_result, time_limit=2.0)
        if len(final_r) < best_len:
            best_result = final_r
            best_len = len(final_r)

    # Construct output
    result = [list(all_k_blocks[b]) for b in best_result]
    return result
