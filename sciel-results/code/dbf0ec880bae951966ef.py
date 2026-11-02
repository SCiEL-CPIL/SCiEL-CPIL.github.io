import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    timeout = 110  # Safety margin
    
    # 1. Precomputation
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_id = {tsub: i for i, tsub in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute the list of t-subset IDs it covers
    block_tsubs_list = []
    for b in all_blocks:
        ids = [tsub_to_id[ts] for ts in itertools.combinations(b, t)]
        block_tsubs_list.append(ids)
    
    block_tsubs = np.array(block_tsubs_list, dtype=int)
    
    # Build reverse index: for each t-subset, which blocks cover it
    # tsub_blocks[tsub_id] = list of block indices covering this t-subset
    tsub_blocks = [[] for _ in range(n_tsubs)]
    for b_idx in range(n_blocks):
        for ts_id in block_tsubs[b_idx]:
            tsub_blocks[ts_id].append(b_idx)
    
    def greedy_cover(seed_order=None):
        """Greedy set cover with optional randomized tie-breaking."""
        uncovered = np.ones(n_tsubs, dtype=bool)
        chosen = []
        
        if seed_order is not None:
            # Create a priority order for tie-breaking
            tie_order = {b: i for i, b in enumerate(seed_order)}
        else:
            tie_order = None
        
        while np.any(uncovered):
            if time.time() - start_time > timeout:
                break
            
            scores = uncovered[block_tsubs].sum(axis=1)
            
            # Find all blocks with max score
            max_score = scores.max()
            if max_score == 0:
                break
            
            best_candidates = np.where(scores == max_score)[0]
            
            if tie_order is not None and len(best_candidates) > 1:
                # Pick the one with best (lowest) position in tie_order
                best_idx = min(best_candidates, key=lambda x: tie_order.get(x, float('inf')))
            else:
                best_idx = best_candidates[0]
            
            chosen.append(best_idx)
            uncovered[block_tsubs[best_idx]] = False
        
        return chosen
    
    def local_search_1exchange(chosen_indices):
        """Try 1-exchange: remove one block, add one block that covers all uniquely-covered t-subsets."""
        chosen_set = set(chosen_indices)
        cover_counts = np.zeros(n_tsubs, dtype=int)
        for idx in chosen_indices:
            cover_counts[block_tsubs[idx]] += 1
        
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            for idx in list(chosen_set):
                # What t-subsets does this block uniquely cover?
                unique_ts = block_tsubs[idx][cover_counts[block_tsubs[idx]] == 1]
                if len(unique_ts) == 0:
                    # Block is redundant, just remove it
                    chosen_set.discard(idx)
                    cover_counts[block_tsubs[idx]] -= 1
                    improved = True
                    continue
                
                # Look for a replacement block that covers all unique_ts
                # and is not already in chosen_set
                for cand in tsub_blocks[unique_ts[0]]:
                    if cand in chosen_set:
                        continue
                    # Check if cand covers all unique_ts
                    cand_tsubs = set(block_tsubs[cand].tolist())
                    if all(ts in cand_tsubs for ts in unique_ts):
                        # Also check that removing idx and adding cand doesn't leave any gap
                        # (it shouldn't, since unique_ts are covered by cand, and non-unique are covered by others)
                        chosen_set.discard(idx)
                        cover_counts[block_tsubs[idx]] -= 1
                        chosen_set.add(cand)
                        cover_counts[block_tsubs[cand]] += 1
                        improved = True
                        break
        
        return list(chosen_set)
    
    def local_search_2exchange(chosen_indices):
        """Try 2-exchange: remove two blocks, add one block that covers all uniquely-covered t-subsets."""
        chosen_set = set(chosen_indices)
        cover_counts = np.zeros(n_tsubs, dtype=int)
        for idx in chosen_indices:
            cover_counts[block_tsubs[idx]] += 1
        
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            chosen_list = list(chosen_set)
            for i in range(len(chosen_list)):
                for j in range(i + 1, len(chosen_list)):
                    if time.time() - start_time > timeout:
                        return list(chosen_set)
                    idx1, idx2 = chosen_list[i], chosen_list[j]
                    
                    # Temporarily remove both
                    cover_counts[block_tsubs[idx1]] -= 1
                    cover_counts[block_tsubs[idx2]] -= 1
                    
                    # Find t-subsets that are now uncovered
                    uncovered_ts = []
                    for ts_id in block_tsubs[idx1]:
                        if cover_counts[ts_id] == 0:
                            uncovered_ts.append(ts_id)
                    for ts_id in block_tsubs[idx2]:
                        if cover_counts[ts_id] == 0 and ts_id not in uncovered_ts:
                            uncovered_ts.append(ts_id)
                    
                    if len(uncovered_ts) > 0:
                        # Look for a single block that covers all uncovered_ts
                        candidates = tsub_blocks[uncovered_ts[0]]
                        for cand in candidates:
                            if cand in chosen_set:
                                continue
                            cand_tsubs = set(block_tsubs[cand].tolist())
                            if all(ts in cand_tsubs for ts in uncovered_ts):
                                # Valid 2->1 exchange
                                chosen_set.discard(idx1)
                                chosen_set.discard(idx2)
                                chosen_set.add(cand)
                                cover_counts[block_tsubs[idx1]] += 1
                                cover_counts[block_tsubs[idx2]] += 1
                                cover_counts[block_tsubs[cand]] += 1
                                improved = True
                                break
                        if improved:
                            break
                    else:
                        # Both blocks were redundant, just remove one
                        chosen_set.discard(idx1)
                        cover_counts[block_tsubs[idx1]] += 1
                        improved = True
                        break
                
                if improved:
                    break
        
        return list(chosen_set)
    
    # Phase 1: Initial greedy
    initial = greedy_cover()
    
    # Phase 2: Local search
    current = local_search_1exchange(initial)
    current = local_search_2exchange(current)
    
    # Phase 3: Multi-start with different random orderings
    best_count = len(current)
    best_solution = current
    
    remaining_time = timeout - (time.time() - start_time)
    if remaining_time > 10:
        n_restarts = min(20, int(remaining_time / 5))
        for r in range(n_restarts):
            if time.time() - start_time > timeout - 5:
                break
            
            # Random permutation of blocks for tie-breaking
            block_indices = list(range(n_blocks))
            random.shuffle(block_indices)
            
            candidate = greedy_cover(seed_order=block_indices)
            candidate = local_search_1exchange(candidate)
            candidate = local_search_2exchange(candidate)
            
            if len(candidate) < best_count:
                best_count = len(candidate)
                best_solution = candidate
    
    # Final cleanup: remove any remaining redundant blocks
    cover_counts = np.zeros(n_tsubs, dtype=int)
    for idx in best_solution:
        cover_counts[block_tsubs[idx]] += 1
    
    final_indices = []
    for idx in best_solution:
        if np.min(cover_counts[block_tsubs[idx]]) > 1:
            cover_counts[block_tsubs[idx]] -= 1
        else:
            final_indices.append(idx)
    
    result_blocks = [list(all_blocks[i]) for i in final_indices]
    
    return result_blocks
