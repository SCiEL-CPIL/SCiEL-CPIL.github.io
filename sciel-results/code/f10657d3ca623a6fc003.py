import numpy as np
import itertools
import time
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Data structures
    # block_tsubs[b] = list of t-subset indices covered by block b
    block_tsubs = [[] for _ in range(n_blocks)]
    # tsub_blocks[ts] = list of block indices covering t-subset ts
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    # Bitmasks for fast set operations
    # tsub_mask[b] has bits set for t-subsets covered by block b
    tsub_mask = [0] * n_blocks
    # tsub_blocks_mask[ts] has bits set for blocks covering t-subset ts
    tsub_blocks_mask = [0] * n_tsubsets
    
    # For local search, we might need sets of blocks for quick intersection
    # But n_blocks is ~50k, so bitmasks are better.
    # However, for finding candidates for a specific set of t-subsets,
    # we can intersect the tsub_blocks lists if the set is small.
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks[idx].append(b_idx)
            block_tsubs[b_idx].append(idx)
        tsub_mask[b_idx] = mask
        tsub_blocks_mask[idx] |= (1 << b_idx) # Note: idx here is the last one, need to update for all
    
    # Correcting tsub_blocks_mask construction
    tsub_blocks_mask = [0] * n_tsubsets
    for b_idx in range(n_blocks):
        for ts_idx in block_tsubs[b_idx]:
            tsub_blocks_mask[ts_idx] |= (1 << b_idx)

    full_tsub_mask = (1 << n_tsubsets) - 1
    
    def greedy_cover(seed):
        """
        Greedy set cover with heap-based lazy updates.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        selected_mask = 0  # Bitmask of selected blocks
        
        # Initial gains: number of tsubs covered by each block
        gains = [0] * n_blocks
        for b_idx in range(n_blocks):
            gains[b_idx] = tsub_mask[b_idx].bit_count()
            
        # Heap stores (-gain, block_idx)
        heap = [(-gains[b_idx], b_idx) for b_idx in range(n_blocks)]
        heapq.heapify(heap)
        
        while uncovered_mask:
            if time.time() > deadline - 5.0:
                break

            best_block = -1
            best_gain = 0

            # Pop stale entries
            while heap:
                neg_g, b_idx = heapq.heappop(heap)
                g = -neg_g
                # Check if this entry is still valid
                # 1. Not already selected
                # 2. Gain matches current calculated gain (we don't maintain exact gains in a separate array easily with lazy updates, 
                #    but we can re-calculate gain for a block if needed, or trust the heap if we push updates correctly)
                # Actually, standard lazy heap approach: 
                # When we select a block, we update the gains of its neighbors and push new entries.
                # We need to check if the popped entry's gain is still the "current" gain.
                # Since we don't store "current gain" in an array that is always updated, we can just re-calculate the gain 
                # of the top candidate to verify, or store the version.
                # Simpler: Store current gain in an array, update it, and push.
                # When popping, if the popped gain != current_gain[b_idx], it's stale.
                
                if gains[b_idx] != g:
                    continue
                if selected_mask & (1 << b_idx):
                    continue
                
                best_block = b_idx
                best_gain = g
                break

            if best_block == -1:
                break

            selected.append(best_block)
            selected_mask |= (1 << best_block)
            gains[best_block] = -1 # Mark as selected, effectively removing it
            
            # Update gains for blocks covering newly covered tsubs
            new_covered = tsub_mask[best_block] & uncovered_mask
            uncovered_mask &= ~tsub_mask[best_block]
            
            m = new_covered
            while m:
                lsb = m & -m
                ts_idx = lsb.bit_length() - 1
                # For every block covering this ts_idx, decrease its gain
                cm = tsub_blocks_mask[ts_idx]
                while cm:
                    lsb2 = cm & -cm
                    b_idx2 = lsb2.bit_length() - 1
                    if not (selected_mask & (1 << b_idx2)) and gains[b_idx2] > 0:
                        gains[b_idx2] -= 1
                        heapq.heappush(heap, (-gains[b_idx2], b_idx2))
                    cm ^= lsb2
                m ^= lsb

        return selected

    def local_search_1exchange(chosen_indices, seed):
        """
        Try 1-exchange: remove one block, add one block that covers all uniquely-covered t-subsets.
        """
        rng = random.Random(seed)
        chosen_set = set(chosen_indices)
        if not chosen_set:
            return []
            
        # cover_counts[ts] = number of blocks in chosen_set covering ts
        cover_counts = np.zeros(n_tsubsets, dtype=np.int32)
        for idx in chosen_set:
            for ts_id in block_tsubs[idx]:
                cover_counts[ts_id] += 1
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Shuffle to avoid bias
            current_list = list(chosen_set)
            rng.shuffle(current_list)
            
            for idx in current_list:
                if idx not in chosen_set:
                    continue
                
                # Find t-subsets uniquely covered by this block
                unique_ts_mask = 0
                unique_ts_list = []
                for ts_id in block_tsubs[idx]:
                    if cover_counts[ts_id] == 1:
                        unique_ts_mask |= (1 << ts_id)
                        unique_ts_list.append(ts_id)
                
                if not unique_ts_list:
                    # Block is redundant, remove it
                    chosen_set.discard(idx)
                    for ts_id in block_tsubs[idx]:
                        cover_counts[ts_id] -= 1
                    improved = True
                    break
                
                # Find candidates that cover all unique_ts_list
                # Intersect tsub_blocks for each ts in unique_ts_list
                # Start with the one with fewest blocks
                min_ts = min(unique_ts_list, key=lambda x: tsub_blocks_mask[x].bit_count())
                candidates_mask = tsub_blocks_mask[min_ts]
                
                for ts_id in unique_ts_list[1:]:
                    candidates_mask &= tsub_blocks_mask[ts_id]
                    if candidates_mask == 0:
                        break
                
                if candidates_mask == 0:
                    continue
                    
                candidates_mask &= ~(1 << idx) # Exclude self
                
                if candidates_mask == 0:
                    continue
                
                # Pick a random candidate from the set
                # Extract indices from mask
                cand_list = []
                m = candidates_mask
                while m:
                    lsb = m & -m
                    idx_cand = lsb.bit_length() - 1
                    cand_list.append(idx_cand)
                    m ^= lsb
                
                if not cand_list:
                    continue
                    
                rng.shuffle(cand_list)
                
                # Try the first few candidates
                for cand in cand_list[:5]:
                    # Apply swap
                    chosen_set.discard(idx)
                    for ts_id in block_tsubs[idx]:
                        cover_counts[ts_id] -= 1
                    
                    chosen_set.add(cand)
                    for ts_id in block_tsubs[cand]:
                        cover_counts[ts_id] += 1
                    
                    improved = True
                    break
                
                if improved:
                    break
        
        return list(chosen_set)

    def local_search_2exchange(chosen_indices, seed):
        """
        Try 2-exchange: remove two blocks, add one block that covers all uniquely-covered t-subsets.
        """
        rng = random.Random(seed)
        chosen_set = set(chosen_indices)
        if len(chosen_set) < 2:
            return list(chosen_set)
            
        cover_counts = np.zeros(n_tsubsets, dtype=np.int32)
        for idx in chosen_set:
            for ts_id in block_tsubs[idx]:
                cover_counts[ts_id] += 1
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_list = list(chosen_set)
            rng.shuffle(current_list)
            
            # Sample pairs to avoid O(N^2) if N is large, but N is ~150, so O(N^2) is fine
            for i in range(len(current_list)):
                for j in range(i + 1, len(current_list)):
                    idx1 = current_list[i]
                    idx2 = current_list[j]
                    if idx1 not in chosen_set or idx2 not in chosen_set:
                        continue
                    
                    # Temporarily remove both
                    for ts_id in block_tsubs[idx1]:
                        cover_counts[ts_id] -= 1
                    for ts_id in block_tsubs[idx2]:
                        cover_counts[ts_id] -= 1
                    
                    # Find t-subsets that are now uncovered (count 0)
                    uncovered_mask = 0
                    uncovered_list = []
                    
                    # Check tsubs of idx1
                    for ts_id in block_tsubs[idx1]:
                        if cover_counts[ts_id] == 0:
                            uncovered_mask |= (1 << ts_id)
                            uncovered_list.append(ts_id)
                    
                    # Check tsubs of idx2
                    for ts_id in block_tsubs[idx2]:
                        if cover_counts[ts_id] == 0 and not (uncovered_mask & (1 << ts_id)):
                            uncovered_mask |= (1 << ts_id)
                            uncovered_list.append(ts_id)
                    
                    if not uncovered_list:
                        # Both blocks were redundant (or one was, and the other became redundant)
                        # Just remove one
                        chosen_set.discard(idx1)
                        # Restore counts for idx1? No, we want to remove it.
                        # We subtracted counts for both.
                        # If we remove idx1, we need to restore counts for idx2?
                        # Wait, the logic is: we removed both. If no tsub is uncovered, 
                        # it means every tsub covered by idx1 or idx2 is covered by some other block.
                        # So we can safely remove both? No, we only add 1 block.
                        # If we remove both and add 1, the net change is -1 block.
                        # If uncovered_list is empty, it means we can remove BOTH without adding anything?
                        # No, the exchange is 2 -> 1.
                        # If uncovered_list is empty, it means the union of tsubs covered by idx1 and idx2
                        # is fully covered by the rest of the set.
                        # So we can remove idx1 and idx2 and add NO block? That would be a 2->0 exchange.
                        # But we must maintain a covering.
                        # If uncovered_list is empty, we can just remove one block and keep the other?
                        # Or remove both and add a block that covers nothing new?
                        # Actually, if uncovered_list is empty, we can remove idx1, and idx2 is still there.
                        # Let's just remove idx1.
                        chosen_set.discard(idx1)
                        # We subtracted counts for idx1 and idx2.
                        # We need to restore counts for idx2 because it stays.
                        for ts_id in block_tsubs[idx2]:
                            cover_counts[ts_id] += 1
                        # Counts for idx1 remain subtracted (removed).
                        improved = True
                        break
                    
                    # Look for a single block that covers all uncovered_list
                    min_ts = min(uncovered_list, key=lambda x: tsub_blocks_mask[x].bit_count())
                    candidates_mask = tsub_blocks_mask[min_ts]
                    
                    for ts_id in uncovered_list[1:]:
                        candidates_mask &= tsub_blocks_mask[ts_id]
                        if candidates_mask == 0:
                            break
                    
                    if candidates_mask == 0:
                        # Restore counts
                        for ts_id in block_tsubs[idx1]:
                            cover_counts[ts_id] += 1
                        for ts_id in block_tsubs[idx2]:
                            cover_counts[ts_id] += 1
                        continue
                    
                    # Exclude idx1 and idx2
                    candidates_mask &= ~(1 << idx1)
                    candidates_mask &= ~(1 << idx2)
                    
                    if candidates_mask == 0:
                        # Restore counts
                        for ts_id in block_tsubs[idx1]:
                            cover_counts[ts_id] += 1
                        for ts_id in block_tsubs[idx2]:
                            cover_counts[ts_id] += 1
                        continue
                    
                    # Pick a candidate
                    cand_list = []
                    m = candidates_mask
                    while m:
                        lsb = m & -m
                        idx_cand = lsb.bit_length() - 1
                        cand_list.append(idx_cand)
                        m ^= lsb
                    
                    if not cand_list:
                        # Restore counts
                        for ts_id in block_tsubs[idx1]:
                            cover_counts[ts_id] += 1
                        for ts_id in block_tsubs[idx2]:
                            cover_counts[ts_id] += 1
                        continue
                        
                    rng.shuffle(cand_list)
                    cand = cand_list[0]
                    
                    # Apply 2->1 exchange
                    chosen_set.discard(idx1)
                    chosen_set.discard(idx2)
                    chosen_set.add(cand)
                    
                    # Restore counts for idx1, idx2 (they were subtracted)
                    for ts_id in block_tsubs[idx1]:
                        cover_counts[ts_id] += 1
                    for ts_id in block_tsubs[idx2]:
                        cover_counts[ts_id] += 1
                    # Add counts for cand
                    for ts_id in block_tsubs[cand]:
                        cover_counts[ts_id] += 1
                    
                    improved = True
                    break
                
                if improved:
                    break
        
        return list(chosen_set)

    # Phase 1: Initial greedy
    initial = greedy_cover(42)
    
    # Phase 2: Local search
    current = local_search_1exchange(initial, 42)
    current = local_search_2exchange(current, 42)
    
    # Phase 3: Multi-start with different random seeds
    best_count = len(current)
    best_solution = current
    
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 5:
        n_restarts = min(30, int(remaining_time / 2))
        for r in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            seed = 1000 + r
            candidate = greedy_cover(seed)
            candidate = local_search_1exchange(candidate, seed)
            candidate = local_search_2exchange(candidate, seed)
            
            if len(candidate) < best_count:
                best_count = len(candidate)
                best_solution = candidate
    
    # Final cleanup: remove any remaining redundant blocks
    cover_counts = np.zeros(n_tsubsets, dtype=np.int32)
    for idx in best_solution:
        for ts_id in block_tsubs[idx]:
            cover_counts[ts_id] += 1
    
    final_indices = []
    for idx in best_solution:
        # Check if any tsub is uniquely covered by this block
        is_redundant = True
        for ts_id in block_tsubs[idx]:
            if cover_counts[ts_id] == 1:
                is_redundant = False
                break
        if not is_redundant:
            final_indices.append(idx)
        else:
            for ts_id in block_tsubs[idx]:
                cover_counts[ts_id] -= 1
    
    result_blocks = [list(all_blocks[i]) for i in final_indices]
    
    return result_blocks
