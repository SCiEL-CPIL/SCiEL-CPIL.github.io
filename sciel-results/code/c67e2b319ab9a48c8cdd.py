import itertools
import time
import random
import sys

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    # 1. Precompute all t-subsets and map to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute all k-blocks and their coverage masks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # block_masks[i] is a set of t-subset indices covered by block i
    # Using sets for faster intersection/union? Or bitmasks?
    # Bitmasks are faster for "is subset" checks if we use bit operations.
    # But n_t_subs = 3876. Python ints handle this fine.
    
    block_masks = [0] * n_k_blocks
    # Also store the list of t-subset indices for each block for fast iteration
    block_t_indices = [[] for _ in range(n_k_blocks)]
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_t_indices[i] = indices
        
    # Precompute for each t-subset, which blocks cover it
    # t_to_blocks[t_idx] = list of block indices
    t_to_blocks = [[] for _ in range(n_t_subs)]
    for i in range(n_k_blocks):
        for idx in block_t_indices[i]:
            t_to_blocks[idx].append(i)

    full_mask = (1 << n_t_subs) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    # Helper: Greedy construction
    def greedy_construct(randomize=False):
        uncovered_mask = full_mask
        selected = []
        
        while uncovered_mask != 0:
            if time.time() > deadline - 20:
                break
            
            best_idx = -1
            best_gain = -1
            best_candidates = []
            
            # Optimization: If randomize, we can pick from top candidates
            # But standard greedy is good enough for initial solution
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_candidates = [i]
                elif gain == best_gain:
                    best_candidates.append(i)
            
            if best_idx == -1 or best_gain == 0:
                break
            
            if randomize and len(best_candidates) > 1:
                best_idx = random.choice(best_candidates)
            
            selected.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
        
        return selected

    # Helper: Local Search
    def local_search(selected):
        if not selected:
            return selected
        
        current = list(selected)
        
        # Build cover_count
        cover_count = [0] * n_t_subs
        for bi in current:
            for idx in block_t_indices[bi]:
                cover_count[idx] += 1
        
        # Set of selected blocks for fast lookup
        selected_set = set(current)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Phase 1: Remove redundant blocks
            # A block is redundant if it doesn't cover any t-subset exclusively
            new_current = []
            for bi in current:
                is_redundant = True
                for idx in block_t_indices[bi]:
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove bi
                    for idx in block_t_indices[bi]:
                        cover_count[idx] -= 1
                    selected_set.remove(bi)
                    improved = True
                else:
                    new_current.append(bi)
            current = new_current
            
            if time.time() > deadline - 10:
                break
                
            # Phase 2: 2-1 Exchange
            # Try to replace 2 blocks with 1 block, or remove 2 blocks
            n = len(current)
            # We iterate over pairs. To avoid O(N^2) explosion if N is large, 
            # we rely on N being small (~150).
            # Optimization: Only check pairs that might be beneficial?
            # Hard to know. Just do it.
            
            found_improvement = False
            # We need to be careful with modifying current while iterating
            # We'll collect changes and apply them
            
            # To speed up, we can randomize the order of pairs
            pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    pairs.append((i, j))
            random.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() > deadline - 10:
                    break
                if i >= len(current) or j >= len(current):
                    continue # Indices might have changed if we modified current? 
                             # We are not modifying current inside the loop yet.
                
                bi = current[i]
                bj = current[j]
                
                # Temporarily remove bi and bj
                # Check if all t-subsets in bi and bj are covered by others
                uncovered = []
                # We need to check cover_count - 2 (if they share t-subsets) or -1
                # It's safer to compute the "effective" cover count for the union
                
                # Let's just decrement and check
                # We need to know if any t-subset becomes 0
                
                # Optimization:
                # t_subsets_in_bi = set(block_t_indices[bi])
                # t_subsets_in_bj = set(block_t_indices[bj])
                # union = t_subsets_in_bi | t_subsets_in_bj
                
                # Instead of sets, we can iterate through the union of indices
                # But we need to handle overlaps.
                
                # Simple approach:
                # Decrement counts for bi
                for idx in block_t_indices[bi]:
                    cover_count[idx] -= 1
                for idx in block_t_indices[bj]:
                    cover_count[idx] -= 1
                
                # Check for zeros
                is_valid_removal = True
                uncovered_indices = []
                # We only need to check indices in bi or bj
                # But we decremented them. If any is 0, it's uncovered.
                
                # To avoid iterating over all 3876, we iterate over the union of bi and bj indices
                # We can just check the indices we touched.
                
                # But if bi and bj share an index, we decremented it twice.
                # If cover_count was 2, it becomes 0.
                
                # We need to check all indices in bi and bj.
                # To avoid duplicates, we can use a set, but that's slow.
                # We can just check bi, then check bj (duplicates are fine, just check twice).
                
                for idx in block_t_indices[bi]:
                    if cover_count[idx] == 0:
                        is_valid_removal = False
                        uncovered_indices.append(idx)
                for idx in block_t_indices[bj]:
                    if cover_count[idx] == 0:
                        if idx not in uncovered_indices: # Avoid duplicate in list
                            uncovered_indices.append(idx)
                        # If it was already 0 from bi, it's still 0.
                        # We just need to know if it's 0.
                        
                if not is_valid_removal:
                    # Restore counts
                    for idx in block_t_indices[bi]:
                        cover_count[idx] += 1
                    for idx in block_t_indices[bj]:
                        cover_count[idx] += 1
                    continue
                
                # If we get here, we can remove bi and bj
                # But wait, we might be able to add ONE block to cover the uncovered ones?
                # No, if is_valid_removal is True, uncovered_indices is empty.
                # So we can just remove both.
                
                # Remove bi and bj from current
                # We need to remove them from the list.
                # Since we are iterating over pairs, modifying list is tricky.
                # Let's mark them for removal.
                
                # Actually, if we remove 2, size drops by 2.
                # Let's just do it.
                
                # Remove bj (higher index) first to keep indices valid for bi?
                # No, we are in a loop over pairs.
                # Let's restart the pair loop or break.
                
                # Simpler:
                # Remove bi and bj.
                current.remove(bj)
                current.remove(bi)
                selected_set.remove(bj)
                selected_set.remove(bi)
                improved = True
                found_improvement = True
                break # Restart pair loop with new current
                
            if found_improvement:
                continue # Restart while loop to re-evaluate redundancy and new pairs
            
            # If no 2-removal found, try 2-1 exchange (replace 2 with 1)
            # This is harder. We need to find a block that covers the uncovered ones.
            # But we only found uncovered ones if is_valid_removal was False.
            # In that case, we restored counts.
            
            # Let's add a specific pass for 2-1 exchange.
            # For each pair (bi, bj):
            # 1. Remove them.
            # 2. Find uncovered indices.
            # 3. If uncovered is not empty, find a block bk that covers all uncovered.
            # 4. If found, replace bi, bj with bk.
            
            # This is expensive. We only do it if time permits.
            
            if time.time() > deadline - 15:
                break
                
            n = len(current)
            pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    pairs.append((i, j))
            random.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() > deadline - 15:
                    break
                if i >= len(current) or j >= len(current):
                    continue
                
                bi = current[i]
                bj = current[j]
                
                # Temporarily remove
                for idx in block_t_indices[bi]:
                    cover_count[idx] -= 1
                for idx in block_t_indices[bj]:
                    cover_count[idx] -= 1
                
                uncovered_indices = []
                for idx in block_t_indices[bi]:
                    if cover_count[idx] == 0:
                        uncovered_indices.append(idx)
                for idx in block_t_indices[bj]:
                    if cover_count[idx] == 0:
                        if idx not in uncovered_indices:
                            uncovered_indices.append(idx)
                
                if not uncovered_indices:
                    # Can remove both, but we handled that in previous phase?
                    # If previous phase didn't catch it, maybe order matters.
                    # Remove both.
                    current.remove(bj)
                    current.remove(bi)
                    selected_set.remove(bj)
                    selected_set.remove(bi)
                    improved = True
                    break
                else:
                    # Try to find a block bk that covers all uncovered_indices
                    # bk must not be bi or bj (already removed)
                    # bk must cover all idx in uncovered_indices
                    
                    # Candidate blocks: intersection of t_to_blocks[idx] for idx in uncovered_indices
                    # To speed up, sort uncovered_indices by len(t_to_blocks[idx])
                    uncovered_indices.sort(key=lambda x: len(t_to_blocks[x]))
                    
                    candidates = None
                    for idx in uncovered_indices:
                        blocks_covering_idx = t_to_blocks[idx]
                        if candidates is None:
                            candidates = set(blocks_covering_idx)
                        else:
                            candidates.intersection_update(blocks_covering_idx)
                            
                        if not candidates:
                            break
                    
                    if candidates:
                        # Filter out bi, bj and any block already in selected_set
                        # Actually, if bk is in selected_set, we are replacing bi, bj with bk.
                        # If bk is bi or bj, that's not a replacement.
                        valid_candidates = [bk for bk in candidates if bk not in selected_set]
                        
                        if valid_candidates:
                            # Pick one (random or first)
                            bk = random.choice(valid_candidates)
                            
                            # Apply change
                            # Remove bi, bj
                            current.remove(bj)
                            current.remove(bi)
                            selected_set.remove(bj)
                            selected_set.remove(bi)
                            
                            # Add bk
                            current.append(bk)
                            selected_set.add(bk)
                            
                            # Update cover_count
                            # We already decremented bi, bj.
                            # Now increment bk.
                            for idx in block_t_indices[bk]:
                                cover_count[idx] += 1
                                
                            improved = True
                            break
                    else:
                        # No candidate found, restore counts
                        for idx in block_t_indices[bi]:
                            cover_count[idx] += 1
                        for idx in block_t_indices[bj]:
                            cover_count[idx] += 1

            if improved:
                continue # Restart while loop
        
        return current

    # Main Loop: Multiple Restarts
    max_restarts = 10
    restart_time = (deadline - time.time() - 10) / max_restarts
    
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Greedy with randomization
        selected = greedy_construct(randomize=True)
        
        # Local Search
        selected = local_search(selected)
        
        # Verify and update best
        cov = 0
        for bi in selected:
            cov |= block_masks[bi]
        if cov == full_mask and len(selected) < best_count:
            best_count = len(selected)
            best_blocks = list(selected)
            
        # If we found a very good solution, we can spend more time polishing
        if best_count <= 152:
             # Do more local search passes on the best solution?
             pass

    if best_blocks is None:
        # Fallback
        selected = greedy_construct(randomize=False)
        selected = local_search(selected)
        best_blocks = selected

    return [list(all_k_blocks[i]) for i in best_blocks]
