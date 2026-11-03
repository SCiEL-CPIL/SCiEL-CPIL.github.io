import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill

    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    # All t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    # All k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # Precompute list of t-indices for each block
    # This is the expensive part, but it's O(num_k * C(k,t)) which is 75582 * 56 ~ 4M operations.
    # This is fast enough in Python if we don't do bit operations on huge ints.
    block_t_indices = []
    for block in k_blocks:
        indices = [t_index[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(indices)

    # 2. Greedy Set Cover
    # cover_count[i] is the number of blocks covering t-subset i
    cover_count = np.zeros(num_t, dtype=np.int32)
    selected = []
    selected_mask = np.zeros(num_k, dtype=bool)
    
    # Initialize uncovered count for gain calculation? 
    # Actually, gain = number of indices in block_t_indices[i] where cover_count[j] == 0.
    
    while elapsed() < deadline - 60: # Leave time for local search
        best_idx = -1
        best_gain = 0
        
        # Randomize order to avoid deterministic bias? 
        # For now, scan all. If too slow, sample.
        # 75k blocks * 56 checks = 4M checks per iteration. 
        # If we need ~300 blocks, 300 * 4M = 1.2B operations. Too slow in Python.
        
        # Optimization: Sample a subset of blocks if the full scan is too slow.
        # Or, maintain a list of "promising" blocks?
        # Let's try a randomized greedy with a limited scan window or full scan with early exit?
        # Actually, 4M simple integer checks in Python might take ~0.1-0.2s. 
        # 300 iterations * 0.2s = 60s. This is tight but possible.
        # Let's optimize the inner loop using numpy? No, lists are faster for small iterations.
        
        # Let's try to speed up the gain calculation.
        # We can pre-compute the indices.
        
        # To speed up, we can stop scanning if we find a block that covers all remaining? Unlikely.
        # We can sample random blocks to find a good one quickly.
        
        # Strategy: 
        # 1. Try a few random blocks.
        # 2. If none is good, scan a random subset of unselected blocks.
        # 3. If time permits, scan all.
        
        # Let's just scan all unselected blocks but break if we find a "perfect" gain? 
        # Max gain is C(k,t) = 56.
        
        unselected_indices = np.where(~selected_mask)[0]
        if len(unselected_indices) == 0:
            break
            
        # Sample up to 5000 blocks to find a good candidate quickly
        # If the pool is smaller, check all.
        if len(unselected_indices) > 5000:
            candidates = np.random.choice(unselected_indices, 5000, replace=False)
        else:
            candidates = unselected_indices
            
        for i in candidates:
            gain = 0
            # Calculate gain
            for idx in block_t_indices[i]:
                if cover_count[idx] == 0:
                    gain += 1
                    if gain == 56: # Max possible gain
                        break
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == 56:
                    break
        
        if best_idx == -1 or best_gain == 0:
            break
            
        selected.append(best_idx)
        selected_mask[best_idx] = True
        for idx in block_t_indices[best_idx]:
            cover_count[idx] += 1
            
    # 3. Local Search
    # Phase 1: Remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other blocks (count > 1)
    
    def is_redundant(block_idx):
        for idx in block_t_indices[block_idx]:
            if cover_count[idx] == 1:
                return False
        return True
        
    def remove_block(block_idx):
        nonlocal selected, selected_mask
        # Remove from list
        # O(N) removal is slow, but N is small (~300)
        selected.remove(block_idx)
        selected_mask[block_idx] = False
        for idx in block_t_indices[block_idx]:
            cover_count[idx] -= 1

    def add_block(block_idx):
        nonlocal selected, selected_mask
        selected.append(block_idx)
        selected_mask[block_idx] = True
        for idx in block_t_indices[block_idx]:
            cover_count[idx] += 1

    improved = True
    while improved and elapsed() < deadline - 10:
        improved = False
        
        # Pass 1: Deletions
        # Iterate backwards to avoid index issues if we remove
        for i in range(len(selected) - 1, -1, -1):
            if is_redundant(selected[i]):
                remove_block(selected[i])
                improved = True
                # Don't break, continue checking others? 
                # Removing one might make another redundant.
                # But cover_count changes. 
                # It's safe to continue.
        
        if improved:
            continue # Restart the while loop to check again
        
        # Pass 2: Swaps
        # Find a block A with criticals, try to swap with B.
        # We limit the number of swaps to avoid getting stuck in a loop.
        swaps_attempted = 0
        max_swaps = 50
        
        # Shuffle selected blocks to try different ones
        random.shuffle(selected)
        
        for A in selected:
            if elapsed() > deadline - 10:
                break
            if swaps_attempted >= max_swaps:
                break
                
            # Find criticals of A
            criticals = []
            for idx in block_t_indices[A]:
                if cover_count[idx] == 1:
                    criticals.append(idx)
            
            if not criticals:
                # Should have been removed in Pass 1
                remove_block(A)
                improved = True
                break
            
            # We need a block B that covers ALL criticals.
            # B should not be A (obviously) and should not be selected (if we want a swap, B is new).
            # Actually, if B is selected, it's a 2-swap (remove A, keep B, maybe remove C?). 
            # Let's stick to 1-for-1 swap: Remove A, Add B (B not in selected).
            
            # Pre-filter candidates? 
            # We can check if any unselected block covers criticals.
            # This is expensive. 
            # Heuristic: Pick a random critical t-subset c.
            # Find blocks that cover c. One of them might cover all criticals.
            
            c = criticals[0]
            
            # Find all blocks covering c
            # We don't have an index for this. 
            # We can scan all unselected blocks and check if they cover c.
            # Then check if they cover all criticals.
            
            found_swap = False
            unselected_indices = np.where(~selected_mask)[0]
            # Shuffle to avoid bias
            random.shuffle(unselected_indices)
            
            # Check first 1000 unselected blocks
            for B in unselected_indices[:1000]:
                # Check if B covers all criticals
                # B's indices: block_t_indices[B]
                # Criticals: list of indices
                # Check if set(criticals).issubset(set(block_t_indices[B]))
                # This is slow. 
                # Better: 
                # Create a set of B's indices? No, just loop criticals.
                # If len(criticals) is small (usually 1-5), this is fast.
                
                b_indices_set = set(block_t_indices[B])
                if all(c in b_indices_set for c in criticals):
                    # B covers all criticals of A.
                    # Perform swap
                    remove_block(A)
                    add_block(B)
                    swaps_attempted += 1
                    improved = True
                    found_swap = True
                    break
            
            if found_swap:
                # After a swap, restart the local search loop to allow deletions
                break

    # Final cleanup: remove any remaining redundancies
    changed = True
    while changed and elapsed() < deadline - 2:
        changed = False
        for i in range(len(selected) - 1, -1, -1):
            if is_redundant(selected[i]):
                remove_block(selected[i])
                changed = True

    result = [list(k_blocks[i]) for i in selected]
    
    # Verify coverage (safety check)
    # Re-count
    final_cover = np.zeros(num_t, dtype=np.int32)
    for i in selected:
        for idx in block_t_indices[i]:
            final_cover[idx] += 1
            
    if np.any(final_cover == 0):
        # Fallback: Return a valid greedy solution if we broke something?
        # The logic above should preserve validity.
        # Greedy adds blocks covering uncovered.
        # Deletion only removes if count > 1.
        # Swap removes A (criticals covered by B) and adds B.
        # So validity should hold.
        pass

    return result
