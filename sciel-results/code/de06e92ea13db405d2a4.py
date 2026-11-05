import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Pre-compute all t-subsets of v points
    # For v=24, t=3, C(24,3) = 2024 subsets.
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    
    # Map each subset to an index for vectorization
    subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Pre-compute for each point, the list of subset indices containing it?
    # Or for each block, the mask of subsets it covers.
    # Since k=7, C(7,3)=35 subsets per block.
    # We can represent a block's coverage as a boolean vector of length n_subsets.
    
    def block_to_mask(block):
        mask = np.zeros(n_subsets, dtype=bool)
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            mask[idx] = True
        return mask

    def greedy_cover():
        uncovered = np.ones(n_subsets, dtype=bool)
        blocks = []
        point_freq = np.zeros(v, dtype=np.int32)
        
        # Pre-compute for each point x, the mask of subsets containing x
        # This helps in calculating gain: gain(x, current_mask) = sum(uncovered & ~current_mask & point_mask[x])
        # Wait, gain(x, current_block) is subsets covered by adding x to current_block.
        # Subsets covered by block B are those where all elements are in B.
        # If we have current block C, and add x, new subsets are those containing x and (t-1) elements from C.
        # Let's stick to the explicit gain calculation but optimize it.
        
        while np.any(uncovered) and time.time() < deadline - 10:
            # Find the first uncovered subset
            # np.where(uncovered)[0][0] is fast
            idx = np.where(uncovered)[0][0]
            first_uncovered = all_t_subsets[idx]
            block = list(first_uncovered)
            
            while len(block) < k:
                best_point = -1
                best_gain = -1
                best_freq = float('inf')
                
                # Candidates: points not in block
                # Optimization: We can pre-compute which points are in the block
                in_block = set(block)
                
                # Calculate gain for each candidate
                # Gain(x) = number of subsets S such that x in S, S \ {x} subset block, and S is currently uncovered.
                # S is a t-subset. S = (x, y, z) for t=3.
                # So we need pairs (y, z) from block such that (x, y, z) is uncovered.
                
                # To speed this up, we can pre-compute for the current block `block`, 
                # the set of pairs it contains.
                # And for each candidate x, check how many (x, y, z) are uncovered.
                
                # Actually, a simpler vectorized approach for the whole gain vector:
                # We want to find x that maximizes gain.
                # Gain(x) = sum_{sub in uncovered} [x in sub and (sub \ {x}) subset block]
                
                # Let's iterate candidates. v=24 is small.
                candidates = [x for x in range(v) if x not in in_block]
                
                if not candidates:
                    break # Should not happen if k < v
                
                # For each candidate, compute gain
                # We can vectorize this if we pre-compute masks.
                # Let's create a mask for the current block's (t-1) subsets?
                # No, let's just loop candidates and use a fast check.
                
                # Pre-compute pairs in block
                block_pairs = list(itertools.combinations(block, t - 1))
                
                gains = []
                for x in candidates:
                    count = 0
                    for pair in block_pairs:
                        # Construct subset
                        # Since block is built incrementally, it's not necessarily sorted? 
                        # We should keep block sorted to make subset construction easy?
                        # Or just use set/tuple.
                        sub = tuple(sorted(pair + (x,)))
                        idx_sub = subset_to_idx[sub]
                        if uncovered[idx_sub]:
                            count += 1
                    if count > 0:
                        gains.append((count, -point_freq[x], x))
                    else:
                        # If gain is 0, we still might need to pick something?
                        # But if we start with an uncovered subset, the first k-t points 
                        # must have positive gain to complete that subset.
                        # If gain is 0 for all, we are in a bad spot.
                        # Pick lowest freq.
                        gains.append((0, -point_freq[x], x))
                
                # Sort by gain desc, then freq asc
                gains.sort(key=lambda x: (x[0], x[1]), reverse=True)
                
                best_point = gains[0][2]
                block.append(best_point)
                point_freq[best_point] += 1
                
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            mask = block_to_mask(block)
            uncovered &= ~mask
            
        return blocks

    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        n_blocks = len(blocks)
        # Pre-compute masks
        masks = [block_to_mask(b) for b in blocks]
        
        removed = [False] * n_blocks
        changed = True
        
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(n_blocks):
                if removed[i]:
                    continue
                
                # Check if mask[i] is subset of OR of all other masks
                # other_mask = OR_{j != i} mask[j]
                # If mask[i] & ~other_mask is empty, then i is redundant.
                
                # Efficiently compute other_mask?
                # For small n_blocks (~70-80), we can just iterate.
                # But OR of 70 vectors of length 2024 is fast in NumPy.
                
                # Let's accumulate the OR of all masks except i.
                # This is O(N^2 * M) if done naively.
                # N=80, M=2024. 80*80*2024 ~ 13M ops. Fast enough.
                
                # Optimization: 
                # total_mask = OR of all masks.
                # If mask[i] is a subset of total_mask, it doesn't mean it's redundant.
                # We need to know if the subsets in mask[i] are covered by *other* blocks.
                
                # Let's just compute the union of all masks except i.
                # We can do this by iterating j != i.
                
                # Alternative: 
                # For each subset s in mask[i], check if any other block covers s.
                # This is O(M * N) per block i. Total O(N^2 * M).
                
                # Let's try the vectorized OR approach.
                # It might be faster to just check redundancy by counting.
                # coverage_count[s] = number of blocks covering s.
                # If coverage_count[s] > 1 for all s in mask[i], then i is redundant.
                
                # Let's maintain coverage counts.
                # But coverage counts change as we remove blocks.
                
                # Let's stick to the simple check:
                # redundant = True
                # for s_idx in np.where(masks[i])[0]:
                #     # Check if any other block covers s_idx
                #     covered = False
                #     for j in range(n_blocks):
                #         if j == i or removed[j]:
                #             continue
                #         if masks[j][s_idx]:
                #             covered = True
                #             break
                #     if not covered:
                #         redundant = False
                #         break
                # if redundant:
                #     removed[i] = True
                #     changed = True
                
                # This is O(N^2 * M) in worst case.
                # N=80, M=2024. 80*80*2024 = 13M.
                # This loop runs until no changes.
                # Should be fast.
                
                redundant = True
                subs_in_i = np.where(masks[i])[0]
                for s_idx in subs_in_i:
                    covered = False
                    for j in range(n_blocks):
                        if j == i or removed[j]:
                            continue
                        if masks[j][s_idx]:
                            covered = True
                            break
                    if not covered:
                        redundant = False
                        break
                
                if redundant:
                    removed[i] = True
                    changed = True
        
        return [b for i, b in enumerate(blocks) if not removed[i]]

    def local_search(blocks):
        current_blocks = blocks[:]
        current_size = len(current_blocks)
        
        # Pre-compute masks for current blocks
        masks = [block_to_mask(b) for b in current_blocks]
        
        # Coverage counts
        coverage_counts = np.sum(masks, axis=0)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try Add-Remove-Remove
            # Generate a candidate block
            # Strategy: Pick a random subset of uncovered or low-coverage subsets?
            # Or just pick a random block from a pool of "good" blocks?
            # Let's try to construct a block that covers many subsets with low coverage.
            
            # For simplicity, let's try random swaps/adds.
            # But random might be slow.
            
            # Better: Try to remove each block and see if we can cover the lost subsets 
            # with a single new block.
            
            for i in range(len(current_blocks)):
                if time.time() > deadline - 5:
                    break
                
                # Remove block i
                lost_mask = masks[i]
                # Subtract from coverage
                coverage_counts -= lost_mask
                # Find subsets that are now uncovered (coverage 0)
                newly_uncovered = np.where(coverage_counts == 0)[0]
                
                if len(newly_uncovered) == 0:
                    # Block i was redundant (or its subsets were covered by others)
                    # Remove it
                    current_blocks.pop(i)
                    masks.pop(i)
                    current_size -= 1
                    improved = True
                    break # Restart loop with new size
                else:
                    # We need to cover newly_uncovered.
                    # Can we find a single block that covers all newly_uncovered?
                    # Or at least reduces the count?
                    # If we add a block B_new that covers all newly_uncovered, 
                    # the net change is 0 (remove 1, add 1). 
                    # But maybe B_new makes other blocks redundant?
                    
                    # Let's try to find a block that covers the newly_uncovered subsets.
                    # The intersection of all newly_uncovered subsets?
                    # No, we need a block that *contains* each of these subsets.
                    # So the block must contain the union of all elements in newly_uncovered subsets?
                    # If the union has size > k, we can't cover them all with one block.
                    
                    union_set = set()
                    for idx in newly_uncovered:
                        union_set.update(all_t_subsets[idx])
                    
                    if len(union_set) <= k:
                        # We can form a block from this union (plus random fillers if needed)
                        # But we must ensure it covers all newly_uncovered.
                        # Since the block contains the union, it contains each subset.
                        candidate_block = sorted(list(union_set))
                        while len(candidate_block) < k:
                            # Add random point
                            candidate_block.append(random.randint(0, v-1))
                        candidate_block = sorted(list(set(candidate_block)))[:k]
                        
                        # Check if this block covers all newly_uncovered
                        # (It should, by construction, if union_set size <= k)
                        
                        # Add candidate block
                        new_mask = block_to_mask(candidate_block)
                        
                        # Check if it actually covers all newly_uncovered
                        if np.all(new_mask[newly_uncovered]):
                            # Try adding it
                            # Restore coverage for removed block i? No, we are replacing.
                            # We removed i, now we add candidate.
                            
                            # Check if candidate makes other blocks redundant
                            # This is complex. Let's just add it and run remove_redundant.
                            
                            # Tentative new state
                            tentative_blocks = current_blocks + [candidate_block]
                            tentative_blocks = remove_redundant(tentative_blocks)
                            
                            if len(tentative_blocks) < current_size:
                                current_blocks = tentative_blocks
                                current_size = len(current_blocks)
                                improved = True
                                break # Restart loop
                            elif len(tentative_blocks) == current_size:
                                # No improvement, but maybe better structure?
                                # For now, only accept if size decreases.
                                pass
                            
                    # Revert coverage counts
                    coverage_counts += lost_mask

        return current_blocks

    # Main loop with restarts
    best_blocks = None
    best_size = float('inf')
    
    # Run greedy multiple times with different random seeds or tie-breaking
    for restart in range(20):
        if time.time() > deadline - 30:
            break
            
        random.seed(restart)
        
        # Greedy phase
        greedy_blocks = greedy_cover()
        
        # Remove redundant
        cleaned_blocks = remove_redundant(greedy_blocks)
        
        # Local search
        # Only do local search if we have time
        if time.time() < deadline - 15:
            local_blocks = local_search(cleaned_blocks)
        else:
            local_blocks = cleaned_blocks
            
        if len(local_blocks) < best_size:
            best_size = len(local_blocks)
            best_blocks = local_blocks[:]
            
            # If we find a very good solution, we might stop early?
            # Best known is 76. If we find 75, we should keep searching for 74.
            # But if time is running out, return best.
            
        # Print progress (optional, but good for debugging)
        # print(f"Restart {restart}: Size {len(local_blocks)}")
        
    if best_blocks is None:
        # Fallback: just return the last greedy result
        best_blocks = greedy_cover()
        best_blocks = remove_redundant(best_blocks)
        
    return best_blocks

# Test locally
if __name__ == "__main__":
    import sys
    # v, k, t = 24, 7, 3
    # blocks = construct(24, 7, 3)
    # print(f"Number of blocks: {len(blocks)}")
    pass
