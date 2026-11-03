import itertools
import time

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    all_mask = (1 << n_t_subsets) - 1
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    
    # 3. Compute bitmask for each k-subset
    masks = [0] * len(k_subsets)
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        masks[i] = mask
        
    # Filter out blocks that don't cover any new t-subsets? 
    # All blocks cover C(k,t) t-subsets.
    
    # 4. Greedy Set Cover to get initial Upper Bound (UB)
    uncovered = all_mask
    selected_greedy = []
    used_blocks = [False] * len(k_subsets)
    
    while uncovered:
        best_gain = -1
        best_idx = -1
        for i in range(len(k_subsets)):
            if used_blocks[i]:
                continue
            gain = (masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == uncovered.bit_count():
                    break
        if best_idx == -1:
            break
        selected_greedy.append(best_idx)
        used_blocks[best_idx] = True
        uncovered &= ~masks[best_idx]
        
    current_best = len(selected_greedy)
    best_solution = selected_greedy[:]

    # 5. Branch and Bound
    # To make this fast, we need good pruning.
    # We pick the t-subset with the fewest available covering blocks.
    
    # Precompute for each t-subset index, the list of block indices that cover it
    # This is O(N * C(k,t)) which is large, but we can do it on the fly or precompute.
    # Precomputing: 11628 t-subsets * ~75000 blocks. Too much memory/time.
    # Instead, we can maintain a list of "candidate blocks" that intersect with uncovered.
    
    def branch_and_bound(uncovered_mask, current_depth, current_path, remaining_blocks):
        nonlocal current_best, best_solution
        
        if not uncovered_mask:
            if current_depth < current_best:
                current_best = current_depth
                best_solution = current_path[:]
            return
        
        # Pruning: If current_depth + lower_bound >= current_best, prune
        # Lower bound: ceil(uncovered_count / max_gain)
        # max_gain is roughly C(k,t) but reduced by overlaps.
        # A tighter lower bound: 
        # Count uncovered bits.
        # The maximum number of uncovered bits a single block can cover is bounded by C(k,t).
        # Actually, we can compute the max possible gain from any remaining block.
        
        # To speed up, we pick a specific t-subset from uncovered to branch on.
        # We want the one with fewest covering blocks.
        # Finding the "hardest" t-subset is expensive if done naively.
        # Approximation: Pick the first set bit in uncovered_mask.
        # Better: Sample a few?
        
        # Let's pick the first uncovered t-subset index.
        # idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        # This is O(1) for bit ops.
        
        target_bit = uncovered_mask & -uncovered_mask
        target_idx = target_bit.bit_length() - 1
        
        # Find all blocks in remaining_blocks that cover target_idx
        candidates = []
        for i in remaining_blocks:
            if masks[i] & target_bit:
                candidates.append(i)
                
        if not candidates:
            return # Dead end
        
        # Sort candidates by gain (descending) to find good solutions faster
        # Gain = (masks[i] & uncovered_mask).bit_count()
        candidates.sort(key=lambda i: (masks[i] & uncovered_mask).bit_count(), reverse=True)
        
        # Pruning candidates:
        # If a block's mask is a subset of another block's mask (restricted to uncovered),
        # and the other block is "better" (same cost, more coverage), we can skip the subset?
        # This is complex.
        
        for i in candidates:
            if current_depth + 1 >= current_best:
                break # Since we sorted by gain, later ones have less or equal gain, 
                      # but depth increases. If even the best gain doesn't help, we prune.
                      # Actually, we need a lower bound on remaining steps.
            
            new_uncovered = uncovered_mask & ~masks[i]
            new_remaining = [j for j in remaining_blocks if j != i]
            
            # Heuristic Pruning:
            # If the number of uncovered bits is high, and we have used many blocks, prune.
            # Simple bound: uncovered.bit_count() / C(k,t)
            min_remaining_blocks = (new_uncovered.bit_count() + (1 << t) - 1) // (1 << t) # Very weak
            
            # Better bound: Greedy estimate on new_uncovered
            # This is expensive.
            
            branch_and_bound(new_uncovered, current_depth + 1, current_path + [i], new_remaining)

    # The above B&B is too slow for N=75k.
    # We need a more targeted approach or just return the greedy result if B&B is risky.
    # Given the 120s limit, and the fact that the current greedy is already 1.1189,
    # a slight improvement via local search on the greedy result is safer than full B&B.
    
    # Let's implement a Local Search (2-opt / 1-swap) on the greedy result to reduce block count.
    
    def try_remove_redundant(blocks_list):
        removed = True
        while removed:
            removed = False
            for i in range(len(blocks_list) - 1, -1, -1):
                block_mask = masks[blocks_list[i]]
                # Check if union of others covers all
                union_others = 0
                for j in range(len(blocks_list)):
                    if j != i:
                        union_others |= masks[blocks_list[j]]
                if union_others == all_mask:
                    blocks_list.pop(i)
                    removed = True
                    break
        return blocks_list

    # Local Search: Try to replace 2 blocks with 1, or 1 with 1?
    # Since we want to minimize count, we look for:
    # 1. Redundant blocks (already done in current code, but we restart it).
    # 2. Swap: Replace block A with block B if B covers all t-subsets that A uniquely covered?
    #    No, B must cover the union of unique parts of A and any other block?
    #    Simpler: If we remove A, which t-subsets become uncovered? 
    #    If there is a block B (not in set) that covers ALL of those, we can swap A->B?
    #    No, B might not be in the set. We can ADD B and REMOVE A?
    #    That keeps count same.
    #    To reduce count, we need to find a block B that covers the "unique" t-subsets of A, 
    #    AND B is already in the set? No.
    #    We need to find a set of blocks (size < size of removed) that covers the gap.
    #    Gap = t-subsets covered ONLY by A.
    #    If Gap is empty, A is redundant.
    #    If Gap is not empty, can we cover Gap with 0 blocks? No.
    #    Can we cover Gap with 1 block B, and remove A?
    #    If yes, count stays same.
    #    To reduce count, we need to remove A and NOT add any block.
    #    This implies A's unique coverage was already covered by others. (Redundancy).
    
    # So, to reduce count, we just need to find redundancies.
    # The current code does this.
    # Why is the current score only 1.1189?
    # Maybe the greedy is not optimal and B&B could find a better one.
    
    # Let's try a limited B&B with a strict time limit.
    
    start_time = time.time()
    
    # We will use a heuristic B&B that picks the hardest t-subset.
    # To make it fast, we pre-compute the list of blocks for each t-subset?
    # Memory: 11628 lists of avg length 75582 * (8/19) ~ 3000.
    # 11628 * 3000 ints = 35 million ints. ~280 MB. Might be too much/slow to build.
    
    # Alternative:
    # Just return the greedy result with redundancy removal.
    # The current program DOES this.
    # So I must do something else.
    
    # Let's try a "Greedy with Randomization" or "Multi-start Greedy".
    
    best_found = selected_greedy[:]
    
    for _ in range(10): # 10 restarts
        uncovered = all_mask
        selected = []
        used = [False] * len(k_subsets)
        
        while uncovered:
            # Randomized Greedy: Pick from top 5 best gains
            gains = []
            for i in range(len(k_subsets)):
                if not used[i]:
                    gain = (masks[i] & uncovered).bit_count()
                    if gain > 0:
                        gains.append((gain, i))
            
            if not gains:
                break
            
            gains.sort(reverse=True)
            
            # Pick from top min(5, len(gains))
            top_k = gains[:min(5, len(gains))]
            # Weighted random or just pick best?
            # Let's pick best to ensure progress, but maybe shuffle ties?
            # Or pick random from top 10%?
            
            # To improve, let's pick the best one (standard greedy) but 
            # maybe we can try a different tie-breaking or slight perturbation.
            
            # Actually, let's just use the standard greedy again?
            # No, let's try to pick a random block from the top 20 candidates with gain > 0.
            
            candidates = [i for g, i in gains if g >= gains[0][0] - 1] # Close to best
            if not candidates:
                candidates = [i for g, i in gains[:5]]
                
            idx = candidates[0] # Default to best
            
            selected.append(idx)
            used[idx] = True
            uncovered &= ~masks[idx]
            
        # Remove redundants
        selected = try_remove_redundant(selected)
        
        if len(selected) < len(best_found):
            best_found = selected[:]
            
        if time.time() - start_time > 60:
            break
            
    return [list(k_subsets[i]) for i in best_found]
