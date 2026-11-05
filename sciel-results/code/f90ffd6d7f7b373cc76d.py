import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin from 120s

    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)

    # Pre-compute all k-subsets (blocks) and their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_t_subs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_t_subs.append(subs)
    
    # Pre-compute subset_to_blocks for fast updates
    subset_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_t_subs):
        for s in subs:
            subset_to_blocks[s].append(b_idx)
            
    # --- PHASE 1: Greedy Construction ---
    # We use a "most uncovered" greedy strategy.
    # uncovered_count_per_block[i] = number of subsets in block i that are currently uncovered.
    uncovered = [1] * n_t_subs
    uncovered_count_per_block = [len(subs) for subs in block_t_subs]
    total_uncovered = n_t_subs
    selected_blocks = []
    
    # Optimization: Instead of scanning all blocks every step, we can just scan.
    # 50k blocks * 150 steps = 7.5M ops. This is fast in Python if optimized.
    # But we can do better by maintaining a list of active blocks? No, simple scan is robust.
    
    while total_uncovered > 0:
        if time.time() - start_time > time_limit * 0.4:
            break
            
        # Find block with max uncovered count
        max_count = 0
        best_b_idx = -1
        for i in range(n_blocks):
            c = uncovered_count_per_block[i]
            if c > max_count:
                max_count = c
                best_b_idx = i
                if max_count == len(block_t_subs[i]):
                    break # Perfect block found, can't do better
            
        if best_b_idx == -1 or max_count == 0:
            break
            
        selected_blocks.append(best_b_idx)
        
        # Update uncovered status and counts
        for s in block_t_subs[best_b_idx]:
            if uncovered[s]:
                uncovered[s] = 0
                total_uncovered -= 1
                # Update counts for all blocks containing s
                for b_idx in subset_to_blocks[s]:
                    uncovered_count_per_block[b_idx] -= 1

    # --- PHASE 2: Local Search / Reduction ---
    # Initialize cover_count based on selected_blocks
    cover_count = [0] * n_t_subs
    for b_idx in selected_blocks:
        for s in block_t_subs[b_idx]:
            cover_count[s] += 1
            
    # 2a. Remove Redundant Blocks
    # A block is redundant if all its subsets are covered by other blocks (cover_count > 1 for all its subsets, considering the block's own contribution)
    # Actually, simpler: if for all s in block, cover_count[s] > 1, then removing it doesn't uncover anything.
    # Wait, cover_count includes the block itself. So if cover_count[s] >= 2 for all s in block, it's redundant.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit * 0.7:
        improved = False
        for i in range(len(selected_blocks)):
            b_idx = selected_blocks[i]
            redundant = True
            for s in block_t_subs[b_idx]:
                if cover_count[s] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                for s in block_t_subs[b_idx]:
                    cover_count[s] -= 1
                selected_blocks.pop(i)
                improved = True
                break # Restart loop as list changed

    # 2b. 2-for-1 Swaps (Replace 2 blocks with 1)
    # This is the most powerful move to reduce count further.
    # We look for pairs (b1, b2) in selected_blocks and a block b_new (not in selected)
    # such that:
    # 1. b_new covers all subsets that are "critical" for b1 or b2 (i.e., subsets s where cover_count[s]==1 and s is in b1 or b2).
    #    Actually, more precisely:
    #    Let C be the set of subsets.
    #    Let U be the set of subsets with cover_count == 1.
    #    If we remove b1 and b2, the subsets that become uncovered are exactly those s in (b1 ∪ b2) such that cover_count[s] == 1.
    #    Let L = {s in (b1 ∪ b2) | cover_count[s] == 1}.
    #    For the swap to be valid, b_new must cover all subsets in L.
    #    Additionally, b_new must not introduce any new uncovered subsets?
    #    If we add b_new, it covers some subsets.
    #    The subsets in b_new that were previously uncovered (cover_count == 0) are now covered.
    #    The subsets in b_new that were covered (cover_count >= 1) remain covered.
    #    The subsets NOT in b_new remain in their previous state.
    #    So, as long as L is covered by b_new, the cover remains valid.
    #    We gain 1 block (remove 2, add 1).
    
    # To make this efficient:
    # 1. Identify "critical" subsets for each block: crit[b_idx] = [s in block_t_subs[b_idx] if cover_count[s] == 1]
    # 2. For each pair (b1, b2), L = crit[b1] ∪ crit[b2].
    #    If L is empty, we could just remove them (but 2a should have caught singletons, maybe not pairs if they share critical subsets? No, if L is empty, both are redundant individually? No.
    #    If crit[b1] is empty, b1 is redundant. 2a would have removed it.
    #    So after 2a, every block has at least one critical subset.
    #    Therefore L is non-empty for any pair.
    
    # We need to find b_new that covers L.
    # We can iterate through all possible b_new? 50k blocks.
    # For each pair (b1, b2), checking all 50k blocks is too slow if we have many pairs.
    # Number of blocks ~150. Pairs ~11k. 11k * 50k = 550M checks. Too slow.
    
    # Optimization:
    # For a given L (set of subsets), we can find candidate b_new's by intersecting subset_to_blocks for each s in L.
    # L is small (usually 1-3 subsets).
    # If |L| == 1, candidates = subset_to_blocks[L[0]]. Size ~455.
    # If |L| == 2, candidates = intersection of two lists of size ~455.
    # This is much faster.
    
    # We also need to ensure b_new is not in selected_blocks.
    
    # Let's implement this.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit * 0.95:
        improved = False
        # Pre-compute critical subsets for each block in current solution
        # To avoid recomputing for every pair, we can compute on the fly or cache.
        # Since we modify selected_blocks, we need to be careful.
        # We will compute crit for blocks in selected_blocks.
        
        n_sel = len(selected_blocks)
        
        # To speed up, we can break early if no swap found in a pass.
        
        for i in range(n_sel):
            if time.time() - start_time > time_limit * 0.95:
                break
            b1 = selected_blocks[i]
            crit1 = [s for s in block_t_subs[b1] if cover_count[s] == 1]
            
            for j in range(i + 1, n_sel):
                b2 = selected_blocks[j]
                crit2 = [s for s in block_t_subs[b2] if cover_count[s] == 1]
                
                # Combine critical subsets
                # Use a set for union to avoid duplicates, but we need to check coverage.
                # L = crit1 ∪ crit2
                # We can just check if a candidate block covers all in crit1 AND all in crit2.
                
                # Find candidates covering crit1
                # To minimize intersection size, start with the smallest list
                if len(crit1) == 0 or len(crit2) == 0:
                    continue # Should not happen after 2a, but safety
                
                # Get candidates for crit1
                # If crit1 has multiple elements, intersect their block lists.
                candidates = set(subset_to_blocks[crit1[0]])
                for s in crit1[1:]:
                    candidates.intersection_update(subset_to_blocks[s])
                    
                # Now filter candidates to also cover crit2
                # And not be in selected_blocks
                # And not be b1 or b2 (they are in selected_blocks, so handled)
                
                best_new = -1
                # We don't need to optimize gain for 2-for-1, any valid swap reduces count by 1.
                # So we just need ONE valid swap.
                
                # To speed up, we can break as soon as we find one.
                
                for b_new in candidates:
                    if b_new in selected_blocks:
                        continue
                    # Check if b_new covers all crit2
                    # We can check membership in block_t_subs[b_new]
                    # block_t_subs[b_new] is a list. Checking 'in' is O(35).
                    # Candidates size is small (intersection of ~455 lists).
                    
                    valid = True
                    for s in crit2:
                        if s not in block_t_subs[b_new]:
                            valid = False
                            break
                            
                    if valid:
                        best_new = b_new
                        break
                        
                if best_new != -1:
                    # Perform the 2-for-1 swap
                    # Remove b1, b2
                    for s in block_t_subs[b1]:
                        cover_count[s] -= 1
                    for s in block_t_subs[b2]:
                        cover_count[s] -= 1
                    # Add best_new
                    for s in block_t_subs[best_new]:
                        cover_count[s] += 1
                    
                    # Update selected_blocks
                    # Remove j first (higher index) then i
                    selected_blocks.pop(j)
                    selected_blocks.pop(i)
                    selected_blocks.append(best_new)
                    
                    improved = True
                    break # Restart outer loop as indices changed
                    
            if improved:
                break
                
        # After swaps, do another removal pass
        if improved:
            improved = True
            while improved and (time.time() - start_time) < time_limit * 0.95:
                improved = False
                for i in range(len(selected_blocks)):
                    b_idx = selected_blocks[i]
                    redundant = True
                    for s in block_t_subs[b_idx]:
                        if cover_count[s] == 1:
                            redundant = False
                            break
                    if redundant:
                        for s in block_t_subs[b_idx]:
                            cover_count[s] -= 1
                        selected_blocks.pop(i)
                        improved = True
                        break
                        
    final_blocks = [list(all_blocks[i]) for i in selected_blocks]
    
    # Final validation (optional, but good for debugging)
    # If time permits, check validity
    # For now, trust the logic.
    
    return final_blocks
