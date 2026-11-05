import itertools
import time
import random
import sys

def construct(v, k, t):
    # Parameters for C(19, 8, 5)
    if v != 19 or k != 8 or t != 5:
        # Fallback for other params, though task specifies C(19,8,5)
        # Simple greedy fallback
        return _simple_greedy(v, k, t)

    start_time = time.time()
    time_limit = 115.0
    
    # 1. Precomputation
    all_points = list(range(v))
    all_t_combos = list(itertools.combinations(all_points, t))
    num_t = len(all_t_combos)
    
    # Map t-combo to index
    t_combo_to_idx = {}
    for i, c in enumerate(all_t_combos):
        t_combo_to_idx[c] = i
        
    # Precompute all k-blocks and their covered t-subsets
    # block_mask -> list of t_subset_indices
    all_k_combos = list(itertools.combinations(all_points, k))
    num_k = len(all_k_combos)
    
    # Map k-combo to index
    k_combo_to_idx = {}
    for i, c in enumerate(all_k_combos):
        k_combo_to_idx[c] = i
        
    # block_to_tsubs[block_idx] = list of t_subset_indices
    block_to_tsubs = [[] for _ in range(num_k)]
    # Also need to quickly check if a block is in the current set?
    # We will maintain a boolean array `in_set` of size num_k.
    
    # Fill block_to_tsubs
    for k_idx, k_combo in enumerate(all_k_combos):
        # Generate all t-subsets of this k-combo
        for t_sub in itertools.combinations(k_combo, t):
            t_idx = t_combo_to_idx[t_sub]
            block_to_tsubs[k_idx].append(t_idx)
            
    # 2. Greedy Initialization
    def run_greedy():
        cover_count = [0] * num_t
        in_set = [False] * num_k
        blocks = []
        uncovered = set(range(num_t))
        
        # To speed up greedy, we can pick a random uncovered t-subset and find best block covering it
        while uncovered:
            # Pick random uncovered
            target_t_idx = random.choice(list(uncovered))
            
            best_gain = -1
            best_block_idx = -1
            
            # Find all blocks covering target_t_idx
            # We need an index: t_subset -> list of blocks covering it
            # Precompute this? 
            # t_subset_to_blocks[t_idx] = list of k_idx
            
            # This precomputation is expensive: 11628 * (avg blocks per t-subset)
            # Avg blocks per t-subset: C(19,8) * C(8,5) / C(19,5) = 75582 * 56 / 11628 = 364.
            # Total entries: 11628 * 364 = 4.2M. Same as above.
            # Let's build it.
            
            # Actually, let's just iterate all blocks? 75k * 56 checks * 11628 steps? No.
            # We need t_subset_to_blocks.
            
            pass # Placeholder, we will build t_subset_to_blocks in main scope
            
        return blocks, cover_count, in_set

    # Build t_subset_to_blocks
    t_subset_to_blocks = [[] for _ in range(num_t)]
    for k_idx in range(num_k):
        for t_idx in block_to_tsubs[k_idx]:
            t_subset_to_blocks[t_idx].append(k_idx)

    def run_greedy():
        cover_count = [0] * num_t
        in_set = [False] * num_k
        blocks = []
        uncovered = set(range(num_t))
        
        while uncovered:
            target_t_idx = random.choice(list(uncovered))
            candidates = t_subset_to_blocks[target_t_idx]
            
            best_gain = -1
            best_block_idx = -1
            
            for b_idx in candidates:
                if in_set[b_idx]:
                    continue
                # Calculate gain: number of uncovered t-subsets in block_to_tsubs[b_idx]
                gain = 0
                for t_idx in block_to_tsubs[b_idx]:
                    if cover_count[t_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    
            if best_block_idx == -1:
                # Should not happen
                break
                
            # Add block
            in_set[best_block_idx] = True
            blocks.append(best_block_idx)
            for t_idx in block_to_tsubs[best_block_idx]:
                if cover_count[t_idx] == 0:
                    uncovered.discard(t_idx)
                cover_count[t_idx] += 1
                
        return blocks, cover_count, in_set

    def local_search(blocks, cover_count, in_set):
        # blocks: list of k_idx
        # cover_count: array
        # in_set: boolean array
        
        # We will try to remove blocks.
        # If removal causes uncovered, we try to find a replacement.
        
        improved = True
        while improved:
            improved = False
            # Shuffle blocks to avoid local minima order dependency
            random.shuffle(blocks)
            
            for i in range(len(blocks)):
                b_idx = blocks[i]
                
                # Try to remove b_idx
                # Decrement counts
                for t_idx in block_to_tsubs[b_idx]:
                    cover_count[t_idx] -= 1
                    
                # Check if any became 0
                deficit = []
                for t_idx in block_to_tsubs[b_idx]:
                    if cover_count[t_idx] == 0:
                        deficit.append(t_idx)
                        
                if not deficit:
                    # Success! Remove it.
                    in_set[b_idx] = False
                    blocks.pop(i)
                    improved = True
                    # Note: popping changes indices. We break to restart loop? 
                    # Or handle carefully. Restarting is safer.
                    break
                else:
                    # Failure. Find best block to cover deficit.
                    # We want a block that covers max deficit elements.
                    # Candidates: any block NOT in set.
                    # To be fast, we can look at blocks covering the deficit elements.
                    # Union of t_subset_to_blocks[d] for d in deficit.
                    
                    candidate_set = set()
                    for d in deficit:
                        candidate_set.update(t_subset_to_blocks[d])
                        
                    best_gain = 0
                    best_new_block = -1
                    
                    # Optimization: Only consider blocks that cover at least 1 deficit element?
                    # Yes, that's what candidate_set is.
                    
                    for cand_idx in candidate_set:
                        if in_set[cand_idx]:
                            continue
                        # Calculate gain on deficit
                        gain = 0
                        # It's faster to iterate deficit and check if cand covers it?
                        # Or iterate block's tsubs and check if in deficit?
                        # Deficit is small (usually).
                        # Let's make a set of deficit for O(1) lookup? 
                        # Deficit size is usually small.
                        deficit_set = set(deficit)
                        for t_idx in block_to_tsubs[cand_idx]:
                            if t_idx in deficit_set:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_new_block = cand_idx
                            
                    if best_new_block != -1 and best_gain > 0:
                        # Replace b_idx with best_new_block
                        # We already decremented b_idx.
                        # Now increment best_new_block.
                        in_set[b_idx] = False
                        in_set[best_new_block] = True
                        blocks[i] = best_new_block
                        for t_idx in block_to_tsubs[best_new_block]:
                            cover_count[t_idx] += 1
                            # Note: cover_count was decremented for b_idx.
                            # If t_idx was in deficit (count 0), it is now 1.
                            # If t_idx was not in deficit (count > 0), it is now count+1.
                            # This is correct.
                        # Did we improve?
                        # We replaced 1 block with 1 block. Count is same.
                        # But we might have created a situation where another block can be removed.
                        # So we mark improved to continue loop.
                        improved = True
                        break # Restart loop to re-shuffle? Or continue?
                        # Breaking and restarting the while loop ensures we don't miss interactions.
                    else:
                        # No good replacement. Revert.
                        for t_idx in block_to_tsubs[b_idx]:
                            cover_count[t_idx] += 1
                        # in_set[b_idx] is still True.
                        
        return blocks, cover_count, in_set

    best_blocks = None
    best_len = float('inf')
    
    while time.time() - start_time < time_limit:
        blocks, cover_count, in_set = run_greedy()
        blocks, cover_count, in_set = local_search(blocks, cover_count, in_set)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [all_k_combos[b] for b in blocks]
            
    return best_blocks

def _simple_greedy(v, k, t):
    # Fallback
    all_points = list(range(v))
    all_t_subsets = set(itertools.combinations(all_points, t))
    blocks = []
    cover_count = {sub: 0 for sub in all_t_subsets}
    
    while any(c == 0 for c in cover_count.values()):
        best_block = None
        max_gain = -1
        for block in itertools.combinations(all_points, k):
            gain = 0
            for sub in itertools.combinations(block, t):
                if cover_count[sub] == 0:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = block
        if max_gain == 0:
            break
        blocks.append(list(best_block))
        for sub in itertools.combinations(best_block, t):
            cover_count[sub] += 1
            
    return blocks
