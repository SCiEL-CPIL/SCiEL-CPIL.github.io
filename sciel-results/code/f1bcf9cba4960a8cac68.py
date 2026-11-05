import itertools
import random
import sys

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute masks for each k-block (which t-subsets it covers)
    # Also precompute list of t-subset indices for each block for fast iteration
    block_t_indices = []
    for block in k_blocks:
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(sub_to_idx[t_sub])
        block_t_indices.append(indices)
        
    # Precompute for each t-subset, which blocks cover it
    # sub_to_blocks[s] = list of block indices
    sub_to_blocks = [[] for _ in range(n_t_subs)]
    for i, indices in enumerate(block_t_indices):
        for s in indices:
            sub_to_blocks[s].append(i)

    # 3. Greedy Set Cover for Initialization
    uncovered = [1] * n_t_subs
    selected = set()
    
    # Calculate initial gains
    # We can just do standard greedy
    # To optimize, we can sort blocks by size of coverage (all same) or random
    # Standard greedy: pick block with max new coverage
    
    # For speed, we'll use a simple greedy without complex data structures first
    # Since n_k_blocks is 75k, and we pick ~335 blocks, 335 * 75k = 25M ops.
    # Each op: count how many uncovered subsets are in block.
    # We can maintain a global 'uncovered' set or array.
    
    # Optimization: Use bitsets for uncovered?
    # n_t_subs = 11628. Bitset is 11628 bits.
    # Python int is good for this.
    uncovered_mask = (1 << n_t_subs) - 1
    
    # Precompute block masks as ints for fast AND
    block_masks = [0] * n_k_blocks
    for i, indices in enumerate(block_t_indices):
        mask = 0
        for s in indices:
            mask |= (1 << s)
        block_masks[i] = mask
            
    while uncovered_mask:
        best_idx = -1
        max_gain = 0
        # Randomize tie-breaking to help local search later? 
        # Or just pick first max.
        for i in range(n_k_blocks):
            gain = (block_masks[i] & uncovered_mask).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
                if gain == n_t_subs: # Perfect block? Unlikely
                    break
        if best_idx == -1:
            break
        selected.add(best_idx)
        uncovered_mask &= ~block_masks[best_idx]
        
    # 4. Local Search
    # State: selected (set of block indices)
    # cover_count: array of length n_t_subs
    
    def update_cover_counts(selected_set):
        counts = [0] * n_t_subs
        for b in selected_set:
            for s in block_t_indices[b]:
                counts[s] += 1
        return counts
        
    cover_count = update_cover_counts(selected)
    
    def get_unique_subsets(b, counts):
        uniques = []
        for s in block_t_indices[b]:
            if counts[s] == 1:
                uniques.append(s)
        return uniques
        
    def try_remove_redundant(selected_set, counts):
        removed = False
        for b in list(selected_set):
            uniques = get_unique_subsets(b, counts)
            if not uniques:
                # Redundant
                selected_set.remove(b)
                for s in block_t_indices[b]:
                    counts[s] -= 1
                removed = True
                # Restart loop to check again, as removing b might make others redundant
                # But for efficiency, we can just continue, though restart is safer
                # Let's break and restart the outer loop for simplicity in logic
                # Actually, just returning True and letting the main loop handle restart is good
                return True
        return False
        
    def try_1_1_swap(selected_set, counts):
        # Try to find b in S and b' not in S such that Unique(b) subset Covers(b')
        # Iterate through blocks in S
        for b in list(selected_set):
            uniques = get_unique_subsets(b, counts)
            if not uniques:
                continue # Should have been removed, but just in case
            
            # Find candidates b' covering all uniques
            # Intersect lists of blocks covering each unique subset
            # Start with blocks covering the first unique subset
            candidates = set(sub_to_blocks[uniques[0]])
            for s in uniques[1:]:
                candidates &= set(sub_to_blocks[s])
            
            # Filter out blocks already in selected_set
            for b_prime in candidates:
                if b_prime not in selected_set:
                    # Valid swap
                    # Remove b, Add b_prime
                    selected_set.remove(b)
                    for s in block_t_indices[b]:
                        counts[s] -= 1
                    selected_set.add(b_prime)
                    for s in block_t_indices[b_prime]:
                        counts[s] += 1
                    return True
        return False
        
    def try_2_2_swap(selected_set, counts):
        # Try to find b1, b2 in S and b3, b4 not in S
        # such that Unique(b1) U Unique(b2) subset Covers(b3) U Covers(b4)
        
        # Heuristic: Pick b1 with smallest non-empty Unique set
        # Pick b2 with smallest non-empty Unique set
        # If time is tight, limit iterations
        
        blocks_in_s = list(selected_set)
        if len(blocks_in_s) < 2:
            return False
            
        # Sort blocks by size of Unique set (smallest first) to find conflicts easily?
        # Or just random
        random.shuffle(blocks_in_s)
        
        # Limit number of pairs to try to avoid timeout
        max_pairs_to_try = 100
        
        pairs_tried = 0
        for i in range(len(blocks_in_s)):
            for j in range(i+1, len(blocks_in_s)):
                b1 = blocks_in_s[i]
                b2 = blocks_in_s[j]
                
                u1 = get_unique_subsets(b1, counts)
                u2 = get_unique_subsets(b2, counts)
                
                # If either is empty, it should have been removed.
                if not u1 or not u2:
                    continue
                    
                target = set(u1) | set(u2)
                
                # We need b3, b4 not in S covering target.
                # Strategy: Pick a random s in target.
                # Candidates for b3 must cover s.
                # Candidates for b4 must cover target \ Covers(b3).
                
                # To keep it fast, we can try a few random s from target
                # Or just pick the first one
                
                # Optimization: 
                # If target is small, we can brute force pairs of candidates?
                # No, candidates can be large.
                
                # Let's pick one specific s from target to anchor b3
                # Pick s with fewest candidates?
                # For simplicity, pick random
                s_anchor = random.choice(list(target))
                cand_b3 = [b for b in sub_to_blocks[s_anchor] if b not in selected_set]
                
                if not cand_b3:
                    continue
                    
                # Shuffle candidates to try different ones
                random.shuffle(cand_b3)
                
                # Limit cand_b3 iterations
                for b3 in cand_b3[:10]: 
                    remaining = target - set(block_t_indices[b3])
                    if not remaining:
                        # b3 covers everything. This is a 1-1 swap for b1 or b2?
                        # If b3 covers u1, 1-1 for b1 should have worked.
                        # If b3 covers u2, 1-1 for b2 should have worked.
                        # If b3 covers u1 U u2, it covers u1, so 1-1 for b1 works.
                        # So this case should be handled by 1-1. 
                        # But if 1-1 failed, this shouldn't happen.
                        # Unless b3 == b1 or b2? No, b3 not in S.
                        # So skip.
                        continue
                        
                    # Need b4 to cover remaining
                    # Pick an anchor s_rem from remaining
                    s_rem = random.choice(list(remaining))
                    cand_b4 = [b for b in sub_to_blocks[s_rem] if b not in selected_set and b != b3]
                    
                    if not cand_b4:
                        continue
                        
                    # Check if any b4 covers all remaining
                    found_b4 = None
                    for b4 in cand_b4[:10]:
                        if remaining.issubset(set(block_t_indices[b4])):
                            found_b4 = b4
                            break
                            
                    if found_b4:
                        # Perform swap
                        # Remove b1, b2
                        selected_set.remove(b1)
                        selected_set.remove(b2)
                        for s in block_t_indices[b1]:
                            counts[s] -= 1
                        for s in block_t_indices[b2]:
                            counts[s] -= 1
                            
                        # Add b3, b4
                        selected_set.add(b3)
                        selected_set.add(found_b4)
                        for s in block_t_indices[b3]:
                            counts[s] += 1
                        for s in block_t_indices[found_b4]:
                            counts[s] += 1
                            
                        return True
                        
                pairs_tried += 1
                if pairs_tried > max_pairs_to_try:
                    break
            if pairs_tried > max_pairs_to_try:
                break
                
        return False

    # Main Local Search Loop
    # We run this for a fixed number of iterations or until no improvement
    # Since we want to minimize, and we start with a good greedy, 
    # we just need to run until stuck.
    
    improved = True
    while improved:
        improved = False
        
        # Phase 1: Remove redundant
        if try_remove_redundant(selected, cover_count):
            improved = True
            # After removal, cover_count is updated
            continue # Restart loop
            
        # Phase 2: 1-1 Swap
        if try_1_1_swap(selected, cover_count):
            improved = True
            continue
            
        # Phase 3: 2-2 Swap
        if try_2_2_swap(selected, cover_count):
            improved = True
            continue
            
    # Final check: ensure all covered
    # The local search maintains validity, so this should be fine.
    
    return [list(k_blocks[i]) for i in selected]
