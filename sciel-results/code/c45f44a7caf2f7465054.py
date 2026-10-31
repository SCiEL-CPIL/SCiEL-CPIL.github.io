import time
import random
import itertools
import sys

# Global cache for precomputed data
_PREPARE_CACHE = {}

def _prepare(v, k, t):
    key = (v, k, t)
    if key in _PREPARE_CACHE:
        return _PREPARE_CACHE[key]
    
    # Generate all t-subsets
    t_subs = [tuple(c) for c in itertools.combinations(range(v), t)]
    n_t = len(t_subs)
    index = {sub: i for i, sub in enumerate(t_subs)}
    
    # Generate all k-blocks
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # Precompute masks for all k-blocks
    masks = []
    # Also build an index: subset_index -> list of block_indices
    sub_to_blocks = [[] for _ in range(n_t)]
    
    for b_idx, b in enumerate(blocks):
        m = 0
        for sub in itertools.combinations(b, t):
            s_idx = index[sub]
            m |= (1 << s_idx)
            sub_to_blocks[s_idx].append(b_idx)
        masks.append(m)
        
    _PREPARE_CACHE[key] = (blocks, masks, n_t, index, sub_to_blocks, n_blocks)
    return _PREPARE_CACHE[key]

def _greedy_construct(masks, n_blocks, n_t, rng, deadline, randomization_factor=0.1):
    """
    Greedy construction with randomization.
    At each step, instead of picking the absolute best block, pick from the top 
    'randomization_factor' fraction of candidates.
    """
    uncovered = (1 << n_t) - 1
    selected = []
    selected_set = set()
    
    # Precompute list of block indices for faster access
    all_indices = list(range(n_blocks))
    
    while uncovered and time.time() < deadline:
        # Calculate scores for all non-selected blocks
        # Optimization: We can skip this if we maintain a priority queue, 
        # but with n_blocks ~ 50k and few iterations, a linear scan is okay if optimized.
        
        scores = []
        for i in all_indices:
            if i in selected_set:
                continue
            m = masks[i]
            new_cov = (m & uncovered).bit_count()
            if new_cov > 0:
                scores.append((new_cov, i))
        
        if not scores:
            break
            
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Determine the threshold for the "elite" set
        # If randomization_factor is 0.1, we pick from the top 10% of scores (or top 10 if small)
        # To be robust, we pick from the top K where K is max(1, int(len(scores) * randomization_factor))
        # But usually, just picking from the top few distinct scores is better.
        # Let's pick from the top 5% or at least top 5 candidates.
        k_elite = max(5, int(len(scores) * randomization_factor))
        elite = scores[:k_elite]
        
        # Randomly pick one from the elite set
        best_cov, best_idx = rng.choice(elite)
        
        selected.append(best_idx)
        selected_set.add(best_idx)
        
        # Update uncovered
        uncovered &= ~masks[best_idx]
        
    return selected

def _local_search(selected, masks, n_blocks, n_t, sub_to_blocks, deadline, rng):
    """
    Local search with 1-for-1 swaps and 2-for-1 swaps.
    """
    # selected is list of indices
    # We maintain counts for each t-subset
    counts = [0] * n_t
    selected_set = set(selected)
    
    # Initialize counts
    for i in selected:
        m = masks[i]
        mm = m
        while mm:
            lb = mm & (-mm)
            idx = lb.bit_length() - 1
            counts[idx] += 1
            mm &= mm -  1
            
    improved = True
    # To prevent infinite loops on neutral moves, we limit the number of iterations
    max_iterations = 1000
    
    while improved and time.time() < deadline and max_iterations > 0:
        improved = False
        max_iterations -= 1
        
        # 1. Try deletions
        to_remove = -1
        for pos, i in enumerate(selected):
            m = masks[i]
            can_remove = True
            mm = m
            while mm:
                lb = mm & (-mm)
                idx = lb.bit_length() - 1
                if counts[idx] == 1:
                    can_remove = False
                    break
                mm &= mm - 1
            if can_remove:
                to_remove = pos
                break
        
        if to_remove != -1:
            i = selected[to_remove]
            m = masks[i]
            mm = m
            while mm:
                lb = mm & (-mm)
                idx = lb.bit_length() - 1
                counts[idx] -= 1
                mm &= mm -  1
            selected.pop(to_remove)
            selected_set.discard(i)
            improved = True
            continue
        
        # 2. Try 1-for-1 Swaps
        swap_found = False
        # Randomize the order of checking blocks to avoid bias
        indices_to_check = list(range(len(selected)))
        rng.shuffle(indices_to_check)
        
        for pos in indices_to_check:
            i = selected[pos]
            m_i = masks[i]
            unique_mask = 0
            unique_indices = []
            mm = m_i
            while mm:
                lb = mm & (-mm)
                idx = lb.bit_length() - 1
                if counts[idx] == 1:
                    unique_mask |= lb
                    unique_indices.append(idx)
                mm &= mm - 1
                
            if not unique_indices:
                # Should have been removed
                continue

            # Find a block j not in selected that covers all unique_indices
            # Use sub_to_blocks to find candidates
            # Start with the rarest unique subset
            min_len = float('inf')
            start_idx = unique_indices[0]
            for u_idx in unique_indices:
                l_len = len(sub_to_blocks[u_idx])
                if l_len < min_len:
                    min_len = l_len
                    start_idx = u_idx
            
            candidate_list = sub_to_blocks[start_idx]
            
            # Filter candidates
            valid_candidates = []
            for j in candidate_list:
                if j in selected_set:
                    continue
                m_j = masks[j]
                ok = True
                for u_idx in unique_indices:
                    if not (m_j & (1 << u_idx)):
                        ok = False
                        break
                if ok:
                    valid_candidates.append(j)
            
            if valid_candidates:
                # Pick a random valid candidate
                j = rng.choice(valid_candidates)
                
                # Perform swap: remove i, add j
                mm = m_i
                while mm:
                    lb = mm & (-mm)
                    idx = lb.bit_length() - 1
                    counts[idx] -= 1
                    mm &= mm - 1
                selected_set.discard(i)
                
                m_j = masks[j]
                mm = m_j
                while mm:
                    lb = mm & (-mm)
                    idx = lb.bit_length() - 1
                    counts[idx] += 1
                    mm &= mm - 1
                selected_set.add(j)
                
                selected[pos] = j
                swap_found = True
                improved = True
                break # Restart loop after swap
        
        if swap_found:
            continue
            
        # 3. Try 2-for-1 Swaps
        # Replace two blocks (i, k) with one block (j)
        # Condition: j covers all unique subsets of i and k.
        # This reduces the count by 1.
        
        # To keep it fast, we only try a limited number of pairs.
        # We look for pairs of blocks that have "overlapping" unique sets? 
        # Actually, j must cover Unique(i) U Unique(k).
        
        # Heuristic: Pick two random blocks from selected.
        # If len(selected) > 2, try a few random pairs.
        if len(selected) > 2:
            pairs_tried = 0
            max_pairs = 20 # Limit work per iteration
            
            while pairs_tried < max_pairs and time.time() < deadline:
                # Pick two random distinct indices
                p1 = rng.randint(0, len(selected) - 1)
                p2 = rng.randint(0, len(selected) - 1)
                if p1 == p2:
                    continue
                pairs_tried += 1
                
                i = selected[p1]
                k = selected[p2]
                
                # Calculate unique sets for i and k
                # Note: If i and k share a subset, and that subset is unique to i, 
                # it's still unique to the pair {i,k} only if no other block covers it.
                # The condition for 2-for-1 is: 
                # Let U_i = {s in i | count[s]==1}
                # Let U_k = {s in k | count[s]==1}
                # Let U_pair = U_i U U_k
                # We need a block j not in selected \ {i,k} such that j covers U_pair.
                # Also, we must ensure that removing i and k doesn't uncover anything else.
                # Since U_i contains all subsets of i that are ONLY covered by i, 
                # and U_k contains all subsets of k that are ONLY covered by k,
                # if j covers U_i U U_k, then all subsets of i and k are still covered 
                # (either by j, or by other blocks that already covered them).
                
                u_i = []
                mm = masks[i]
                while mm:
                    lb = mm & (-mm)
                    idx = lb.bit_length() - 1
                    if counts[idx] == 1:
                        u_i.append(idx)
                    mm &= mm - 1
                
                u_k = []
                mm = masks[k]
                while mm:
                    lb = mm & (-mm)
                    idx = lb.bit_length() - 1
                    if counts[idx] == 1:
                        u_k.append(idx)
                    mm &= mm - 1
                
                # Combine unique indices
                # We need to cover all indices in u_i and u_k.
                # Note: An index cannot be in both u_i and u_k because 
                # if count[s]==1, it is covered by exactly one block. 
                # If it's in i, it's not in k (unless i==k, which we excluded).
                # So u_i and u_k are disjoint.
                
                if not u_i and not u_k:
                    # Both are redundant? Step 1 should have caught this.
                    # But if counts changed, maybe.
                    # If both have no unique subsets, we can remove one.
                    # Step 1 handles single removals.
                    continue
                
                # Find a block j that covers all indices in u_i and u_k.
                # Use the rarest subset from the union to find candidates.
                all_unique = u_i + u_k
                if not all_unique:
                    continue
                
                min_len = float('inf')
                start_idx = all_unique[0]
                for idx in all_unique:
                    l_len = len(sub_to_blocks[idx])
                    if l_len < min_len:
                        min_len = l_len
                        start_idx = idx
                        
                candidate_list = sub_to_blocks[start_idx]
                
                found_j = -1
                for j in candidate_list:
                    if j in selected_set:
                        # j must not be i or k either, but i,k are in selected_set.
                        # If j==i or j==k, it's in selected_set, so skipped.
                        continue
                    m_j = masks[j]
                    ok = True
                    for idx in all_unique:
                        if not (m_j & (1 << idx)):
                            ok = False
                            break
                    if ok:
                        found_j = j
                        break
                
                if found_j != -1:
                    # Perform 2-for-1 swap: remove i and k, add j
                    # Remove i
                    mm = masks[i]
                    while mm:
                        lb = mm & (-mm)
                        idx = lb.bit_length() - 1
                        counts[idx] -= 1
                        mm &= mm - 1
                    selected_set.discard(i)
                    
                    # Remove k
                    mm = masks[k]
                    while mm:
                        lb = mm & (-mm)
                        idx = lb.bit_length() - 1
                        counts[idx] -= 1
                        mm &= mm - 1
                    selected_set.discard(k)
                    
                    # Add j
                    m_j = masks[found_j]
                    mm = m_j
                    while mm:
                        lb = mm & (-mm)
                        idx = lb.bit_length() - 1
                        counts[idx] += 1
                        mm &= mm - 1
                    selected_set.add(found_j)
                    
                    # Update selected list
                    # Remove i and k, add j
                    # To keep indices stable, we can just pop and append, or replace.
                    # Since order doesn't matter for validity, we can just modify the list.
                    # Be careful with indices if we pop.
                    # Let's create a new list to avoid index shifting issues during iteration (though we break out).
                    new_selected = []
                    for idx_sel in selected:
                        if idx_sel != i and idx_sel != k:
                            new_selected.append(idx_sel)
                    new_selected.append(found_j)
                    selected = new_selected
                    # Update selected_set is already done
                    
                    improved = True
                    break # Restart loop after successful 2-for-1 swap
            
            if improved:
                continue

    return selected

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0 # Safety margin
    
    blocks, masks, n_t, index, sub_to_blocks, n_blocks = _prepare(v, k, t)
    
    best_size = float('inf')
    best_selected = []
    
    # Iterated Local Search
    # We use multiple restarts with different random seeds and randomization factors
    num_restarts = 40 # Increased restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        rng = random.Random(42 + r)
        
        # Vary randomization factor
        # Early restarts: less random (greedy)
        # Later restarts: more random (diverse)
        rand_factor = 0.05 + 0.15 * (r / num_restarts)
        
        # 1. Greedy Construction
        sel = _greedy_construct(masks, n_blocks, n_t, rng, deadline - 5.0, rand_factor)
        
        # 2. Local Search
        sel = _local_search(sel, masks, n_blocks, n_t, sub_to_blocks, deadline - 5.0, rng)
        
        if len(sel) < best_size:
            best_size = len(sel)
            best_selected = list(sel)
            
        # Optional: If we have time, perturb the best solution and run LS again?
        # For now, fresh restarts with varying randomness are effective.
        
    # Final validity check and fallback
    if best_selected:
        covered = 0
        for i in best_selected:
            covered |= masks[i]
        
        if covered != (1 << n_t) - 1:
            # Fallback: add any missing blocks greedily
            uncovered = ((1 << n_t) - 1) & ~covered
            selected_set = set(best_selected)
            for i in range(n_blocks):
                if i in selected_set:
                    continue
                if masks[i] & uncovered:
                    best_selected.append(i)
                    selected_set.add(i)
                    uncovered &= ~masks[i]
                    if not uncovered:
                        break

    # Return list of blocks (as lists of ints)
    result = [list(blocks[i]) for i in best_selected]
    return result
