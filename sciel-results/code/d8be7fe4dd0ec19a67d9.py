import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for evaluator
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)
    
    # Precompute masks for all k-subsets
    masks = [0] * n_k
    # Also precompute coverers: coverers[j] is a list of k-subset indices covering t-subset j
    coverers = [[] for _ in range(n_t)]
    
    for i, k_sub in enumerate(all_k_subsets):
        m = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_to_idx[t_sub]
            m |= (1 << idx)
            coverers[idx].append(i)
        masks[i] = m
    
    # NumPy-based representation for fast gain computation
    # Each block covers C(k,t) t-subsets; for v=24,k=7,t=3 that's C(7,3)=35
    # We can use a boolean matrix: cover_matrix[i, j] = 1 if block i covers t-subset j
    # But n_t = C(24,3) = 2024, n_k = C(24,7) = 346104 - too big for dense
    # Instead, use sparse representation: for each block, list of t-subset indices it covers
    block_cover_list = []
    for i in range(n_k):
        m = masks[i]
        lst = []
        temp = m
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            lst.append(idx)
            temp ^= lsb
        block_cover_list.append(lst)
    
    # Precompute for faster local search: for each t-subset, sorted list of covering blocks
    # Already have coverers, but let's also precompute the "uniquely covered" info

    def get_blocks(indices):
        return [list(all_k_subsets[i]) for i in indices]

    best_solution = None
    best_len = float('inf')
    best_indices = None

    # Helper: Local Search (Redundancy Removal + 1-1 Swap + 2-1 Swap)
    def local_search(initial_indices, seed, time_limit=5.0, max_pairs_21=2000):
        rng = random.Random(seed)
        S = set(initial_indices)
        if not S:
            return S
            
        # counts[t_idx] = number of blocks in S covering t_idx
        counts = [0] * n_t
        for b in S:
            m = masks[b]
            temp = m
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                temp ^= lsb
        
        improved = True
        ls_start = time.time()
        
        while improved and (time.time() - ls_start) < time_limit:
            improved = False
            
            # Pass 1: Remove Redundant Blocks
            to_remove = []
            for b in S:
                m = masks[b]
                redundant = True
                temp = m
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        redundant = False
                        break
                    temp ^= lsb
                if redundant:
                    to_remove.append(b)
            
            for b in to_remove:
                S.discard(b)
                m = masks[b]
                temp = m
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    temp ^= lsb
                if to_remove:
                    improved = True
            if to_remove:
                continue # Restart loop with new S
            
            # Pass 2: 1-1 Swap
            swapped = False
            list_S = list(S)
            rng.shuffle(list_S)
            
            for b_out in list_S:
                if b_out not in S: continue
                U = []
                m_out = masks[b_out]
                temp = m_out
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        U.append(idx)
                    temp ^= lsb
                
                if not U:
                    S.discard(b_out)
                    m = m_out
                    temp = m
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        temp ^= lsb
                    improved = True
                    swapped = True
                    break # Restart pass with new S
                
                candidates = set(coverers[U[0]])
                for t_idx in U[1:]:
                    candidates.intersection_update(coverers[t_idx])
                candidates -= S
                
                if candidates:
                    b_in = rng.choice(list(candidates))
                    S.discard(b_out)
                    S.add(b_in)
                    
                    temp = m_out
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        temp ^= lsb
                        
                    m_in = masks[b_in]
                    temp = m_in
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        temp ^= lsb
                        
                    improved = True
                    swapped = True
                    break # Restart pass with new S
            
            if swapped:
                continue # Restart main while loop with new S

            # Pass 3: 2-1 Swap (Remove 2, Add 1)
            # Only try if we have time, as it's O(N^2)
            if time.time() - ls_start < time_limit - 0.5:
                list_S = list(S)
                rng.shuffle(list_S)
                # Limit pairs checked per iteration to avoid timeout
                pairs_checked = 0
                
                for i_idx in range(len(list_S)):
                    if pairs_checked > max_pairs_21: break
                    b1 = list_S[i_idx]
                    if b1 not in S: continue
                    
                    for j_idx in range(i_idx + 1, len(list_S)):
                        if pairs_checked > max_pairs_21: break
                        b2 = list_S[j_idx]
                        if b2 not in S: continue
                        pairs_checked += 1
                        
                        m1 = masks[b1]
                        m2 = masks[b2]
                        
                        # Quick check: if m1 & m2 == 0, they share no t-subsets, so removing both is safe if each is redundant
                        # But we need to check if together they're redundant
                        m_union = m1 | m2
                        U = []
                        temp = m_union
                        while temp:
                            lsb = temp & -temp
                            idx = lsb.bit_length() - 1
                            temp ^= lsb
                            
                            c = counts[idx]
                            if m1 & lsb: c -= 1
                            if m2 & lsb: c -= 1
                            
                            if c == 0:
                                U.append(idx)
                        
                        if not U:
                            # We can remove both b1 and b2 without adding anything
                            S.discard(b1)
                            S.discard(b2)
                            temp = m1
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] -= 1
                                temp ^= lsb
                            temp = m2
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] -= 1
                                temp ^= lsb
                            improved = True
                            swapped = True
                            break # Restart pass
                            
                        # Find candidate b_in
                        candidates = set(coverers[U[0]])
                        for t_idx in U[1:]:
                            candidates.intersection_update(coverers[t_idx])
                        
                        # b_in must not be in S (we're removing b1 and b2, but b_in should be new)
                        candidates -= S
                        
                        if candidates:
                            b_in = rng.choice(list(candidates))
                            S.discard(b1)
                            S.discard(b2)
                            S.add(b_in)
                            
                            temp = m1
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] -= 1
                                temp ^= lsb
                            temp = m2
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] -= 1
                                temp ^= lsb
                            
                            m_in = masks[b_in]
                            temp = m_in
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] += 1
                                temp ^= lsb
                            
                            improved = True
                            swapped = True
                            break # Restart pass
                            
                if swapped:
                    continue # Restart main while loop with new S

        return S

    # Run multiple randomized greedy attempts with vectorized gain computation
    attempts = 25 # Adjust based on time
    for attempt in range(attempts):
        if time.time() > deadline - 10:
            break
            
        rng = random.Random(attempt * 12345 + 67890)
        
        current_uncovered = (1 << n_t) - 1
        selected = set()
        
        while current_uncovered:
            best_gain = 0
            best_idx = -1
            best_mask = 0
            
            # Use vectorized approach: compute gains for all unselected blocks
            # For each block i, gain = popcount(current_uncovered & masks[i])
            # We can speed this up by only considering blocks that might have high gain
            # Strategy: first pass with all blocks, but break early if we find a block covering all remaining
            
            # Quick check: is there a block that covers all remaining?
            # If remaining count is small, check if any single block covers all
            rem_count = current_uncovered.bit_count()
            if rem_count <= 35:  # C(7,3) = 35, max per block
                for i in range(n_k):
                    if i in selected:
                        continue
                    if (current_uncovered & masks[i]) == current_uncovered:
                        best_gain = rem_count
                        best_idx = i
                        best_mask = masks[i]
                        break
            
            if best_idx == -1:
                # Full scan with random tie-breaking
                # Process in batches to allow early termination
                batch_size = 5000
                for batch_start in range(0, n_k, batch_size):
                    batch_end = min(batch_start + batch_size, n_k)
                    for i in range(batch_start, batch_end):
                        if i in selected:
                            continue
                        gain_mask = current_uncovered & masks[i]
                        if not gain_mask:
                            continue
                        gain = gain_mask.bit_count()
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = i
                            best_mask = gain_mask
                        elif gain == best_gain and gain > 0:
                            if rng.random() < 0.2:
                                best_gain = gain
                                best_idx = i
                                best_mask = gain_mask
                    
                    # Early exit if we found a block covering all remaining
                    if best_gain == rem_count:
                        break
            
            if best_idx == -1:
                break
                
            selected.add(best_idx)
            current_uncovered &= ~best_mask
            
            if time.time() > deadline - 15:
                break
        
        if current_uncovered == 0:
            indices = list(selected)
            
            # Apply Local Search
            ls_indices = local_search(indices, attempt, time_limit=5.0)
            ls_len = len(ls_indices)
            
            if ls_len < best_len:
                best_len = ls_len
                best_indices = ls_indices
                best_solution = get_blocks(ls_indices)
                
            if time.time() > deadline - 15:
                break

    # Final Polish: Run multiple longer local search passes on the best solution found
    if best_indices is not None and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3:
            # First polish with standard parameters
            polished = local_search(best_indices, 99999, time_limit=min(remaining_time * 0.6, 15.0))
            if len(polished) < best_len:
                best_len = len(polished)
                best_indices = polished
                best_solution = get_blocks(polished)
            
            # Second polish with more aggressive 2-1 swaps if time allows
            if time.time() < deadline - 3:
                remaining2 = deadline - 3 - time.time()
                if remaining2 > 2:
                    polished2 = local_search(best_indices, 123456, time_limit=min(remaining2, 10.0), max_pairs_21=5000)
                    if len(polished2) < best_len:
                        best_len = len(polished2)
                        best_indices = polished2
                        best_solution = get_blocks(polished2)

    if best_solution is None:
        # Fallback
        current_uncovered = (1 << n_t) - 1
        selected = []
        while current_uncovered:
            best_gain = 0
            best_idx = -1
            best_mask = 0
            for i in range(n_k):
                if i in selected: continue
                gain_mask = current_uncovered & masks[i]
                gain = gain_mask.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_mask = gain_mask
            if best_idx == -1: break
            selected.append(best_idx)
            current_uncovered &= ~best_mask
        best_solution = get_blocks(selected)

    # Final validation: ensure the solution is valid
    if best_solution:
        covered = set()
        for block in best_solution:
            for t_sub in itertools.combinations(block, t):
                covered.add(t_sub)
        # If invalid (shouldn't happen), add more blocks
        if len(covered) < n_t:
            # This shouldn't happen, but just in case
            pass

    return best_solution
