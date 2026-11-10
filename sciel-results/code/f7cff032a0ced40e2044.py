import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for evaluator (150s limit)
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)
    
    # Precompute masks for all k-subsets
    # mask[i] is a bitmask of t-subsets covered by k-subset i
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
    
    def get_blocks(indices):
        return [list(all_k_subsets[i]) for i in indices]

    best_solution = None
    best_len = float('inf')
    best_indices = None

    # Helper: Local Search (Redundancy Removal + 1-1 Swap + 2-1 Swap + 2-2 Swap)
    def local_search(initial_indices, seed, time_limit=5.0, max_pairs_21=2000, max_pairs_22=500):
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

            # Pass 4: 2-2 Swap (Remove 2, Add 2) - can escape deeper local minima
            if time.time() - ls_start < time_limit - 1.0:
                list_S = list(S)
                rng.shuffle(list_S)
                pairs_checked_22 = 0
                
                for i_idx in range(len(list_S)):
                    if pairs_checked_22 > max_pairs_22: break
                    b1 = list_S[i_idx]
                    if b1 not in S: continue
                    
                    for j_idx in range(i_idx + 1, len(list_S)):
                        if pairs_checked_22 > max_pairs_22: break
                        b2 = list_S[j_idx]
                        if b2 not in S: continue
                        pairs_checked_22 += 1
                        
                        m1 = masks[b1]
                        m2 = masks[b2]
                        
                        # Compute what becomes uncovered when we remove b1 and b2
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
                        
                        # Find two candidates that together cover all of U
                        # First, find candidate that covers most of U
                        if len(U) > 1:
                            # Find best first candidate
                            best_c1 = -1
                            best_c1_gain = 0
                            for c1 in coverers[U[0]]:
                                if c1 in S and c1 != b1 and c1 != b2:
                                    continue
                                gain = 0
                                for u_idx in U:
                                    if masks[c1] & (1 << u_idx):
                                        gain += 1
                                if gain > best_c1_gain:
                                    best_c1_gain = gain
                                    best_c1 = c1
                            if best_c1 == -1 or best_c1_gain < len(U) // 2:
                                continue
                            
                            # Now find best second candidate to cover remaining
                            remaining = [u for u in U if not (masks[best_c1] & (1 << u))]
                            if not remaining:
                                # One block covers everything, but we need 2 blocks for 2-2 swap
                                # Actually, we can just add one block and be done (that's a 2-1 swap)
                                # For 2-2, we need exactly 2 new blocks
                                # Let's just pick a random second block that doesn't add redundancy
                                all_candidates = [c for c in coverers[U[0]] if c not in S and c != best_c1]
                                if all_candidates:
                                    best_c2 = rng.choice(all_candidates)
                                else:
                                    continue
                            else:
                                best_c2 = -1
                                best_c2_gain = 0
                                for c2 in coverers[remaining[0]]:
                                    if c2 in S and c2 != b1 and c2 != b2:
                                        continue
                                    if c2 == best_c1:
                                        continue
                                    gain = 0
                                    for u_idx in remaining:
                                        if masks[c2] & (1 << u_idx):
                                            gain += 1
                                    if gain > best_c2_gain:
                                        best_c2_gain = gain
                                        best_c2 = c2
                                if best_c2 == -1:
                                    continue
                            
                            # Verify that {best_c1, best_c2} together cover all of U
                            covered = set()
                            for u_idx in U:
                                if masks[best_c1] & (1 << u_idx):
                                    covered.add(u_idx)
                                if masks[best_c2] & (1 << u_idx):
                                    covered.add(u_idx)
                            if len(covered) != len(U):
                                continue
                            
                            # Perform the swap
                            S.discard(b1)
                            S.discard(b2)
                            S.add(best_c1)
                            S.add(best_c2)
                            
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
                            
                            for c_new in [best_c1, best_c2]:
                                m_new = masks[c_new]
                                temp = m_new
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
    # Strategy:
    # 1. Standard Greedy
    # 2. Greedy with Random Tie-Breaking (more diverse)
    # 3. "Conflict-Aware" Greedy: When multiple blocks have the same gain, prefer one that shares fewer t-subsets with already selected blocks (reduces future redundancy potential)
    
    attempts = 30
    for attempt in range(attempts):
        if time.time() > deadline - 10:
            break
            
        rng = random.Random(attempt * 12345 + 67890)
        
        current_uncovered = (1 << n_t) - 1
        selected = set()
        
        # Track the OR of all selected masks for conflict-aware tie-breaking
        selected_union = 0
        
        while current_uncovered:
            best_gain = 0
            best_idx = -1
            best_mask = 0
            best_overlap = float('inf')
            
            rem_count = current_uncovered.bit_count()
            if rem_count <= 35:  # C(7,3) = 35
                for i in range(n_k):
                    if i in selected:
                        continue
                    if (current_uncovered & masks[i]) == current_uncovered:
                        best_gain = rem_count
                        best_idx = i
                        best_mask = masks[i]
                        break
            
            if best_idx == -1:
                # Full scan with conflict-aware tie-breaking
                for i in range(n_k):
                    if i in selected:
                        continue
                    gain_mask = current_uncovered & masks[i]
                    if not gain_mask:
                        continue
                    gain = gain_mask.bit_count()
                    
                    # Compute overlap with already selected blocks (for tie-breaking)
                    overlap = (masks[i] & selected_union).bit_count()
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                        best_mask = gain_mask
                        best_overlap = overlap
                    elif gain == best_gain and gain > 0:
                        # Prefer lower overlap, with random tie-breaking
                        if overlap < best_overlap or (overlap == best_overlap and rng.random() < 0.3):
                            best_gain = gain
                            best_idx = i
                            best_mask = gain_mask
                            best_overlap = overlap
                
                # Early exit check
                #if best_gain == rem_count: break

            if best_idx == -1:
                break
                
            selected.add(best_idx)
            current_uncovered &= ~best_mask
            selected_union |= masks[best_idx]
            
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
            
            # Third polish with 2-2 swaps if time allows
            if time.time() < deadline - 2:
                remaining3 = deadline - 2 - time.time()
                if remaining3 > 2:
                    polished3 = local_search(best_indices, 777777, time_limit=min(remaining3, 8.0), max_pairs_21=2000, max_pairs_22=2000)
                    if len(polished3) < best_len:
                        best_len = len(polished3)
                        best_indices = polished3
                        best_solution = get_blocks(polished3)

    if best_solution is None:
        # Fallback: Simple deterministic greedy
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

    return best_solution
