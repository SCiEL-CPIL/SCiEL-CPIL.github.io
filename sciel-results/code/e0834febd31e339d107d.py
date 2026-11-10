import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for evaluator (150s limit)
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)
    
    # Precompute masks for all k-subsets
    masks = [0] * n_k
    coverers = [[] for _ in range(n_t)]
    
    for i, k_sub in enumerate(all_k_subsets):
        m = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_to_idx[t_sub]
            m |= (1 << idx)
            coverers[idx].append(i)
        masks[i] = m
    
    # Convert masks to numpy arrays for fast bit operations where possible
    # But Python ints with bit_count() are actually very fast for this size
    
    def get_blocks(indices):
        return [list(all_k_subsets[i]) for i in indices]
    
    best_solution = None
    best_len = float('inf')
    best_indices = None
    
    # Helper: count covered t-subsets
    def count_covered(selected_set):
        covered = [0] * n_t
        for b in selected_set:
            m = masks[b]
            temp = m
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                covered[idx] += 1
                temp ^= lsb
        return covered
    
    # Local Search: Remove redundant, 1-1 swap, 2-1 swap
    def local_search(initial_indices, seed, time_limit=5.0, max_pairs_21=3000):
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
                continue
            
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
                    break
                
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
                    break
            
            if swapped:
                continue
            
            # Pass 3: 2-1 Swap
            if time.time() - ls_start < time_limit - 0.5:
                list_S = list(S)
                rng.shuffle(list_S)
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
                            break
                            
                        candidates = set(coverers[U[0]])
                        for t_idx in U[1:]:
                            candidates.intersection_update(coverers[t_idx])
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
                            break
                            
                if swapped:
                    continue

        return S

    # Simulated Annealing
    def simulated_annealing(initial_indices, seed, time_limit=10.0):
        rng = random.Random(seed)
        S = set(initial_indices)
        if not S:
            return S
            
        counts = [0] * n_t
        for b in S:
            m = masks[b]
            temp = m
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                temp ^= lsb
        
        current_cost = len(S)
        best_cost = current_cost
        best_S = set(S)
        
        T = 2.0
        T_min = 0.01
        cooling = 0.995
        
        sa_start = time.time()
        
        while T > T_min and (time.time() - sa_start) < time_limit:
            move_type = rng.random()
            
            if move_type < 0.3:
                # Remove a redundant block
                candidates = []
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
                        candidates.append(b)
                
                if candidates:
                    b_out = rng.choice(candidates)
                    S.discard(b_out)
                    m = masks[b_out]
                    temp = m
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        temp ^= lsb
                    new_cost = len(S)
                    delta = new_cost - current_cost
                    if delta <= 0 or rng.random() < np.exp(-delta / T):
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_S = set(S)
                    else:
                        S.add(b_out)
                        temp = m
                        while temp:
                            lsb = temp & -temp
                            idx = lsb.bit_length() - 1
                            counts[idx] += 1
                            temp ^= lsb
            
            elif move_type < 0.7:
                # 1-1 swap
                b_out = rng.choice(list(S))
                m_out = masks[b_out]
                U = []
                temp = m_out
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        U.append(idx)
                    temp ^= lsb
                
                if U:
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
                        
                        new_cost = len(S)
                        delta = new_cost - current_cost
                        if delta <= 0 or rng.random() < np.exp(-delta / T):
                            current_cost = new_cost
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_S = set(S)
                        else:
                            S.discard(b_in)
                            S.add(b_out)
                            temp = m_in
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] -= 1
                                temp ^= lsb
                            temp = m_out
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] += 1
                                temp ^= lsb
                else:
                    # Can remove without adding
                    S.discard(b_out)
                    temp = m_out
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        temp ^= lsb
                    new_cost = len(S)
                    delta = new_cost - current_cost
                    if delta <= 0 or rng.random() < np.exp(-delta / T):
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_S = set(S)
                    else:
                        S.add(b_out)
                        temp = m_out
                        while temp:
                            lsb = temp & -temp
                            idx = lsb.bit_length() - 1
                            counts[idx] += 1
                            temp ^= lsb
            
            else:
                # 2-1 swap
                if len(S) >= 2:
                    b1, b2 = rng.sample(list(S), 2)
                    m1 = masks[b1]
                    m2 = masks[b2]
                    
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
                        new_cost = len(S)
                        delta = new_cost - current_cost
                        if delta <= 0 or rng.random() < np.exp(-delta / T):
                            current_cost = new_cost
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_S = set(S)
                        else:
                            S.add(b1)
                            S.add(b2)
                            temp = m1
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] += 1
                                temp ^= lsb
                            temp = m2
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] += 1
                                temp ^= lsb
                    else:
                        candidates = set(coverers[U[0]])
                        for t_idx in U[1:]:
                            candidates.intersection_update(coverers[t_idx])
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
                            
                            new_cost = len(S)
                            delta = new_cost - current_cost
                            if delta <= 0 or rng.random() < np.exp(-delta / T):
                                current_cost = new_cost
                                if current_cost < best_cost:
                                    best_cost = current_cost
                                    best_S = set(S)
                            else:
                                S.discard(b_in)
                                S.add(b1)
                                S.add(b2)
                                temp = m_in
                                while temp:
                                    lsb = temp & -temp
                                    idx = lsb.bit_length() - 1
                                    counts[idx] -= 1
                                    temp ^= lsb
                                temp = m1
                                while temp:
                                    lsb = temp & -temp
                                    idx = lsb.bit_length() - 1
                                    counts[idx] += 1
                                    temp ^= lsb
                                temp = m2
                                while temp:
                                    lsb = temp & -temp
                                    idx = lsb.bit_length() - 1
                                    counts[idx] += 1
                                    temp ^= lsb
            
            T *= cooling
        
        return list(best_S)
    
    # Run multiple randomized greedy attempts
    attempts = 20
    for attempt in range(attempts):
        if time.time() > deadline - 15:
            break
            
        rng = random.Random(attempt * 12345 + 67890)
        
        current_uncovered = (1 << n_t) - 1
        selected = set()
        
        while current_uncovered:
            best_gain = 0
            best_idx = -1
            best_mask = 0
            
            rem_count = current_uncovered.bit_count()
            if rem_count <= 35:
                for i in range(n_k):
                    if i in selected:
                        continue
                    if (current_uncovered & masks[i]) == current_uncovered:
                        best_gain = rem_count
                        best_idx = i
                        best_mask = masks[i]
                        break
            
            if best_idx == -1:
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
                    
                    if best_gain == rem_count:
                        break
            
            if best_idx == -1:
                break
                
            selected.add(best_idx)
            current_uncovered &= ~best_mask
            
            if time.time() > deadline - 20:
                break
        
        if current_uncovered == 0:
            indices = list(selected)
            
            # Apply Local Search first
            ls_indices = local_search(indices, attempt, time_limit=4.0)
            ls_len = len(ls_indices)
            
            if ls_len < best_len:
                best_len = ls_len
                best_indices = ls_indices
                best_solution = get_blocks(ls_indices)
            
            # Then try Simulated Annealing
            if time.time() < deadline - 20 and ls_len < 85:
                sa_indices = simulated_annealing(ls_indices, attempt + 1000, time_limit=5.0)
                sa_len = len(sa_indices)
                
                if sa_len < best_len:
                    best_len = sa_len
                    best_indices = sa_indices
                    best_solution = get_blocks(sa_indices)
            
            if time.time() > deadline - 20:
                break

    # Final Polish
    if best_indices is not None and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3:
            polished = local_search(best_indices, 99999, time_limit=min(remaining_time * 0.5, 12.0), max_pairs_21=5000)
            if len(polished) < best_len:
                best_len = len(polished)
                best_indices = polished
                best_solution = get_blocks(polished)
            
            if time.time() < deadline - 3:
                remaining2 = deadline - 3 - time.time()
                if remaining2 > 2:
                    polished2 = simulated_annealing(best_indices, 123456, time_limit=min(remaining2, 8.0))
                    if len(polished2) < best_len:
                        best_len = len(polished2)
                        best_indices = polished2
                        best_solution = get_blocks(polished2)

    if best_solution is None:
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
