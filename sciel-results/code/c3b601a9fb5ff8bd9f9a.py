"""
Combined approach: Binary carry-free structural construction + efficient scoring + local search refinement.
Key insight: Elements with disjoint binary supports prevent carries in addition,
keeping |U+U| small while |U-U| can grow.
"""
import time
import random
import numpy as np
from math import log

# EVOLVE-BLOCK-START

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
        
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1.0
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1.0
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
        
    return 1 + log(ratio) / log(D)

def score_set_fast(U):
    """Faster scoring using numpy for larger sets."""
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    
    arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums (i <= j)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(U[i] + U[j])
    
    # Compute all pairwise differences
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = U[j] - U[i]
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1.0
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1.0
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
        
    return 1 + log(ratio) / log(D)

def compute_sum_diff_sets(U):
    """Compute sum and difference sets, return (sums_set, diffs_set)."""
    n = len(U)
    sums = set()
    diffs = set()
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    return sums, diffs

def greedy_binary_carry_free(deadline, max_val_limit=200000, max_n_limit=60):
    """
    Greedily construct U using binary carry-free structure.
    Select elements that maximize |U-U| growth while minimizing |U+U| growth.
    Prefer elements with sparse binary representations to avoid carries.
    """
    U = [0]
    U_set = {0}
    current_sums, current_diffs = compute_sum_diff_sets(U)
    
    # Track which binary positions are "used" to encourage disjoint supports
    used_bits = set()
    
    while len(U) < max_n_limit and time.time() < deadline - 3:
        # Generate candidates: prefer numbers with few bits set and high value
        candidates = set()
        
        # Power-of-2 based candidates (single bit)
        for k in range(1, 25):
            val = 1 << k
            if val <= max_val_limit and val not in U_set:
                candidates.add(val)
        
        # Sparse binary candidates (2-3 bits set)
        for k1 in range(1, 22):
            for k2 in range(k1 + 2, 24):  # Gap of at least 2 to avoid carry
                val = (1 << k1) | (1 << k2)
                if val <= max_val_limit and val not in U_set:
                    candidates.add(val)
        
        # Arithmetic neighbors from existing elements
        for a in U:
            for b in U:
                s = a + b
                d = abs(a - b)
                if 0 < s <= max_val_limit and s not in U_set:
                    candidates.add(s)
                if 0 < d <= max_val_limit and d not in U_set:
                    candidates.add(d)
        
        # Some random candidates for diversity
        for _ in range(30):
            r = random.randint(1, min(max_val_limit, 5000))
            if r not in U_set:
                candidates.add(r)
        
        if not candidates:
            break
        
        # Score candidates by objective improvement
        best_cand = None
        best_score = -1.0
        current_score = score_set(U)
        
        cand_list = list(candidates)
        random.shuffle(cand_list)
        
        # Limit candidates to try for time management
        max_to_try = min(len(cand_list), 80)
        
        for c in cand_list[:max_to_try]:
            new_U = U + [c]
            new_U.sort()
            new_score = score_set(new_U)
            if new_score > best_score:
                best_score = new_score
                best_cand = c
        
        if best_cand is not None and best_score > current_score:
            U.append(best_cand)
            U.sort()
            U_set.add(best_cand)
            # Update bit tracking
            for bit in range(25):
                if best_cand & (1 << bit):
                    used_bits.add(bit)
        else:
            # If no improvement, try the candidate with best ratio improvement
            best_ratio_cand = None
            best_ratio_delta = 0.0
            cur_sum_cnt = len(current_sums)
            cur_diff_cnt = len(current_diffs)
            
            for c in cand_list[:50]:
                # Estimate new sum/diff counts
                new_sum_cnt = cur_sum_cnt
                new_diff_cnt = cur_diff_cnt
                
                for u in U:
                    ns = u + c
                    if ns not in current_sums:
                        new_sum_cnt += 1
                    nd = abs(u - c)
                    if nd != 0 and nd not in current_diffs:
                        new_diff_cnt += 2  # +/- d
                    elif nd == 0:
                        pass  # 0 already in diffs
                
                if new_sum_cnt > 0:
                    new_ratio = new_diff_cnt / new_sum_cnt
                    old_ratio = cur_diff_cnt / cur_sum_cnt if cur_sum_cnt > 0 else 0
                    delta = new_ratio - old_ratio
                    if delta > best_ratio_delta:
                        best_ratio_delta = delta
                        best_ratio_cand = c
            
            if best_ratio_cand is not None and best_ratio_cand not in U_set:
                U.append(best_ratio_cand)
                U.sort()
                U_set.add(best_ratio_cand)
            else:
                break
    
    return U

def local_search_binary(U, deadline, max_val_limit=10000, max_n_limit=50):
    """
    Local search focused on binary-structured improvements.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    no_improvement = 0
    max_no_improvement = 20
    
    while time.time() < deadline - 2 and no_improvement < max_no_improvement:
        improved = False
        move_type = random.random()
        
        if move_type < 0.4:
            # ADD: try binary-structured candidates
            if len(U) < max_n_limit:
                U_set = set(U)
                candidates = set()
                
                # Single-bit candidates
                for k in range(1, 25):
                    val = 1 << k
                    if val <= max_val_limit and val not in U_set:
                        candidates.add(val)
                
                # Two-bit candidates with gap
                for k1 in range(1, 20):
                    for k2 in range(k1 + 2, 23):
                        val = (1 << k1) | (1 << k2)
                        if val <= max_val_limit and val not in U_set:
                            candidates.add(val)
                
                # Arithmetic neighbors
                for a in U:
                    for b in U:
                        s = a + b
                        d = abs(a - b)
                        if 0 < s <= max_val_limit and s not in U_set:
                            candidates.add(s)
                        if 0 < d <= max_val_limit and d not in U_set:
                            candidates.add(d)
                
                cand_list = list(candidates)
                random.shuffle(cand_list)
                
                for c in cand_list[:15]:
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
                        
        elif move_type < 0.6:
            # REMOVE: try removing the element that hurts the score most
            if len(U) > 3:
                removable = [x for x in U if x != 0]
                random.shuffle(removable)
                for c in removable[:5]:
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
                        
        else:
            # REPLACE: swap an element with a better candidate
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                U_set = set(U)
                
                candidates = set()
                for k in range(1, 25):
                    val = 1 << k
                    if val <= max_val_limit and val not in U_set:
                        candidates.add(val)
                for k1 in range(1, 20):
                    for k2 in range(k1 + 2, 23):
                        val = (1 << k1) | (1 << k2)
                        if val <= max_val_limit and val not in U_set:
                            candidates.add(val)
                
                cand_list = list(candidates)
                random.shuffle(cand_list)
                
                for new_val in cand_list[:10]:
                    if new_val == old_val:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
        
        if improved:
            no_improvement = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement += 1
    
    return best_local_score, best_local_set

def construct():
    deadline = time.time() + 140  # Safety margin for 150s limit
    best_score = -1.0
    best_set = [0, 1, 4, 9, 11]
    
    # Initial best
    s = score_set(best_set)
    if s > best_score:
        best_score = s
        best_set = list(best_set)
    
    # 1. Try structured candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        [0, 1, 3, 7, 15, 31, 63, 127, 255],
        [0, 1, 4, 10, 20, 35, 56, 84, 120],
        [0, 1, 4, 10, 18, 28, 40, 54, 70],
        [0, 1, 3, 9, 27, 81, 243, 729],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        [0, 1, 4, 16, 64, 256, 1024, 4096],
        [0, 1, 3, 6, 10, 15, 21, 28, 36],
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # 2. Greedy binary carry-free construction
    if time.time() < deadline - 30:
        greedy_set = greedy_binary_carry_free(deadline - 15, max_val_limit=200000, max_n_limit=60)
        greedy_score = score_set(greedy_set)
        if greedy_score > best_score:
            best_score = greedy_score
            best_set = greedy_set[:]
            structured_candidates.append(greedy_set)
    
    # 3. Local search from multiple starting points
    max_val_limit = 5000
    max_n_limit = 40
    
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        
        # Choose starting set
        if restarts % 4 == 0:
            base = random.choice(structured_candidates)
            U = list(base)
        elif restarts % 4 == 1:
            # Start from binary-structured set
            n = random.randint(5, 20)
            bits = random.sample(range(1, 22), n)
            bits.sort()
            U = [0]
            for b in bits:
                U.append(1 << b)
            U.sort()
        elif restarts % 4 == 2:
            # Start from geometric
            B = random.uniform(1.5, 4.0)
            K = random.randint(5, 15)
            U = [0]
            val = 1.0
            for i in range(1, K):
                U.append(int(val))
                val *= B
            U = sorted(set(U))
        else:
            # Random start
            n = random.randint(5, 20)
            elements = random.sample(range(1, max_val_limit + 1), n)
            U = [0] + elements
            U.sort()
        
        # Run local search
        local_score, local_set = local_search_binary(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            structured_candidates.append(local_set)
    
    # 4. Final polish on best set
    if time.time() < deadline - 2:
        U = list(best_set)
        final_score, final_set = local_search_binary(U, deadline, max_val_limit, max_n_limit)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
    
    return best_set

# EVOLVE-BLOCK-END
