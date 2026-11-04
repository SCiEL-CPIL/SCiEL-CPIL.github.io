import time
from math import log, floor, sqrt
import random
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
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
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included and all are nonnegative distinct integers."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def base_expansion_set(B, k, max_elements=None):
    """
    Generate a set using base-B expansion hierarchy.
    Elements are u = sum_{j=0}^{k-1} b_j * B^j where b_j in {0, 1}.
    """
    if max_elements is None:
        max_elements = 2 ** k
    
    U = set()
    U.add(0)
    
    for mask in range(1, min(2 ** k, max_elements)):
        val = 0
        for j in range(k):
            if mask & (1 << j):
                val += B ** j
        U.add(val)
    
    return sorted(U)

def base_expansion_mixed(B, k, sparsity=0.5, max_elements=None):
    """
    Generate a set using base-B expansion with sparsity control.
    """
    if max_elements is None:
        max_elements = 2 ** k
    
    U = set()
    U.add(0)
    
    for mask in range(1, min(2 ** k, max_elements)):
        active = bin(mask).count('1')
        if abs(active / k - sparsity) <= 0.15 or active == k:
            val = 0
            for j in range(k):
                if mask & (1 << j):
                    val += B ** j
            U.add(val)
    
    return sorted(U)

def generate_candidates_enhanced(U, limit_val):
    """
    Generate candidate elements: sums, differences, and near-misses.
    """
    U_set = set(U)
    candidates = set()
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            # Near-misses: s±1, s±2
            for delta in [-2, -1, 1, 2]:
                sv = s + delta
                if 0 < sv <= limit_val and sv not in U_set:
                    candidates.add(sv)
                dv = d + delta
                if 0 < dv <= limit_val and dv not in U_set:
                    candidates.add(dv)
            
    return list(candidates)

def local_search_aggressive(U, deadline, max_val_limit=1000, max_n_limit=35):
    """
    Aggressive local search with enhanced candidate generation.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    candidates = generate_candidates_enhanced(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 25
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        if not candidates or random.random() < 0.15:
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates_enhanced(U, new_limit)
        
        if move_type < 0.45:
            # ADD
            if len(U) < max_n_limit and candidates:
                random.shuffle(candidates)
                for c in candidates[:15]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates_enhanced(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.65:
            # REMOVE
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates_enhanced(U, max_val_limit)
        else:
            # REPLACE
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                random.shuffle(candidates)
                for new_val in candidates[:15]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates_enhanced(U, max(max_val_limit, max(U)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def final_polish(U, deadline):
    """
    High-precision polish: fine-grained local search with small deltas.
    """
    current = sorted(set(U))
    current_score = score_set(current)
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 1 and iteration < 800:
        improved = False
        iteration += 1
        n = len(current)
        
        # Try replacing each element with nearby values (fine granularity)
        for i in range(1, n):
            if time.time() > deadline - 1:
                break
            old_val = current[i]
            for delta in range(-5, 6):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                new_set = [current[j] if j != i else new_val for j in range(n)]
                new_set = sorted(set(new_set))
                if len(new_set) != n:
                    continue
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        # Try adding an element near the max
        if not improved and n < 28 and time.time() < deadline - 1:
            max_val = max(current)
            for new_val in range(max_val - 8, max_val + 9):
                if new_val <= 0:
                    continue
                if new_val in current:
                    continue
                new_set = sorted(current + [new_val])
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break

        # Try removing an element
        if not improved and n > 3 and time.time() < deadline - 1:
            for i in range(1, n):
                new_set = [current[j] for j in range(n) if j != i]
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break

    return current_score, current

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Structured Candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        [0, 1, 2, 4, 8, 16, 32, 64, 128],
        [0, 1, 3, 7, 15, 31, 63, 127],
        [0, 1, 4, 10, 20, 35, 56, 84],
        [0, 1, 3, 4, 9, 10, 12, 13],
        [0, 1, 4, 10, 18, 28, 40, 54],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        [0, 1, 3, 9, 27, 81, 243],
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Base-Expansion Hierarchy Search
    for B in [3, 4, 5, 6, 8, 10, 12, 16]:
        if time.time() > deadline - 70:
            break
        for k in range(2, 8):
            U = base_expansion_set(B, k, max_elements=200)
            if len(U) >= 2:
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
            
            for sparsity in [0.3, 0.5, 0.7]:
                if time.time() > deadline - 65:
                    break
                U = base_expansion_mixed(B, k, sparsity=sparsity, max_elements=200)
                if len(U) >= 2:
                    s = score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = list(U)

    # 3. Geometric Search
    b_candidates = [b_int / 10.0 for b_int in range(12, 101)]
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 50:
            break
        for m in range(2, 21):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                if s > best_score:
                    best_score = s
                    best_set = U

    # 4. Aggressive Local Search with multiple restarts
    seeds = [best_set]
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 500
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 15:
        restarts += 1
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        local_score, local_set = local_search_aggressive(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            if len(seeds) < 5:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 5. Final Polish: High-precision refinement
    if time.time() < deadline - 2:
        polish_score, polished_set = final_polish(best_set, deadline)
        if polish_score > best_score:
            best_score = polish_score
            best_set = polished_set

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
