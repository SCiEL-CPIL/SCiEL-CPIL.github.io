import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)

    return _direct_sum_construct(q, n, radius)

def _direct_sum_construct(q, n, radius):
    if n < 4:
        if n == 0:
            return [[]]
        if n == 1:
            return [[i] for i in range(q)]
        if n == 2:
            return [[a, b] for a in range(q) for b in range(q)]
        if n == 3:
            return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]

    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        return [list(w) for w in product(range(q), repeat=n)]

def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187

    # Precompute all words and neighbors
    all_words = [list(w) for w in product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    neighbors = []
    for w in all_words:
        covered = [word_to_idx[tuple(w)]]
        w_list = list(w)
        for i in range(n):
            orig_val = w_list[i]
            for val in range(q):
                if val != orig_val:
                    w_list[i] = val
                    covered.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig_val
        neighbors.append(covered)
        
    # Precompute static coverers
    coverers_static = [[] for _ in range(N)]
    for cw_idx in range(N):
        for pt in neighbors[cw_idx]:
            coverers_static[pt].append(cw_idx)

    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask

    all_mask = (1 << N) - 1
    
    best_global_code = None
    best_global_size = float('inf')
    
    # Run multiple rounds
    while time.time() < deadline - 5:
        round_start = time.time()
        # Each round should take ~20-30s
        round_deadline = min(deadline - 5, round_start + 25)
        
        # 1. Greedy Initialization
        random.seed(int(round_start * 1000)) # Varying seed
        current_cover_mask = 0
        codewords = []
        is_chosen = [False] * N
        
        while current_cover_mask != all_mask:
            if time.time() > round_deadline - 10:
                break
            best_i = -1
            best_count = 0
            for i in range(N):
                if is_chosen[i]:
                    continue
                new_bits = ball_bitsets[i] & ~current_cover_mask
                if new_bits:
                    count = new_bits.bit_count()
                    if count > best_count:
                        best_count = count
                        best_i = i
            if best_i == -1:
                break
            codewords.append(best_i)
            is_chosen[best_i] = True
            current_cover_mask |= ball_bitsets[best_i]

        if current_cover_mask != all_mask:
            while current_cover_mask != all_mask:
                i = random.randint(0, N - 1)
                if not is_chosen[i]:
                    codewords.append(i)
                    is_chosen[i] = True
                current_cover_mask |= ball_bitsets[i]
                if time.time() > round_deadline - 1:
                    break
            if current_cover_mask != all_mask:
                continue # Invalid code, skip local search

        code = codewords[:]
        
        # 2. Local Search
        cover_count = np.zeros(N, dtype=np.int16)
        for cw in code:
            cover_count[neighbors[cw]] += 1
            
        code_set = set(code)
        
        def try_remove_redundant(code, code_set, cover_count, dl):
            removed = 0
            for i in range(len(code)):
                if time.time() > dl:
                    break
                cw = code[i]
                b = neighbors[cw]
                if np.min(cover_count[b]) > 1:
                    cover_count[b] -= 1
                    code.pop(i)
                    code_set.remove(cw)
                    removed += 1
            return removed

        try_remove_redundant(code, code_set, cover_count, round_deadline)
        
        local_best_code = code[:]
        local_best_size = len(code)
        
        no_improvement = 0
        max_no_improvement = 200
        
        while time.time() < round_deadline - 1:
            improved = False
            
            removed = try_remove_redundant(code, code_set, cover_count, round_deadline)
            if removed > 0:
                improved = True
                no_improvement = 0
                if len(code) < local_best_size:
                    local_best_size = len(code)
                    local_best_code = code[:]
                continue
            
            if not code:
                break
                
            idx = random.randint(0, len(code) - 1)
            old_cw = code[idx]
            
            b_old = neighbors[old_cw]
            critical_points = []
            for pt in b_old:
                if cover_count[pt] == 1:
                    critical_points.append(pt)
                    
            if not critical_points:
                cover_count[b_old] -= 1
                code.pop(idx)
                code_set.remove(old_cw)
                improved = True
                no_improvement = 0
                if len(code) < local_best_size:
                    local_best_size = len(code)
                    local_best_code = code[:]
                continue
                
            # Find candidates
            if len(critical_points) > 5:
                no_improvement += 1
                if no_improvement > max_no_improvement:
                    break
                continue

            crit_sorted = sorted(critical_points, key=lambda p: len(coverers_static[p]))
            candidates = set(coverers_static[crit_sorted[0]])
            for p in crit_sorted[1:]:
                candidates.intersection_update(coverers_static[p])
                
            candidates.discard(old_cw)
            candidates.difference_update(code_set)
            
            if not candidates:
                no_improvement += 1
                if no_improvement > max_no_improvement:
                    break
                continue
                
            best_new_cw = -1
            best_score = -float('inf')
            
            cand_list = list(candidates)
            if len(cand_list) > 50:
                random.shuffle(cand_list)
                cand_list = cand_list[:50]
                
            for new_cw in cand_list:
                b_new = neighbors[new_cw]
                score = np.sum(cover_count[b_new])
                if score > best_score or (score == best_score and random.random() < 0.1):
                    best_score = score
                    best_new_cw = new_cw
                    
            if best_new_cw != -1:
                cover_count[b_old] -= 1
                cover_count[neighbors[best_new_cw]] += 1
                code[idx] = best_new_cw
                code_set.remove(old_cw)
                code_set.add(best_new_cw)
                
                improved = True
                no_improvement = 0
                
                try_remove_redundant(code, code_set, cover_count, round_deadline)
                
                if len(code) < local_best_size:
                    local_best_size = len(code)
                    local_best_code = code[:]
                continue
                
            if not improved:
                no_improvement += 1
                if no_improvement > max_no_improvement:
                    break

        # End Local Search
        if local_best_size < best_global_size:
            best_global_size = local_best_size
            best_global_code = local_best_code[:]
            
    return [all_words[i] for i in best_global_code]
