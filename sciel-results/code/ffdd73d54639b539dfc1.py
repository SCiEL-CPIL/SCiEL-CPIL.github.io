import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit

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
    
    # Precompute all words
    all_words = [list(w) for w in product(range(q), repeat=n)]
    
    # Precompute neighbors (ball contents)
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
    
    # Precompute neighbors as numpy arrays for faster indexing
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    # Precompute ball sizes for potential use
    ball_sizes = [len(nb) for nb in neighbors]
    
    # 1. Initial Construction: Greedy Set Cover using Bitsets
    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask
        
    all_mask = (1 << N) - 1
    current_cover_mask = 0
    codewords = []
    is_chosen = np.zeros(N, dtype=bool)
    
    # Greedy Set Cover
    while current_cover_mask != all_mask:
        if time.time() > deadline - 80:
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
            break # Should not happen
            
        codewords.append(best_i)
        is_chosen[best_i] = True
        current_cover_mask |= ball_bitsets[best_i]
        
    # Ensure validity
    if current_cover_mask != all_mask:
        while current_cover_mask != all_mask:
            i = random.randint(0, N-1)
            if not is_chosen[i]:
                codewords.append(i)
                is_chosen[i] = True
            current_cover_mask |= ball_bitsets[i]
            if time.time() > deadline - 10:
                break

    # 2. Local Search Setup
    cover_count = np.zeros(N, dtype=np.int32)
    for cw in codewords:
        cover_count[neighbors_np[cw]] += 1
        
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    # Precompute coverers ONCE (point -> list of codeword indices whose ball covers it)
    coverers = [[] for _ in range(N)]
    for cw_idx, ball in enumerate(neighbors):
        for pt in ball:
            coverers[pt].append(cw_idx)
    
    # Also precompute as numpy arrays for faster intersection
    coverers_np = [np.array(c, dtype=np.int32) for c in coverers]
    
    # Helper: Try to remove redundant codewords
    def try_remove_redundant(codewords, cover_count):
        removed = 0
        i = 0
        while i < len(codewords):
            if time.time() > deadline - 15:
                break
            cw = codewords[i]
            # Check if all points in ball(cw) have count > 1
            b = neighbors_np[cw]
            if np.min(cover_count[b]) > 1:
                cover_count[b] -= 1
                codewords.pop(i)
                removed += 1
            else:
                i += 1
        return removed

    # Initial redundancy removal
    try_remove_redundant(codewords, cover_count)
    if len(codewords) < best_size:
        best_size = len(codewords)
        best_codewords = list(codewords)
        
    # Local Search Loop with Iterated Local Search (ILS)
    random.seed(42)
    no_improvement_count = 0
    max_no_improvement = 50
    
    # Temperature for simulated annealing (used for swap acceptance)
    temperature = 1.0
    cooling_rate = 0.98
    
    while time.time() < deadline - 5:
        improved = False
        
        # 1. Try Removals
        removed = try_remove_redundant(codewords, cover_count)
        if removed > 0:
            improved = True
            no_improvement_count = 0
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
            continue # Restart loop after modification
        
        # 2. Try Swaps
        if codewords:
            # Pick a random codeword to try swapping
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            # Find critical points: points in ball(old_cw) with count == 1
            b_old = neighbors_np[old_cw]
            critical_points = []
            for pt in b_old:
                if cover_count[pt] == 1:
                    critical_points.append(pt)
            
            if not critical_points:
                # Should have been removed by redundancy check
                # But if it wasn't (e.g. time limit), force remove
                cover_count[b_old] -= 1
                codewords.pop(idx)
                improved = True
                no_improvement_count = 0
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                continue
            
            # Find candidates that cover ALL critical points
            # Use precomputed coverers
            if len(critical_points) == 1:
                candidate_arr = coverers_np[critical_points[0]]
            else:
                # Intersect coverers for all critical points
                # Start with the smallest coverers list for efficiency
                crit_indices = sorted(critical_points, key=lambda p: len(coverers[p]))
                candidate_set = set(coverers[crit_indices[0]])
                for p in crit_indices[1:]:
                    if not candidate_set:
                        break
                    candidate_set.intersection_update(coverers[p])
                candidate_arr = np.array(list(candidate_set), dtype=np.int32) if candidate_set else np.array([], dtype=np.int32)
            
            # Remove old_cw and existing codewords from candidates
            if len(candidate_arr) > 0:
                existing_set = set(codewords)
                existing_set.discard(old_cw)
                mask = np.ones(len(candidate_arr), dtype=bool)
                for cw in existing_set:
                    mask &= (candidate_arr != cw)
                candidate_arr = candidate_arr[mask]
            
            if len(candidate_arr) > 0:
                # Pick the best candidate
                best_new_cw = -1
                min_cost = float('inf')
                
                # Limit candidates to check
                if len(candidate_arr) > 200:
                    random.shuffle(candidate_arr)
                    candidate_arr = candidate_arr[:200]
                
                # Vectorized cost computation
                # cost for each candidate = sum of cover_count over its ball
                # We can compute this efficiently
                b_new_all = np.array([neighbors_np[cw] for cw in candidate_arr])
                costs = np.array([np.sum(cover_count[b]) for b in b_new_all])
                
                best_idx = np.argmin(costs)
                best_new_cw = candidate_arr[best_idx]
                min_cost = costs[best_idx]
                
                # Simulated annealing: accept worse solutions with probability
                b_new = neighbors_np[best_new_cw]
                new_total_cost = np.sum(cover_count[b_new])
                old_total_cost = np.sum(cover_count[b_old])
                delta_cost = new_total_cost - old_total_cost
                
                # Accept if better, or with probability exp(-delta/T) if worse
                accept = True
                if delta_cost > 0:
                    accept = random.random() < np.exp(-delta_cost / max(temperature, 0.01))
                
                if accept:
                    # Perform swap
                    cover_count[b_old] -= 1
                    cover_count[b_new] += 1
                    codewords[idx] = best_new_cw
                    improved = True
                    no_improvement_count = 0
                    # Try to remove redundancies after swap
                    try_remove_redundant(codewords, cover_count)
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_codewords = list(codewords)
                    continue # Restart loop
        
        if not improved:
            no_improvement_count += 1
            temperature *= cooling_rate
            
            if no_improvement_count >= max_no_improvement:
                # Perturbation: Add a few random codewords to break local structure
                num_perturb = random.randint(2, 5)
                for _ in range(num_perturb):
                    rand_cw = random.randint(0, N - 1)
                    if rand_cw not in codewords:
                        codewords.append(rand_cw)
                        cover_count[neighbors_np[rand_cw]] += 1
                no_improvement_count = 0
                temperature = max(temperature, 0.5)
                # Immediately try to clean up
                try_remove_redundant(codewords, cover_count)
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)

    # Final Polish
    final_codewords = list(best_codewords)
    final_cover_count = np.zeros(N, dtype=np.int32)
    for cw in final_codewords:
        final_cover_count[neighbors_np[cw]] += 1
        
    # Run multiple passes of redundancy removal
    for _ in range(10):
        if time.time() > deadline - 2:
            break
        removed = try_remove_redundant(final_codewords, final_cover_count)
        if removed == 0:
            break
        
    # Verify final validity
    if np.min(final_cover_count) > 0:
        return [list(all_words[cw]) for cw in final_codewords]
    else:
        # Fallback: Return the initial greedy solution if something went wrong
        return [list(all_words[cw]) for cw in codewords]
