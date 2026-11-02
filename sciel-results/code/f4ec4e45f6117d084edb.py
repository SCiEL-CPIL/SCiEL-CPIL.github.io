"""Combined program: algebraic constructive initialization + aggressive local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

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
    ball_size = 1 + n * (q - 1)  # 14
    # Adjust deadline to be safe within 120s limit
    deadline = start_time + 110

    # Precompute all words
    all_words = [list(w) for w in product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

    # Precompute neighbors (ball contents) as numpy arrays
    neighbors = np.empty((N, ball_size), dtype=np.int32)
    for i, w in enumerate(all_words):
        w_list = list(w)
        neighbors[i, 0] = i
        pos = 1
        for j in range(n):
            orig_val = w_list[j]
            for val in range(q):
                if val != orig_val:
                    w_list[j] = val
                    neighbors[i, pos] = word_to_idx[tuple(w_list)]
                    pos += 1
            w_list[j] = orig_val

    # Precompute coverers: coverers[point] = list of codewords that cover point
    coverers = [[] for _ in range(N)]
    for cw_idx in range(N):
        for pt in neighbors[cw_idx]:
            coverers[pt].append(cw_idx)

    # 1. Initial Construction: Fast Greedy Set Cover with bitsets
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
        if time.time() > deadline - 100:
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

    # Ensure validity
    if current_cover_mask != all_mask:
        while current_cover_mask != all_mask:
            i = random.randint(0, N - 1)
            if not is_chosen[i]:
                codewords.append(i)
                is_chosen[i] = True
            current_cover_mask |= ball_bitsets[i]
            if time.time() > deadline - 10:
                break

    # 2. Local Search Setup
    cover_count = np.zeros(N, dtype=np.int16)
    for cw in codewords:
        cover_count[neighbors[cw]] += 1

    best_codewords = list(codewords)
    best_size = len(codewords)
    current_codeword_set = set(codewords)

    # Helper: Calculate redundancy weight (sum of max(0, count-1))
    def calc_redundancy_weight(cc):
        return np.sum(np.maximum(cc - 1, 0))

    current_weight = calc_redundancy_weight(cover_count)

    # Local Search Loop
    random.seed(42)
    no_improvement_count = 0
    max_no_improvement = 200

    while time.time() < deadline - 3:
        improved = False

        # 1. Vectorized Redundancy Removal
        # mins[i] is the minimum cover count in the ball of word i
        mins = np.min(cover_count[neighbors], axis=1)
        redundant_mask = mins > 1
        
        # Find indices in codewords that are redundant
        to_remove = []
        for cw in codewords:
            if redundant_mask[cw]:
                to_remove.append(cw)
                
        if to_remove:
            for cw in to_remove:
                cover_count[neighbors[cw]] -= 1
                codewords.remove(cw)
                current_codeword_set.remove(cw)
            current_weight = calc_redundancy_weight(cover_count)
            improved = True
            no_improvement_count = 0
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
            continue

        # 2. Try Swaps
        if codewords:
            # Pick a random codeword to try swapping
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]

            # Find critical points: points in ball(old_cw) with count == 1
            b_old = neighbors[old_cw]
            critical_points = b_old[cover_count[b_old] == 1]

            if len(critical_points) == 0:
                # Should have been removed by redundancy check
                cover_count[b_old] -= 1
                codewords.pop(idx)
                current_codeword_set.remove(old_cw)
                current_weight = calc_redundancy_weight(cover_count)
                improved = True
                no_improvement_count = 0
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                continue

            # Find candidates that cover ALL critical points
            # Start with the coverers of the critical point with fewest coverers
            min_len = float('inf')
            min_pt = critical_points[0]
            for pt in critical_points:
                cl = len(coverers[pt])
                if cl < min_len:
                    min_len = cl
                    min_pt = pt

            candidates = coverers[min_pt]
            for p in critical_points:
                if p == min_pt:
                    continue
                # Intersect candidates with coverers[p]
                # Since lists are small, simple set intersection is fast
                set_p = set(coverers[p])
                candidates = [c for c in candidates if c in set_p]
            
            if len(candidates) == 0:
                continue

            # Filter out old_cw and existing codewords
            candidates_list = [c for c in candidates if c != old_cw and c not in current_codeword_set]

            if candidates_list:
                if len(candidates_list) > 50:
                    random.shuffle(candidates_list)
                    candidates_list = candidates_list[:50]

                best_new_cw = -1
                min_new_weight = current_weight + 1 # Must be strictly better or equal? Let's aim for strictly better weight first.
                # Actually, we want to minimize weight.
                # Weight change = (new_weight - current_weight)
                # We accept if new_weight < current_weight.
                # Or if new_weight == current_weight and it allows future removals? 
                # Let's stick to strict improvement in weight for stability, or allow equal if it changes the set.
                # Better: Accept if new_weight <= current_weight, but prefer lower.
                
                for new_cw in candidates_list:
                    b_new = neighbors[new_cw]
                    # Hypothetical new cover count
                    # old_cw is removed, new_cw is added
                    # delta = -1 for b_old, +1 for b_new
                    # We can calculate the change in weight efficiently
                    # weight = sum(max(0, c-1))
                    # change in weight for a point p:
                    # if c=1 -> 0, c=2 -> 1. 
                    # Removing old:
                    # points in b_old:
                    #   if count was 1 -> 0 (weight 0 -> 0, change 0)
                    #   if count was 2 -> 1 (weight 1 -> 0, change -1)
                    #   if count was 3 -> 2 (weight 2 -> 1, change -1)
                    # Adding new:
                    # points in b_new:
                    #   if count was 0 -> 1 (weight 0 -> 0, change 0)
                    #   if count was 1 -> 2 (weight 0 -> 1, change +1)
                    #   if count was 2 -> 3 (weight 1 -> 2, change +1)
                    
                    # Since critical points of old have count 1, they drop to 0.
                    # Other points in b_old have count >= 2.
                    # Points in b_new have count >= 0.
                    
                    # Efficient calculation:
                    # weight_change = 0
                    # For p in b_old:
                    #   if cover_count[p] > 1: weight_change -= 1
                    # For p in b_new:
                    #   if cover_count[p] > 0: weight_change += 1
                    # Note: points in intersection are handled correctly?
                    # If p in b_old and b_new:
                    #   old count c.
                    #   remove: c-1. add: c.
                    #   net change 0 in count.
                    #   weight change:
                    #     if c=1 (crit): remove->0 (0), add->1 (0). change 0.
                    #     if c=2: remove->1 (0), add->2 (1). change +1.
                    #     if c=3: remove->2 (1), add->3 (2). change +1.
                    #   My formula:
                    #     p in b_old, count 2 > 1 -> -1.
                    #     p in b_new, count 2 > 0 -> +1.
                    #     Net 0. Correct.
                    
                    # So:
                    # weight_change = np.sum(cover_count[b_new] > 0) - np.sum(cover_count[b_old] > 1)
                    
                    delta = np.sum(cover_count[b_new] > 0) - np.sum(cover_count[b_old] > 1)
                    new_weight = current_weight + delta
                    
                    if new_weight < min_new_weight:
                        min_new_weight = new_weight
                        best_new_cw = new_cw

                if best_new_cw != -1 and min_new_weight <= current_weight:
                    # Perform swap
                    cover_count[b_old] -= 1
                    cover_count[neighbors[best_new_cw]] += 1
                    codewords[idx] = best_new_cw
                    current_codeword_set.remove(old_cw)
                    current_codeword_set.add(best_new_cw)
                    current_weight = min_new_weight
                    improved = True
                    no_improvement_count = 0
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_codewords = list(codewords)
                    continue

        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Try a random restart from best_codewords?
                # Or just break
                break

    # Return the best result found
    result = best_codewords
    return [all_words[i] for i in result]
# EVOLVE-BLOCK-END
