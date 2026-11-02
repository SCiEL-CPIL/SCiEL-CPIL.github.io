"""Combined program: bitset greedy init + coverer-intersection swaps + redundancy removal."""
import itertools
import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)

    # Fallback for other cases
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
        return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
    return [list(w) for w in product(range(q), repeat=n)]

def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187

    # Precompute all words
    all_words = [list(w) for w in product(range(q), repeat=n)]

    # Map word tuple to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

    # Precompute neighbors (ball of radius 1)
    neighbors = []
    for w in all_words:
        covered = [word_to_idx[tuple(w)]]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val != orig_val:
                    w_list[pos] = val
                    covered.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig_val
        neighbors.append(covered)

    # Precompute coverers: coverers[point] = list of codeword indices that cover point
    coverers = [[] for _ in range(N)]
    for cw_idx, ball in enumerate(neighbors):
        for pt in ball:
            coverers[pt].append(cw_idx)

    # Precompute ball bitsets for fast greedy
    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask

    all_mask = (1 << N) - 1

    # --- Greedy Set Cover (deterministic) ---
    def greedy_set_cover():
        current_cover_mask = 0
        codewords = []
        is_chosen = [False] * N

        while current_cover_mask != all_mask:
            if time.time() > deadline - 60:
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

        # Ensure full coverage
        if current_cover_mask != all_mask:
            random.shuffle(range(N))
            for i in range(N):
                if current_cover_mask == all_mask:
                    break
                if not is_chosen[i]:
                    codewords.append(i)
                    is_chosen[i] = True
                    current_cover_mask |= ball_bitsets[i]

        return codewords

    # --- Redundancy removal ---
    def try_remove_redundant(codewords, cover_count):
        removed = 0
        i = 0
        while i < len(codewords):
            if time.time() > deadline - 30:
                break
            cw = codewords[i]
            b = neighbors[cw]
            if np.min(cover_count[b]) > 1:
                cover_count[b] -= 1
                codewords.pop(i)
                removed += 1
            else:
                i += 1
        return removed

    # --- Local search with swap using coverer intersection ---
    def local_search(codewords, cover_count):
        best_codewords = list(codewords)
        best_size = len(codewords)
        no_improvement = 0
        max_no_improvement = 80

        while time.time() < deadline - 2:
            improved = False

            # 1. Try removals
            removed = try_remove_redundant(codewords, cover_count)
            if removed > 0:
                improved = True
                no_improvement = 0
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                continue

            # 2. Try swaps
            if codewords:
                idx = random.randint(0, len(codewords) - 1)
                old_cw = codewords[idx]
                b_old = neighbors[old_cw]

                # Find critical points
                critical_points = [pt for pt in b_old if cover_count[pt] == 1]

                if not critical_points:
                    cover_count[b_old] -= 1
                    codewords.pop(idx)
                    improved = True
                    no_improvement = 0
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_codewords = list(codewords)
                    continue

                # Find candidates covering all critical points via coverer intersection
                candidate_set = set(coverers[critical_points[0]])
                for p in critical_points[1:]:
                    candidate_set.intersection_update(coverers[p])

                candidate_set.discard(old_cw)
                for cw in codewords:
                    candidate_set.discard(cw)

                if candidate_set:
                    candidates_list = list(candidate_set)
                    if len(candidates_list) > 30:
                        random.shuffle(candidates_list)
                        candidates_list = candidates_list[:30]

                    # Pick best candidate: minimize sum of cover_count in its ball
                    best_new_cw = -1
                    min_cost = float('inf')
                    for cand in candidates_list:
                        b_cand = neighbors[cand]
                        cost = cover_count[b_cand].sum()
                        if cost < min_cost:
                            min_cost = cost
                            best_new_cw = cand

                    if best_new_cw != -1:
                        # Perform swap
                        cover_count[b_old] -= 1
                        cover_count[neighbors[best_new_cw]] += 1
                        codewords[idx] = best_new_cw
                        improved = True
                        no_improvement = 0
                        if len(codewords) < best_size:
                            best_size = len(codewords)
                            best_codewords = list(codewords)
                        continue

            if not improved:
                no_improvement += 1
                if no_improvement >= max_no_improvement:
                    break

        return best_codewords

    # --- Main search loop ---
    random.seed(42)

    # Initial greedy
    codewords = greedy_set_cover()
    cover_count = np.zeros(N, dtype=np.int16)
    for cw in codewords:
        cover_count[neighbors[cw]] += 1

    # Initial redundancy removal
    try_remove_redundant(codewords, cover_count)

    # Local search
    best_codewords = local_search(codewords, cover_count)
    best_size = len(best_codewords)

    # Randomized restarts
    while time.time() < deadline - 5:
        # Randomized greedy: shuffle evaluation order
        current_cover_mask = 0
        codewords = []
        is_chosen = [False] * N
        order = list(range(N))
        random.shuffle(order)

        while current_cover_mask != all_mask:
            if time.time() > deadline - 10:
                break
            best_i = -1
            best_count = 0
            for i in order:
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
            for i in range(N):
                if current_cover_mask == all_mask:
                    break
                if not is_chosen[i]:
                    codewords.append(i)
                    is_chosen[i] = True
                    current_cover_mask |= ball_bitsets[i]

        cover_count = np.zeros(N, dtype=np.int16)
        for cw in codewords:
            cover_count[neighbors[cw]] += 1

        try_remove_redundant(codewords, cover_count)
        result = local_search(codewords, cover_count)

        if len(result) < best_size:
            best_size = len(result)
            best_codewords = result

    return [all_words[i] for i in best_codewords]
