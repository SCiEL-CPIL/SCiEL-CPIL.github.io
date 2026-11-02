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

    # Precompute all words
    all_words = [list(w) for w in product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

    # Precompute neighbors (ball contents)
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

    # Precompute coverers: coverers[point] = list of codewords that cover point
    coverers = [[] for _ in range(N)]
    for cw_idx, ball in enumerate(neighbors):
        for pt in ball:
            coverers[pt].append(cw_idx)

    # Convert neighbors to numpy arrays for faster vector ops
    # We will use bitsets for the initial greedy phase for speed.

    # 1. Initial Construction: Algebraic + Greedy Hybrid
    # Use the structure of the ternary Hamming code to get a good starting point.
    # The [7,4,3] ternary Hamming code has 27 codewords. Its cosets partition the space.
    # A covering code of radius 1 can be constructed by picking one word from each coset
    # such that the union of their balls covers the space.
    
    # However, a simpler and effective approach for initialization is to use the
    # "direct sum" construction but optimized, or a greedy set cover.
    # Let's use a fast greedy set cover with bitsets.

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

        # Optimization: Use a random subset of candidates if time is tight
        # But for now, scan all. N=2187 is small enough.
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
            break  # Should not happen

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

    # Helper: Try to remove redundant codewords
    def try_remove_redundant(codewords, cover_count):
        removed = 0
        i = 0
        while i < len(codewords):
            if time.time() > deadline - 30:
                break
            cw = codewords[i]
            # Check if all points in ball(cw) have count > 1
            b = neighbors[cw]
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

    # Local Search Loop
    random.seed(42)
    no_improvement_count = 0
    max_no_improvement = 100

    while time.time() < deadline - 2:
        improved = False

        # 1. Try Removals
        removed = try_remove_redundant(codewords, cover_count)
        if removed > 0:
            improved = True
            no_improvement_count = 0
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
            continue  # Restart loop after modification

        # 2. Try Swaps with Critical Word Heuristic
        if codewords:
            # Pick a random codeword to try swapping
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]

            # Find critical points: points in ball(old_cw) with count == 1
            b_old = neighbors[old_cw]
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
            # candidates = intersection of coverers[p] for p in critical_points
            # Start with coverers of the first critical point
            candidate_set = set(coverers[critical_points[0]])
            for p in critical_points[1:]:
                candidate_set.intersection_update(coverers[p])

            # Remove old_cw and existing codewords from candidates
            candidate_set.discard(old_cw)
            for cw in codewords:
                candidate_set.discard(cw)

            if candidate_set:
                # Pick the best candidate
                # Heuristic: Pick the one with the lowest "cost"
                # Cost could be the sum of cover_count in its ball (lower is better, less redundancy)
                best_new_cw = -1
                min_cost = float('inf')

                # Limit candidates to check to avoid O(N) scan if set is huge
                # But intersection of coverers is usually small.
                candidates_list = list(candidate_set)
                if len(candidates_list) > 20:
                    random.shuffle(candidates_list)
                    candidates_list = candidates_list[:20]

                for new_cw in candidates_list:
                    b_new = neighbors[new_cw]
                    # Check if swapping improves coverage or maintains it
                    # The new codeword must cover all critical points (guaranteed by construction)
                    # We want to minimize the increase in redundancy
                    # Cost = sum of cover_count[p] for p in b_new, excluding critical points
                    # Actually, simpler: just try the swap and see if it's valid
                    # Since it covers all critical points, the swap is valid.
                    # We prefer candidates that cover many points already well-covered (to allow future removals)
                    # Or candidates that are "far" from other codewords?
                    # Let's use: cost = number of points in b_new with cover_count > 1
                    # Lower cost means it overlaps less with existing coverage
                    cost = 0
                    for p in b_new:
                        if cover_count[p] > 1:
                            cost += 1
                    if cost < min_cost:
                        min_cost = cost
                        best_new_cw = new_cw

                if best_new_cw != -1:
                    # Perform swap
                    cover_count[b_old] -= 1
                    cover_count[neighbors[best_new_cw]] += 1
                    codewords[idx] = best_new_cw
                    improved = True
                    no_improvement_count = 0
                    # Try to remove redundancies after swap
                    try_remove_redundant(codewords, cover_count)
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_codewords = list(codewords)
                    continue

        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                break

    # Return the best result found
    result = best_codewords
    # Ensure distinctness (should be guaranteed)
    return [all_words[i] for i in result]
# EVOLVE-BLOCK-END
