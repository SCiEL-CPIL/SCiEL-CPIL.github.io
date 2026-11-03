import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor bitmasks
    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                m |= (1 << (i + diff * powers[pos]))
        masks[i] = m

    # Build numpy cover matrix: cover_matrix[w, x] = 1 if x is within radius of w
    # This allows vectorized gain calculations
    # cover_matrix is bool array of shape (total_space, total_space)
    # For 2187x2187 that's about 4.8M bytes, very manageable
    cover_matrix = np.zeros((total_space, total_space), dtype=np.uint8)
    for w in range(total_space):
        mm = masks[w]
        while mm:
            lowbit = mm & -mm
            idx = lowbit.bit_length() - 1
            cover_matrix[w, idx] = 1
            mm ^= lowbit

    # Also build reverse: for each point x, which words cover it
    # rev_cover[x] = list of words w such that cover_matrix[w, x] = 1
    rev_cover = [None] * total_space
    for x in range(total_space):
        rev_cover[x] = np.where(cover_matrix[:, x] == 1)[0].tolist()

    # Precompute bit counts for masks (size of neighborhood)
    mask_sizes = np.array([masks[i].bit_count() for i in range(total_space)], dtype=np.int32)

    # Precompute neighborhood lists for faster iteration in local search
    mask_lists = [None] * total_space
    for i in range(total_space):
        mm = masks[i]
        lst = []
        while mm:
            lowbit = mm & -mm
            lst.append(lowbit.bit_length() - 1)
            mm ^= lowbit
        mask_lists[i] = lst

    best_size = total_space
    best_selected = list(range(total_space))

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def build_counts(selected):
        """Build coverage count array from selected codewords."""
        if not selected:
            return np.zeros(total_space, dtype=np.int32)
        sel_arr = np.array(selected, dtype=np.int32)
        counts = cover_matrix[sel_arr].sum(axis=0).astype(np.int32)
        return counts

    def polish(selected, time_limit):
        """Remove redundant codewords iteratively."""
        selected = selected[:]
        if not selected:
            return selected

        counts = build_counts(selected)

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                # Check if any point in w's neighborhood has count == 1
                removable = True
                for pos in mask_lists[w]:
                    if counts[pos] == 1:
                        removable = False
                        break

                if removable:
                    for pos in mask_lists[w]:
                        counts[pos] -= 1
                    selected.pop(idx)
                    changed = True

        return selected

    def local_search(selected, time_limit):
        """
        Local search with 1-for-1 swaps and 2-for-1 swaps using numpy cover matrix.
        """
        selected = selected[:]
        if not selected:
            return selected

        counts = build_counts(selected)
        selected_set = set(selected)

        # 1-for-1 swap
        max_passes = 80
        for p in range(max_passes):
            if time.time() > time_limit:
                break
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]

                # Identify unique points covered by w
                unique_points = []
                for pos in mask_lists[w]:
                    if counts[pos] == 1:
                        unique_points.append(pos)

                if not unique_points:
                    # w is redundant
                    for pos in mask_lists[w]:
                        counts[pos] -= 1
                    selected.pop(idx)
                    selected_set.remove(w)
                    improved = True
                    continue

                # Find candidates that cover all unique_points using rev_cover
                if len(unique_points) == 1:
                    candidates = rev_cover[unique_points[0]]
                else:
                    candidates = rev_cover[unique_points[0]]
                    for pt in unique_points[1:]:
                        if len(candidates) > len(rev_cover[pt]):
                            candidates = rev_cover[pt]
                        rev_set = set(rev_cover[pt])
                        candidates = [c for c in candidates if c in rev_set]
                        if not candidates:
                            break

                best_gain = -1
                best_word = -1
                for w2 in candidates:
                    if w2 in selected_set:
                        continue
                    m2 = masks[w2]
                    gain = (m2 & ~m).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_word = w2

                if best_word >= 0 and best_gain > 0:
                    # Perform swap
                    for pos in mask_lists[w]:
                        counts[pos] -= 1

                    for pos in mask_lists[best_word]:
                        counts[pos] += 1

                    selected[idx] = best_word
                    selected_set.remove(w)
                    selected_set.add(best_word)
                    improved = True

            if not improved:
                break

        # 2-for-1 swap - optimized with early exit
        if time.time() < time_limit - 5:
            for _ in range(20):
                if time.time() > time_limit:
                    break
                improved = False
                indices = list(range(len(selected)))
                random.shuffle(indices)

                num_pairs_to_try = min(60, len(indices) * (len(indices) - 1) // 2)

                for _ in range(num_pairs_to_try):
                    i_idx = random.randint(0, len(indices) - 1)
                    j_idx = random.randint(0, len(indices) - 1)
                    if i_idx == j_idx:
                        continue
                    idx1 = indices[i_idx]
                    idx2 = indices[j_idx]
                    if idx1 >= len(selected) or idx2 >= len(selected):
                        continue
                    w1 = selected[idx1]
                    w2 = selected[idx2]

                    m1 = masks[w1]
                    m2 = masks[w2]

                    # Calculate gap using list iteration with early exit
                    gap_points = []
                    for pos in mask_lists[w1]:
                        if counts[pos] < 2:
                            gap_points.append(pos)
                    for pos in mask_lists[w2]:
                        if counts[pos] < 2:
                            if counts[pos] < 1:  # Only in w2, not w1
                                if pos not in gap_points:
                                    gap_points.append(pos)
                        elif counts[pos] < 3:
                            # In both w1 and w2
                            if pos not in gap_points:
                                gap_points.append(pos)

                    # Quick check: if too many gaps, skip
                    if len(gap_points) > 15:
                        continue

                    if not gap_points:
                        # 2-for-0 move
                        for pos in mask_lists[w1]:
                            counts[pos] -= 1
                        for pos in mask_lists[w2]:
                            counts[pos] -= 1
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        selected.pop(idx2)
                        selected.pop(idx1)
                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        improved = True
                        break
                    else:
                        # Find w3 that covers all gap_points
                        if len(gap_points) == 1:
                            candidates = rev_cover[gap_points[0]]
                        else:
                            candidates = rev_cover[gap_points[0]]
                            for pt in gap_points[1:]:
                                if len(candidates) > len(rev_cover[pt]):
                                    candidates = rev_cover[pt]
                                rev_set = set(rev_cover[pt])
                                candidates = [c for c in candidates if c in rev_set]
                                if not candidates:
                                    break

                        best_word = -1
                        for w3 in candidates:
                            if w3 in selected_set:
                                continue
                            best_word = w3
                            break

                        if best_word >= 0:
                            for pos in mask_lists[w1]:
                                counts[pos] -= 1
                            for pos in mask_lists[w2]:
                                counts[pos] -= 1
                            for pos in mask_lists[best_word]:
                                counts[pos] += 1

                            if idx1 > idx2:
                                idx1, idx2 = idx2, idx1
                            selected.pop(idx2)
                            selected.pop(idx1)
                            selected.append(best_word)

                            selected_set.remove(w1)
                            selected_set.remove(w2)
                            selected_set.add(best_word)
                            improved = True
                            break

                if improved:
                    selected = polish(selected, time_limit)
                    counts = build_counts(selected)
                    selected_set = set(selected)

        return selected

    def run_greedy_vectorized(randomized=False, alpha=1.0):
        """Vectorized greedy set cover using numpy cover matrix."""
        selected = []
        selected_set = set()
        remaining = np.ones(total_space, dtype=np.uint8)

        while remaining.any() and time.time() < deadline - 10:
            # Vectorized gain: for each word w, gain = sum(cover_matrix[w] * remaining)
            gains = cover_matrix.dot(remaining).astype(np.int32)

            # Mask out already selected
            if selected:
                sel_arr = np.array(selected, dtype=np.int32)
                gains[sel_arr] = -1

            if randomized:
                noise = np.random.normal(0, alpha, size=total_space).astype(np.int32)
                gains = gains + noise

            best_word = int(np.argmax(gains))
            if gains[best_word] <= 0 and not randomized:
                break

            selected.append(best_word)
            selected_set.add(best_word)

            # Update remaining
            remaining[cover_matrix[best_word] == 1] = 0

        return selected

    # Strategy 1: Standard greedy + polish + local search
    if time.time() < deadline - 20:
        sol = run_greedy_vectorized(randomized=False)
        polished = polish(sol[:], deadline - 20)
        ls = local_search(polished[:], deadline - 20)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: Multiple randomized greedy restarts with varied alpha
    restarts = 0
    while time.time() < deadline - 30 and restarts < 80:
        restarts += 1
        # Vary alpha more aggressively for diversity
        alpha = random.uniform(0.05, 1.0)
        sol = run_greedy_vectorized(randomized=True, alpha=alpha)
        if len(sol) < best_size + 15:
            polished = polish(sol[:], deadline - 30)
            ls = local_search(polished[:], deadline - 30)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Final polish and local search on the best solution found
    if time.time() < deadline - 5:
        polished = polish(best_selected[:], deadline - 5)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    if time.time() < deadline - 2:
        final_ls = local_search(best_selected[:], deadline - 2)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_selected = final_ls[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
