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

    # Build numpy cover matrix
    cover_matrix = np.zeros((total_space, total_space), dtype=np.uint8)
    for w in range(total_space):
        mm = masks[w]
        while mm:
            lowbit = mm & -mm
            idx = lowbit.bit_length() - 1
            cover_matrix[w, idx] = 1
            mm ^= lowbit

    # Also build reverse: for each point x, which words cover it
    rev_cover = [None] * total_space
    for x in range(total_space):
        rev_cover[x] = np.where(cover_matrix[:, x] == 1)[0].tolist()

    # Precompute bit counts for masks
    mask_sizes = np.array([masks[i].bit_count() for i in range(total_space)], dtype=np.int32)

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
                m = masks[w]
                removable = True
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        removable = False
                        break
                    mm ^= lowbit

                if removable:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
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

                unique_points = []
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_points.append(pos)
                    mm ^= lowbit

                if not unique_points:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    selected_set.remove(w)
                    improved = True
                    continue

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
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit

                    mm = masks[best_word]
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit

                    selected[idx] = best_word
                    selected_set.remove(w)
                    selected_set.add(best_word)
                    improved = True

            if not improved:
                break

        # 2-for-1 swap
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
                    m12 = m1 | m2

                    gap_points = []
                    mm = m12
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        in_m1 = (m1 & lowbit) != 0
                        in_m2 = (m2 & lowbit) != 0
                        if in_m1 and in_m2:
                            if counts[pos] < 3:
                                gap_points.append(pos)
                        else:
                            if counts[pos] < 2:
                                gap_points.append(pos)
                        mm ^= lowbit

                    if not gap_points:
                        mm = m1
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        mm = m2
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        selected.pop(idx2)
                        selected.pop(idx1)
                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        improved = True
                        break
                    else:
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
                            mm = m1
                            while mm:
                                lowbit = mm & -mm
                                pos = lowbit.bit_length() - 1
                                counts[pos] -= 1
                                mm ^= lowbit
                            mm = m2
                            while mm:
                                lowbit = mm & -mm
                                pos = lowbit.bit_length() - 1
                                counts[pos] -= 1
                                mm ^= lowbit
                            mm = masks[best_word]
                            while mm:
                                lowbit = mm & -mm
                                pos = lowbit.bit_length() - 1
                                counts[pos] += 1
                                mm ^= lowbit

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

    def run_greedy_vectorized(randomized=False, alpha=1.0, temperature=1.0):
        """Vectorized greedy set cover using numpy cover matrix."""
        selected = []
        selected_set = set()
        remaining = np.ones(total_space, dtype=np.uint8)

        while remaining.any() and time.time() < deadline - 10:
            gains = cover_matrix.dot(remaining).astype(np.int32)

            if selected:
                sel_arr = np.array(selected, dtype=np.int32)
                gains[sel_arr] = -1

            if randomized:
                # Soft-max selection: use temperature to control randomness
                # Higher temperature = more random selection
                if temperature > 0:
                    # Use softmax-like selection
                    max_gain = np.max(gains[gains >= 0]) if np.any(gains >= 0) else 0
                    if max_gain > 0:
                        # Create probabilities based on gain
                        adjusted = gains - max_gain
                        probs = np.exp(adjusted / temperature)
                        probs[gains < 0] = 0
                        total = probs.sum()
                        if total > 0:
                            probs = probs / total
                            best_word = np.random.choice(total_space, p=probs)
                            best_word = int(best_word)
                        else:
                            best_word = int(np.argmax(gains))
                    else:
                        break
                else:
                    noise = np.random.normal(0, alpha, size=total_space).astype(np.int32)
                    gains = gains + noise
                    best_word = int(np.argmax(gains))
            else:
                best_word = int(np.argmax(gains))
                if gains[best_word] <= 0:
                    break

            if gains[best_word] <= 0 and not randomized:
                break

            selected.append(best_word)
            selected_set.add(best_word)

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

    # Strategy 2: Multiple randomized greedy restarts with varying temperature
    restarts = 0
    while time.time() < deadline - 30 and restarts < 55:
        restarts += 1
        # Vary temperature for more diverse solutions
        temp = random.uniform(0.5, 2.0)
        sol = run_greedy_vectorized(randomized=True, temperature=temp)
        if len(sol) < best_size + 10:
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
