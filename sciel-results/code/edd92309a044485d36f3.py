import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 142.0

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

    full_mask = (1 << total_space) - 1

    # Precompute masks as numpy array for vectorized operations
    # Convert each mask to a boolean array of length total_space
    masks_bool = np.zeros((total_space, total_space), dtype=np.bool_)
    for i in range(total_space):
        m = masks[i]
        while m:
            lowbit = m & -m
            idx = lowbit.bit_length() - 1
            masks_bool[i, idx] = True
            m ^= lowbit

    best_size = total_space
    best_selected = list(range(total_space))

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def polish(selected, time_limit):
        """Remove redundant codewords iteratively with multiple passes."""
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        changed = True
        passes = 0
        while changed and time.time() < time_limit:
            changed = False
            passes += 1
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]

                # Check if any position in the mask has count == 1
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
        Local search with 1-for-1 swaps and 2-for-1 swaps.
        """
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        selected_set = set(selected)

        # 1-for-1 swap
        for _ in range(80):
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

                unique_mask = 0
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_mask |= lowbit
                    mm ^= lowbit

                if unique_mask == 0:
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

                best_gain = -1
                best_word = -1
                for w2 in range(total_space):
                    if w2 in selected_set:
                        continue
                    m2 = masks[w2]
                    if (m2 & unique_mask) == unique_mask:
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
        if time.time() < time_limit - 8:
            for _ in range(15):
                if time.time() > time_limit:
                    break
                improved = False
                indices = list(range(len(selected)))
                random.shuffle(indices)

                for i_idx in range(min(40, len(indices))):
                    idx1 = indices[i_idx]
                    if idx1 >= len(selected):
                        continue
                    w1 = selected[idx1]

                    idx2 = random.randint(0, len(selected) - 1)
                    if idx2 == idx1:
                        continue
                    w2 = selected[idx2]

                    m1 = masks[w1]
                    m2 = masks[w2]

                    u_ab = 0
                    inter = m1 & m2
                    mm = inter
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 2:
                            u_ab |= lowbit
                        mm ^= lowbit

                    diff1 = m1 & ~m2
                    mm = diff1
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 1:
                            u_ab |= lowbit
                        mm ^= lowbit

                    diff2 = m2 & ~m1
                    mm = diff2
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 1:
                            u_ab |= lowbit
                        mm ^= lowbit

                    if u_ab == 0:
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

                    if u_ab.bit_count() > 8:
                        continue

                    best_word = -1
                    for w3 in range(total_space):
                        if w3 in selected_set:
                            continue
                        m3 = masks[w3]
                        if (m3 & u_ab) == u_ab:
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
                    counts = np.zeros(total_space, dtype=np.int32)
                    for w in selected:
                        mm = masks[w]
                        while mm:
                            lowbit = mm & -mm
                            idx = lowbit.bit_length() - 1
                            counts[idx] += 1
                            mm ^= lowbit
                    selected_set = set(selected)

        return selected

    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover with vectorized gain computation."""
        remaining_bool = np.ones(total_space, dtype=np.bool_)
        selected = []
        selected_set = set()

        while remaining_bool.any() and time.time() < deadline - 8:
            # Vectorized gain computation
            gains = np.count_nonzero(masks_bool & remaining_bool[None, :], axis=1)
            gains = gains.astype(np.float64)
            
            # Mask out already selected words
            for w in selected_set:
                gains[w] = -1
            
            if randomized:
                gains += np.random.normal(0, alpha, size=total_space)
            
            best_word = int(np.argmax(gains))
            
            if gains[best_word] <= 0:
                break

            selected.append(best_word)
            selected_set.add(best_word)
            remaining_bool &= ~masks_bool[best_word]

        return selected

    # Strategy 1: Standard greedy + polish + local search
    if time.time() < deadline - 15:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 15)
        ls = local_search(polished[:], deadline - 15)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: Multiple randomized greedy restarts (more iterations due to speedup)
    max_restarts = 200
    for r in range(max_restarts):
        if time.time() > deadline - 25:
            break

        sol = run_greedy(randomized=True, alpha=random.uniform(0.1, 0.6))
        if len(sol) < best_size + 8:
            polished = polish(sol[:], deadline - 25)
            ls = local_search(polished[:], deadline - 25)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Strategy 3: Two-phase greedy (first cover high-degree words, then fill)
    if time.time() < deadline - 30:
        # Phase 1: greedily pick words that cover many remaining words
        # Phase 2: greedily pick words that cover the fewest remaining (to minimize overlap)
        remaining_bool = np.ones(total_space, dtype=np.bool_)
        selected = []
        selected_set = set()

        # Phase 1: standard greedy until 50% covered
        phase1_limit = int(total_space * 0.5)
        while remaining_bool.any() and len(selected) < phase1_limit and time.time() < deadline - 30:
            gains = np.count_nonzero(masks_bool & remaining_bool[None, :], axis=1)
            gains = gains.astype(np.float64)
            for w in selected_set:
                gains[w] = -1
            best_word = int(np.argmax(gains))
            if gains[best_word] <= 0:
                break
            selected.append(best_word)
            selected_set.add(best_word)
            remaining_bool &= ~masks_bool[best_word]

        # Phase 2: greedily pick words with minimum overlap
        while remaining_bool.any() and time.time() < deadline - 30:
            new_cover = np.count_nonzero(masks_bool & remaining_bool[None, :], axis=1).astype(np.float64)
            covered_bool = ~remaining_bool
            overlap = np.count_nonzero(masks_bool & covered_bool[None, :], axis=1).astype(np.float64)
            scores = new_cover * 10 - overlap
            for w in selected_set:
                scores[w] = -1
            best_word = int(np.argmax(scores))
            if scores[best_word] <= 0:
                break
            selected.append(best_word)
            selected_set.add(best_word)
            remaining_bool &= ~masks_bool[best_word]

        polished = polish(selected[:], deadline - 30)
        ls = local_search(polished[:], deadline - 30)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 4: Weighted greedy with adaptive weights
    if time.time() < deadline - 20:
        # Use a weighted approach where words with higher "uniqueness" get priority
        remaining_bool = np.ones(total_space, dtype=np.bool_)
        selected = []
        selected_set = set()
        
        # Compute initial uniqueness score for each word
        # A word is more unique if it covers positions that are hard to cover
        position_counts = np.zeros(q, dtype=np.int32)
        for i in range(total_space):
            for pos in range(n):
                digit_val = (i // powers[pos]) % q
                position_counts[digit_val] += 1
        
        # Precompute rarity scores for all words
        rarity_scores = np.zeros(total_space, dtype=np.float64)
        for w in range(total_space):
            rs = 0.0
            for pos in range(n):
                digit_val = (w // powers[pos]) % q
                rs += 1.0 / (position_counts[digit_val] + 1)
            rarity_scores[w] = rs
        
        while remaining_bool.any() and time.time() < deadline - 20:
            new_cover = np.count_nonzero(masks_bool & remaining_bool[None, :], axis=1).astype(np.float64)
            scores = new_cover * (1 + 0.1 * rarity_scores)
            for w in selected_set:
                scores[w] = -1
            best_word = int(np.argmax(scores))
            if scores[best_word] <= 0:
                break
            selected.append(best_word)
            selected_set.add(best_word)
            remaining_bool &= ~masks_bool[best_word]

        polished = polish(selected[:], deadline - 20)
        ls = local_search(polished[:], deadline - 20)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Final polish
    if time.time() < deadline - 3:
        polished = polish(best_selected[:], deadline - 3)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    # Final local search pass
    if time.time() < deadline - 2:
        final_ls = local_search(best_selected[:], deadline - 2)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_selected = final_ls[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
