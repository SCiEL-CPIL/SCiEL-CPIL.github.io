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

    # Precompute digit arrays: digits[i, pos] = digit of word i at position pos
    digits = np.zeros((total_space, n), dtype=np.int32)
    for pos in range(n):
        for i in range(total_space):
            digits[i, pos] = (i // powers[pos]) % q

    # Precompute neighbor lists: for each word, list of words within distance 1
    # Use a list of arrays for efficiency
    neighbors = []
    for i in range(total_space):
        nbrs = [i]
        for pos in range(n):
            dval = digits[i, pos]
            for nv in range(q):
                if nv == dval:
                    continue
                j = i + (nv - dval) * powers[pos]
                nbrs.append(j)
        neighbors.append(np.array(nbrs, dtype=np.int32))

    # Precompute masks as numpy boolean arrays for vectorized operations
    # mask[i] is a boolean array of length total_space
    # This uses ~2187*2187 bytes = ~4.8MB, acceptable
    mask = np.zeros((total_space, total_space), dtype=np.bool_)
    for i in range(total_space):
        mask[i, neighbors[i]] = True

    full_mask = np.ones(total_space, dtype=np.bool_)

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def polish(selected, time_limit):
        """Remove redundant codewords using vectorized counting."""
        selected = list(selected)
        if not selected:
            return selected

        sel_arr = np.array(selected, dtype=np.int32)
        counts = mask[sel_arr].sum(axis=0).astype(np.int32)

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            # Find removable words: those whose mask has no position with count==1
            sel_arr = np.array(selected, dtype=np.int32)
            # For each selected word, check if any neighbor has count==1
            # mask[sel_arr] has shape (len(selected), total_space)
            # counts has shape (total_space,)
            # A word is removable if no neighbor position has count==1
            m = mask[sel_arr]  # (k, total_space)
            # positions with count==1
            unique_pos = (counts == 1)  # (total_space,)
            # For each selected word, check if any of its neighbors is a unique position
            has_unique = (m & unique_pos).any(axis=1)  # (k,)
            removable = ~has_unique

            if removable.any():
                # Remove all removable words at once
                keep_mask = np.ones(len(selected), dtype=np.bool_)
                keep_mask[removable] = False
                removed_arr = sel_arr[removable]
                counts -= mask[removed_arr].sum(axis=0).astype(np.int32)
                selected = [selected[i] for i in range(len(selected)) if keep_mask[i]]
                changed = True

        return selected

    def local_search(selected, time_limit):
        """
        Local search with 1-for-1 swaps and 2-for-1 swaps using vectorized ops.
        """
        selected = list(selected)
        if not selected:
            return selected

        sel_arr = np.array(selected, dtype=np.int32)
        counts = mask[sel_arr].sum(axis=0).astype(np.int32)
        selected_set = set(selected)

        # 1-for-1 swap
        for _ in range(100):
            if time.time() > time_limit:
                break
            improved = False
            sel_arr = np.array(selected, dtype=np.int32)
            k = len(selected)
            if k == 0:
                break

            # For each selected word, compute unique_mask (neighbors with count==1)
            m = mask[sel_arr]  # (k, total_space)
            unique_pos = (counts == 1)  # (total_space,)
            has_unique = (m & unique_pos).any(axis=1)  # (k,)

            # Words with no unique neighbors can be removed directly
            removable = ~has_unique
            if removable.any():
                removed_arr = sel_arr[removable]
                counts -= mask[removed_arr].sum(axis=0).astype(np.int32)
                for idx in np.where(removable)[0]:
                    w = selected[idx]
                    selected.pop(idx)
                    selected_set.remove(w)
                improved = True
                sel_arr = np.array(selected, dtype=np.int32)
                continue

            # For words with unique neighbors, try to find a replacement
            # Compute unique_mask for each word
            unique_masks = (m & unique_pos)  # (k, total_space) boolean

            # For each candidate word w2 not in selected, check if it covers all unique positions
            # and compute gain
            # This is expensive; sample candidates
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if time.time() > time_limit:
                    break
                if idx >= len(selected):
                    continue
                w = selected[idx]
                um = unique_masks[idx]  # boolean array of unique positions for this word

                if not um.any():
                    # Can remove directly
                    counts -= mask[w]
                    selected.pop(idx)
                    selected_set.remove(w)
                    improved = True
                    continue

                # Find best replacement: a word not in selected that covers all um positions
                # Sample a subset of candidates for speed
                candidates = np.random.choice(total_space, size=min(500, total_space), replace=False)
                candidates = [c for c in candidates if c not in selected_set]

                best_gain = -1
                best_word = -1
                for w2 in candidates:
                    m2 = mask[w2]
                    if not (m2 & um).all():
                        continue
                    gain = (m2 & ~mask[w]).sum()
                    if gain > best_gain:
                        best_gain = gain
                        best_word = w2

                if best_word >= 0 and best_gain > 0:
                    counts -= mask[w]
                    counts += mask[best_word]
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
                k = len(selected)
                if k < 2:
                    break

                sel_arr = np.array(selected, dtype=np.int32)
                counts = mask[sel_arr].sum(axis=0).astype(np.int32)
                selected_set = set(selected)

                # Try random pairs
                for _ in range(30):
                    if time.time() > time_limit:
                        break
                    idx1 = random.randint(0, k - 1)
                    idx2 = random.randint(0, k - 1)
                    if idx1 == idx2:
                        continue
                    w1 = selected[idx1]
                    w2 = selected[idx2]

                    m1 = mask[w1]
                    m2 = mask[w2]

                    # Positions uniquely covered by {w1, w2}
                    # count==2 in intersection, count==1 in symmetric difference
                    inter = m1 & m2
                    sym_diff = m1 ^ m2

                    u_ab = np.zeros(total_space, dtype=np.bool_)
                    u_ab[inter & (counts == 2)] = True
                    u_ab[sym_diff & (counts == 1)] = True

                    if not u_ab.any():
                        # Can remove both
                        counts -= m1
                        counts -= m2
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        selected.pop(idx2)
                        selected.pop(idx1)
                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        improved = True
                        break

                    if u_ab.sum() > 10:
                        continue

                    # Find replacement
                    candidates = np.random.choice(total_space, size=min(300, total_space), replace=False)
                    candidates = [c for c in candidates if c not in selected_set]

                    best_word = -1
                    for w3 in candidates:
                        m3 = mask[w3]
                        if (m3 & u_ab).all():
                            best_word = w3
                            break

                    if best_word >= 0:
                        counts -= m1
                        counts -= m2
                        counts += mask[best_word]
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

        return selected

    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover with vectorized gain computation."""
        covered = np.zeros(total_space, dtype=np.bool_)
        selected = []
        remaining = full_mask.copy()
        selected_set = set()

        # Precompute mask as list of arrays for faster access
        while remaining.any() and time.time() < deadline - 10:
            # Compute gains for all words not yet selected
            # gain[w] = number of remaining positions covered by word w
            # This is mask[w] & remaining, summed
            # Vectorized: for each w, sum of (mask[w] & remaining)
            # But mask is (total_space, total_space), so we can do:
            # gains = (mask & remaining[None, :]).sum(axis=1)
            gains = (mask & remaining[None, :]).sum(axis=1)
            gains[list(selected_set)] = -1  # exclude already selected

            if randomized:
                noise = np.random.normal(0, alpha, size=total_space)
                gains = gains + noise

            best_word = int(np.argmax(gains))
            if gains[best_word] <= 0:
                break

            selected.append(best_word)
            selected_set.add(best_word)
            covered |= mask[best_word]
            remaining = full_mask & ~covered

        return selected

    def run_greedy_weighted():
        """Weighted greedy: prefer words covering rare positions."""
        covered = np.zeros(total_space, dtype=np.bool_)
        selected = []
        remaining = full_mask.copy()
        selected_set = set()

        # Position rarity: how many words cover each position
        # Initially all positions are equally rare
        # As we select words, positions covered by many words become less "rare"
        # Use inverse count as weight
        pos_weight = np.ones(total_space, dtype=np.float64) / total_space

        while remaining.any() and time.time() < deadline - 10:
            # Weighted gain: sum of pos_weight over remaining positions covered by w
            # gain[w] = sum(pos_weight * (mask[w] & remaining))
            # Vectorized
            weighted_remaining = pos_weight * remaining  # (total_space,)
            gains = (mask * weighted_remaining[None, :]).sum(axis=1)
            gains[list(selected_set)] = -1

            best_word = int(np.argmax(gains))
            if gains[best_word] <= 0:
                break

            selected.append(best_word)
            selected_set.add(best_word)
            covered |= mask[best_word]
            remaining = full_mask & ~covered

            # Update weights: positions now covered by more words get lower weight
            counts = mask[sel_arr if 'sel_arr' in dir() else np.array(selected)].sum(axis=0)
            pos_weight = 1.0 / (counts.astype(np.float64) + 1)

        return selected

    best_size = total_space
    best_selected = list(range(total_space))

    # Strategy 1: Standard greedy + polish + local search
    if time.time() < deadline - 20:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 20)
        ls = local_search(polished[:], deadline - 20)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: Multiple randomized greedy restarts
    n_restarts = 0
    for r in range(200):
        if time.time() > deadline - 30:
            break
        n_restarts += 1

        alpha = random.uniform(0.1, 0.8)
        sol = run_greedy(randomized=True, alpha=alpha)
        if len(sol) < best_size + 10:
            polished = polish(sol[:], deadline - 30)
            ls = local_search(polished[:], deadline - 30)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Strategy 3: Weighted greedy
    if time.time() < deadline - 25:
        sol = run_greedy_weighted()
        polished = polish(sol[:], deadline - 25)
        ls = local_search(polished[:], deadline - 25)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 4: Two-phase greedy
    if time.time() < deadline - 25:
        covered = np.zeros(total_space, dtype=np.bool_)
        selected = []
        remaining = full_mask.copy()
        selected_set = set()

        # Phase 1: standard greedy until 60% covered
        phase1_limit = int(total_space * 0.6)
        while remaining.any() and len(selected) < phase1_limit and time.time() < deadline - 25:
            gains = (mask & remaining[None, :]).sum(axis=1)
            gains[list(selected_set)] = -1
            best_word = int(np.argmax(gains))
            if gains[best_word] <= 0:
                break
            selected.append(best_word)
            selected_set.add(best_word)
            covered |= mask[best_word]
            remaining = full_mask & ~covered

        # Phase 2: minimum overlap greedy
        while remaining.any() and time.time() < deadline - 25:
            new_cover = (mask & remaining[None, :]).sum(axis=1)
            overlap = (mask & covered[None, :]).sum(axis=1)
            score = new_cover * 10 - overlap
            score[list(selected_set)] = -1
            best_word = int(np.argmax(score))
            if score[best_word] <= 0:
                break
            selected.append(best_word)
            selected_set.add(best_word)
            covered |= mask[best_word]
            remaining = full_mask & ~covered

        polished = polish(selected[:], deadline - 25)
        ls = local_search(polished[:], deadline - 25)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Final polish and local search
    if time.time() < deadline - 5:
        polished = polish(best_selected[:], deadline - 5)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    if time.time() < deadline - 3:
        final_ls = local_search(best_selected[:], deadline - 3)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_selected = final_ls[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
