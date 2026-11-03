"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
import time
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses NumPy-vectorized greedy set-cover for fast initialization, then iterative
    local search (redundancy removal + targeted swaps) to minimize codeword count.
    """
    if n == 0:
        return [[]]

    total_words = q ** n
    start_time = time.time()
    total_time_limit = 115  # safety margin before 150s kill

    # --- Precompute neighbors ---
    # For q=3, n=8, radius=1: each word has 1 + 8*2 = 17 neighbors
    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        word.reverse()
        return word

    def word_to_int(word, q):
        val = 0
        for d in word:
            val = val * q + d
        return val

    neighbors = [None] * total_words

    if radius == 1:
        # Fast path: precompute digit arrays
        digits_all = np.zeros((total_words, n), dtype=np.int32)
        for w in range(total_words):
            tmp = w
            for pos in range(n - 1, -1, -1):
                digits_all[w, pos] = tmp % q
                tmp //= q

        for w in range(total_words):
            neigh = [w]
            word = digits_all[w].tolist()
            for pos in range(n):
                cur = word[pos]
                for val in range(q):
                    if val == cur:
                        continue
                    word[pos] = val
                    neigh.append(word_to_int(word, q))
                word[pos] = cur
            neighbors[w] = set(neigh)
    else:
        # General case
        for w in range(total_words):
            word = int_to_word(w, n, q)
            neigh = [w]
            if radius >= 1:
                for pos in range(n):
                    cur = word[pos]
                    for val in range(q):
                        if val == cur:
                            continue
                        word[pos] = val
                        neigh.append(word_to_int(word, q))
                    word[pos] = cur
            if radius >= 2:
                word = int_to_word(w, n, q)
                for pos1 in range(n):
                    c1 = word[pos1]
                    for val1 in range(q):
                        if val1 == c1:
                            continue
                        word[pos1] = val1
                        for pos2 in range(pos1 + 1, n):
                            c2 = word[pos2]
                            for val2 in range(q):
                                if val2 == c2:
                                    continue
                                word[pos2] = val2
                                neigh.append(word_to_int(word, q))
                            word[pos2] = c2
                        word[pos1] = c1
            neighbors[w] = set(neigh)

    # Build NumPy matrix: neighbor_matrix[i, j] = 1 if j is in neighbors[i]
    # For memory efficiency with 6561x6561, use uint8
    neighbor_matrix = np.zeros((total_words, total_words), dtype=np.uint8)
    for w in range(total_words):
        for u in neighbors[w]:
            neighbor_matrix[w, u] = 1

    # Precompute neighbor lists as numpy arrays for fast indexing
    neighbor_arrays = []
    for w in range(total_words):
        neighbor_arrays.append(np.array(list(neighbors[w]), dtype=np.int32))

    def greedy_cover_numpy():
        """
        NumPy-vectorized greedy set cover.
        At each step, compute marginal gain for all candidates simultaneously.
        """
        covered = np.zeros(total_words, dtype=np.uint8)
        codewords = []
        uncovered_count = total_words

        while uncovered_count > 0:
            # Compute marginal gain for all candidates:
            # gain[w] = sum of neighbor_matrix[w, :] * (1 - covered)
            # This is a matrix-vector product
            uncovered_mask = 1.0 - covered.astype(np.float32)
            gains = neighbor_matrix.astype(np.float32) @ uncovered_mask

            # Only consider candidates that are themselves uncovered (or any)
            # Actually, we can pick any word as a codeword
            best_word = int(np.argmax(gains))
            best_gain = int(gains[best_word])

            if best_gain == 0:
                break

            codewords.append(best_word)
            # Update covered
            covered[neighbor_arrays[best_word]] = 1
            uncovered_count -= best_gain

            # Check time
            if time.time() - start_time > total_time_limit - 5:
                break

        return codewords

    def local_search(codewords, time_limit):
        """
        Local search: remove redundant codewords, then swap to improve.
        Uses coverage count array for O(1) redundancy checks.
        """
        # Build coverage count
        cov_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            cov_count[neighbor_arrays[cw]] += 1

        cw_set = set(codewords)

        def try_remove_redundant():
            """Remove codewords where all neighbors have cov_count > 1."""
            removed = False
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    # Check if all neighbors of cw have coverage >= 2
                    nb = neighbor_arrays[cw]
                    if np.all(cov_count[nb] >= 2):
                        cov_count[nb] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        removed = True
                        changed = True
                        break
            return removed

        try_remove_redundant()

        start_ls = time.time()

        while time.time() - start_ls < time_limit:
            if try_remove_redundant():
                continue

            if not codewords:
                break

            # Pick a random codeword
            idx = random.randrange(len(codewords))
            old_cw = codewords[idx]

            # Find words uniquely covered by old_cw
            nb_old = neighbor_arrays[old_cw]
            unique_mask = cov_count[nb_old] == 1
            unique_covered = nb_old[unique_mask]

            if len(unique_covered) == 0:
                # Can remove entirely
                cov_count[nb_old] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue

            # Find replacement: must cover all unique_covered
            # Candidates: intersection of neighbors of each unique_covered word
            # For efficiency, use the first unique word's neighbors as starting set
            candidates_set = set(neighbors[int(unique_covered[0])])
            for u in unique_covered[1:]:
                candidates_set &= set(neighbors[int(u)])
                if not candidates_set:
                    break

            if not candidates_set:
                continue

            candidates = [c for c in candidates_set if c not in cw_set and c != old_cw]
            if not candidates:
                continue

            random.shuffle(candidates)
            # Limit candidates for speed
            candidates = candidates[:50]

            best_candidate = -1
            best_extra = -1
            nb_old_set = set(neighbors[old_cw])

            for new_cw in candidates:
                nb_new = neighbor_arrays[new_cw]
                # Extra coverage: words in nb_new not in nb_old_set
                extra = 0
                for u in nb_new:
                    if u not in nb_old_set:
                        extra += 1
                if extra > best_extra:
                    best_extra = extra
                    best_candidate = new_cw

            if best_candidate != -1:
                cov_count[nb_old] -= 1
                nb_new = neighbor_arrays[best_candidate]
                cov_count[nb_new] += 1
                codewords[idx] = best_candidate
                cw_set.discard(old_cw)
                cw_set.add(best_candidate)

        try_remove_redundant()
        return codewords

    # --- Main: multiple restarts ---
    best_codewords = None
    best_len = float('inf')

    while time.time() - start_time < total_time_limit - 3:
        remaining = total_time_limit - (time.time() - start_time)
        if remaining < 8:
            break

        # Greedy init (vectorized)
        cw = greedy_cover_numpy()

        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw[:]

        # Local search with remaining time (leave 3s margin)
        ls_time = min(remaining - 3, 25)
        if ls_time > 2:
            cw = local_search(cw, ls_time)

        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw[:]

        if best_len <= 403:
            break

    if best_codewords is None:
        best_codewords = greedy_cover_numpy()

    # Convert to list of lists
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
