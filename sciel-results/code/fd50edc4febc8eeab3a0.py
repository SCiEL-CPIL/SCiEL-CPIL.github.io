import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    cover_count = 1 + n * (q - 1)

    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        for j, idx in enumerate(idxs):
            neighbors_arr[i, j] = idx

    # Reverse cover: for each word, list of codewords that cover it
    reverse_cover_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    reverse_cover_counts = np.zeros(total_words, dtype=np.int32)
    for i in range(total_words):
        for j in range(cover_count):
            w = neighbors_arr[i, j]
            reverse_cover_arr[w, reverse_cover_counts[w]] = i
            reverse_cover_counts[w] += 1

    # Bit-parallel masks: for each codeword, a Python int whose bit w is set if it covers word w
    # (used for fast 2-1 moves and exact-cover polish)
    cover_bitsets = [0] * total_words
    for i in range(total_words):
        bs = 0
        for w in neighbors_arr[i, :]:
            bs |= (1 << w)
        cover_bitsets[i] = bs

    # Reverse cover as Python-int bitsets (bit i set if codeword i covers word w)
    reverse_bitsets = [0] * total_words
    for i in range(total_words):
        for j in range(cover_count):
            w = neighbors_arr[i, j]
            reverse_bitsets[w] |= (1 << i)

    def greedy_construct_vectorized(random_seed):
        if random_seed is not None:
            random.seed(random_seed)

        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []

        while uncovered.any():
            uncovered_expanded = uncovered[neighbors_arr]
            gains = uncovered_expanded.sum(axis=1)

            valid = gains > 0
            if not valid.any():
                break

            max_gain = gains[valid].max()
            best_candidates = np.where((gains == max_gain) & valid)[0]

            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False

        return code

    def reduce_code_fast(code):
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1

        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbors_arr[c, :]] -= 1
                if np.any(counts == 0):
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    def local_search_improve(code, deadline):
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1

        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)

            for i in indices:
                if time.time() >= deadline:
                    break

                c_remove = code[i]
                counts[neighbors_arr[c_remove, :]] -= 1

                if np.any(counts == 0):
                    counts[neighbors_arr[c_remove, :]] += 1
                    critical_words = neighbors_arr[c_remove, :][counts[neighbors_arr[c_remove, :]] == 1]
                    if len(critical_words) == 0:
                        continue

                    # Bit-parallel intersection of reverse covers
                    cand_mask = None
                    for cw in critical_words:
                        m = reverse_bitsets[cw]
                        if cand_mask is None:
                            cand_mask = m
                        else:
                            cand_mask &= m
                    if cand_mask is None or cand_mask == 0:
                        continue
                    cand_mask &= ~(1 << c_remove)
                    if cand_mask == 0:
                        continue

                    # Pick a random set bit
                    m = cand_mask
                    low_bit = m & (-m)
                    best_w = low_bit.bit_length() - 1

                    code[i] = best_w
                    counts[neighbors_arr[c_remove, :]] -= 1
                    counts[neighbors_arr[best_w, :]] += 1
                    improved = True
                    break
                else:
                    code.pop(i)
                    improved = True
                    break

        return code

    def local_search_21_move(code, deadline):
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1

        improved = True
        while improved and time.time() < deadline:
            improved = False
            if len(code) < 2:
                break

            indices = list(range(len(code)))
            random.shuffle(indices)

            for i in indices:
                if time.time() >= deadline:
                    break
                for j in indices:
                    if time.time() >= deadline:
                        break
                    if j <= i:
                        continue

                    c1 = code[i]
                    c2 = code[j]

                    counts[neighbors_arr[c1, :]] -= 1
                    counts[neighbors_arr[c2, :]] -= 1

                    if np.any(counts == 0):
                        uncovered_words = np.where(counts == 0)[0]

                        # Bit-parallel intersection of reverse covers
                        cand_mask = None
                        for uw in uncovered_words:
                            m = reverse_bitsets[uw]
                            if cand_mask is None:
                                cand_mask = m
                            else:
                                cand_mask &= m

                        if cand_mask is None or cand_mask == 0:
                            counts[neighbors_arr[c1, :]] += 1
                            counts[neighbors_arr[c2, :]] += 1
                            continue

                        cand_mask &= ~(1 << c1)
                        cand_mask &= ~(1 << c2)

                        if cand_mask == 0:
                            counts[neighbors_arr[c1, :]] += 1
                            counts[neighbors_arr[c2, :]] += 1
                            continue

                        # Pick a random candidate
                        m = cand_mask
                        low_bit = m & (-m)
                        new_c = low_bit.bit_length() - 1

                        if j > i:
                            code.pop(j)
                            code.pop(i)
                        else:
                            code.pop(i)
                            code.pop(j)
                        code.append(new_c)
                        counts[neighbors_arr[new_c, :]] += 1
                        improved = True
                        break
                    else:
                        if j > i:
                            code.pop(j)
                            code.pop(i)
                        else:
                            code.pop(i)
                            code.pop(j)
                        improved = True
                        break

                if improved:
                    break

        return code

    def exact_cover_residual(code, deadline):
        """
        Bit-parallel Algorithm X on the residual problem:
        given the current code, find a minimum set of *additional* codewords
        that covers any currently-uncovered words. Uses column selection by
        minimum candidate count (the standard Algorithm X heuristic), which
        drastically prunes because each uncovered word has only a few
        candidate codewords (cover_count of them).
        """
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1

        uncovered = np.where(counts == 0)[0]
        if len(uncovered) == 0:
            return code

        # If there are too many uncovered words, the exact search may explode;
        # fall back to greedy repair in that case.
        if len(uncovered) > 30:
            # greedy repair
            while np.any(counts == 0) and time.time() < deadline:
                unc = np.where(counts == 0)[0]
                gains = np.zeros(total_words, dtype=np.int32)
                for uw in unc:
                    rc = reverse_cover_arr[uw, :reverse_cover_counts[uw]]
                    gains[rc] += 1
                code_set = set(code)
                for c in code_set:
                    gains[c] = 0
                if gains.max() == 0:
                    break
                best_idx = int(np.argmax(gains))
                code.append(best_idx)
                counts[neighbors_arr[best_idx, :]] += 1
            return code

        # Build the exact-cover instance.
        # Columns = uncovered words (0..U-1)
        # Rows = candidate codewords that cover at least one uncovered word
        U = len(uncovered)
        uncovered_set = set(uncovered.tolist())

        # For each candidate codeword, the set of uncovered-word columns it covers
        # Represent each row as a bitmask over the U columns.
        # We only need to consider codewords that cover >= 1 uncovered word.
        rows = []  # list of (codeword_idx, col_mask)
        for cw in range(total_words):
            bs = cover_bitsets[cw]
            # Intersect with uncovered columns
            col_mask = 0
            for k, uw in enumerate(uncovered):
                if (bs >> uw) & 1:
                    col_mask |= (1 << k)
            if col_mask != 0:
                rows.append((cw, col_mask))

        if not rows:
            return code

        # Algorithm X with bit-parallel column masks.
        # State: remaining_columns (bitmask over U columns still needing coverage)
        # We track the set of active rows implicitly; for pruning we just
        # backtrack over rows. To keep it fast we use a recursive search with
        # memoization on (remaining_columns, step_limit) is overkill; instead
        # we do a bounded DFS with column selection by fewest candidates.

        # Precompute for each column (uncovered word) the list of rows covering it
        col_to_rows = [[] for _ in range(U)]
        for ri, (cw, col_mask) in enumerate(rows):
            m = col_mask
            while m:
                lb = m & (-m)
                k = lb.bit_length() - 1
                col_to_rows[k].append(ri)
                m &= m - 1

        all_cols_mask = (1 << U) - 1
        best_solution = None
        best_size = U + 1  # we want to minimize number of added codewords
        node_budget = 200000
        nodes = [0]

        def dfs(remaining_cols, chosen, depth):
            if nodes[0] >= node_budget:
                return
            nodes[0] += 1
            if remaining_cols == 0:
                if len(chosen) < best_size:
                    best_size = len(chosen)
                    best_solution = list(chosen)
                return
            # Prune: even if each remaining column is covered by one new codeword,
            # we can't beat best_size
            if len(chosen) + 1 >= best_size:
                return

            # Column selection: pick the column with fewest covering rows
            # (among rows that still can cover it, i.e. rows not excluded).
            # We don't track excluded rows explicitly; we rely on the fact that
            # once we commit a row, we remove its columns. For simplicity and
            # correctness we pick the column with minimum |col_to_rows[k]|.
            min_k = -1
            min_count = 10**9
            m = remaining_cols
            while m:
                lb = m & (-m)
                k = lb.bit_length() - 1
                c = len(col_to_rows[k])
                if c < min_count:
                    min_count = c
                    min_k = k
                m &= m - 1

            if min_k < 0:
                return

            # Try each row covering column min_k
            for ri in col_to_rows[min_k]:
                cw, col_mask = rows[ri]
                chosen.append(cw)
                dfs(remaining_cols & ~col_mask, chosen, depth + 1)
                chosen.pop()
                if nodes[0] >= node_budget:
                    return
                if best_size <= len(chosen) + 1:
                    # can't improve further with this branch
                    continue

        dfs(all_cols_mask, [], 0)

        if best_solution is not None:
            code.extend(best_solution)
        return code

    def perturbation_repair(code, deadline):
        code = list(code)
        if len(code) < 5:
            return code

        num_remove = max(1, len(code) // 10)
        to_remove = random.sample(range(len(code)), min(num_remove, len(code)))
        to_remove.sort(reverse=True)
        for idx in to_remove:
            code.pop(idx)

        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1

        while np.any(counts == 0) and time.time() < deadline:
            uncovered = np.where(counts == 0)[0]
            gains = np.zeros(total_words, dtype=np.int32)
            for uw in uncovered:
                rc = reverse_cover_arr[uw, :reverse_cover_counts[uw]]
                gains[rc] += 1

            code_set = set(code)
            for c in code_set:
                gains[c] = 0

            if gains.max() == 0:
                break

            best_idx = int(np.argmax(gains))
            code.append(best_idx)
            counts[neighbors_arr[best_idx, :]] += 1

        return code

    start_time = time.time()
    deadline = start_time + 140  # safety margin under 150s kill

    best_code = None
    best_size = float('inf')

    seed = 0
    while time.time() < deadline:
        code = greedy_construct_vectorized(seed)
        code = reduce_code_fast(code)

        if time.time() < deadline - 25:
            code = local_search_improve(code, deadline - 20)
            code = reduce_code_fast(code)

        if time.time() < deadline - 15:
            code = local_search_21_move(code, deadline - 12)
            code = reduce_code_fast(code)

        if time.time() < deadline - 8:
            code = local_search_improve(code, deadline - 6)
            code = reduce_code_fast(code)

        # Exact-cover residual polish: fix any gaps minimally, or confirm none
        if time.time() < deadline - 3:
            code = exact_cover_residual(code, deadline - 2)
            code = reduce_code_fast(code)

        if time.time() < deadline - 2:
            code = perturbation_repair(code, deadline - 1)
            code = reduce_code_fast(code)
            if time.time() < deadline - 0.5:
                code = exact_cover_residual(code, deadline - 0.2)
                code = reduce_code_fast(code)

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        seed += 1

    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))

    return result
