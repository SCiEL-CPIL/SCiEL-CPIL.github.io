import time
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    N = q ** n
    if N > 200000:
        return _constructive_cover(q, n, radius)

    # --- Build index mapping ---
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    word_to_idx = {tuple(w): idx for idx, w in enumerate(all_words)}

    # Precompute all balls as numpy arrays (each ball has 1 + n*(q-1) = 17 elements)
    ball_list = []
    for i in range(N):
        w = all_words[i]
        idxs = [i]
        for p in range(n):
            for v in range(q):
                if v == w[p]:
                    continue
                power = q ** (n - 1 - p)
                new_idx = i - w[p] * power + v * power
                idxs.append(new_idx)
        ball_list.append(np.array(idxs, dtype=np.int32))

    # Build ball matrix for vectorized operations: shape (N, 17)
    ball_matrix = np.array(ball_list, dtype=np.int32)

    # --- Coverage tracking ---
    covered_count = np.zeros(N, dtype=np.int32)
    selected_set = set()

    def add_codeword(idx):
        covered_count[ball_matrix[idx]] += 1
        selected_set.add(idx)

    def remove_codeword(idx):
        covered_count[ball_matrix[idx]] -= 1
        selected_set.discard(idx)

    def is_fully_covered():
        return np.all(covered_count > 0)

    # --- Phase 1: Tetracode product seed (81 words) ---
    tetracode = np.array([[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)])
    
    seed_indices = []
    for i in range(9):
        for j in range(9):
            word = np.concatenate([tetracode[i], tetracode[j]])
            idx = word_to_idx[tuple(word)]
            seed_indices.append(idx)

    for idx in seed_indices:
        add_codeword(idx)

    # --- Phase 2: Vectorized greedy fill ---
    if not is_fully_covered():
        while not is_fully_covered() and time.time() < deadline - 10:
            uncovered_mask = (covered_count == 0).astype(np.int32)
            gains = uncovered_mask[ball_matrix].sum(axis=1)
            for idx in selected_set:
                gains[idx] = 0
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            add_codeword(best_idx)

    # --- Phase 3: Iterative Local Search (ILS) ---
    best_size = len(selected_set)
    best_selected = list(selected_set)

    def try_remove_one():
        """Try to remove a redundant codeword. Returns True if removed."""
        for idx in list(selected_set):
            if np.sum(covered_count[ball_matrix[idx]] == 1) == 0:
                remove_codeword(idx)
                return True
        return False

    def greedy_recover():
        """Greedy re-add to restore full coverage using vectorized gains."""
        while not is_fully_covered() and time.time() < deadline - 5:
            uncovered_mask = (covered_count == 0).astype(np.int32)
            gains = uncovered_mask[ball_matrix].sum(axis=1)
            for idx in selected_set:
                gains[idx] = 0
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            add_codeword(best_idx)

    # ILS loop
    rng = np.random.default_rng(42)
    stall_count = 0
    max_stall = 30

    while time.time() < deadline - 8:
        # Try simple removals
        while try_remove_one():
            pass

        if len(selected_set) < best_size:
            best_size = len(selected_set)
            best_selected = list(selected_set)
            stall_count = 0
        else:
            stall_count += 1

        if stall_count >= max_stall:
            # Perturbation: randomly remove k codewords, then greedy re-add
            k = min(len(selected_set), int(rng.integers(2, 7)))
            to_remove = list(rng.choice(list(selected_set), size=k, replace=False))
            for idx in to_remove:
                remove_codeword(idx)
            greedy_recover()
            while try_remove_one():
                pass
            stall_count = 0
            max_stall += 15

        # Try a targeted swap: remove low-damage codeword, add best replacement
        if time.time() < deadline - 8:
            sel_list = list(selected_set)
            if len(sel_list) > 0:
                damages = np.array([np.sum(covered_count[ball_matrix[idx]] == 1) for idx in sel_list])
                sorted_order = np.argsort(damages)
                for pos in sorted_order:
                    idx = sel_list[pos]
                    if damages[pos] == 0:
                        continue
                    if damages[pos] > 5:
                        break
                    remove_codeword(idx)
                    if not is_fully_covered():
                        uncovered_mask = (covered_count == 0).astype(np.int32)
                        gains = uncovered_mask[ball_matrix].sum(axis=1)
                        for sidx in selected_set:
                            gains[sidx] = 0
                        best_idx = int(np.argmax(gains))
                        if gains[best_idx] >= damages[pos]:
                            add_codeword(best_idx)
                            removed = False
                            while try_remove_one():
                                removed = True
                            if not removed:
                                remove_codeword(best_idx)
                                add_codeword(idx)
                        else:
                            add_codeword(idx)
                    break

        if len(selected_set) < best_size:
            best_size = len(selected_set)
            best_selected = list(selected_set)

    # --- Phase 4: Final cleanup ---
    while try_remove_one():
        pass

    if len(selected_set) < best_size:
        best_size = len(selected_set)
        best_selected = list(selected_set)

    if not is_fully_covered():
        covered_count[:] = 0
        selected_set.clear()
        for idx in best_selected:
            add_codeword(idx)
        while not is_fully_covered() and time.time() < deadline:
            uncovered_mask = (covered_count == 0).astype(np.int32)
            gains = uncovered_mask[ball_matrix].sum(axis=1)
            for idx in selected_set:
                gains[idx] = 0
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            add_codeword(best_idx)
        for idx in list(selected_set):
            if np.sum(covered_count[ball_matrix[idx]] == 1) == 0:
                remove_codeword(idx)
        best_selected = list(selected_set)

    result = [all_words[i].tolist() for i in best_selected]
    return result


def _constructive_cover(q, n, radius):
    """Fallback constructive approach."""
    if q == 3 and n == 8:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(h) + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]

    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    num_words = len(all_words)

    balls = np.zeros((num_words, num_words), dtype=bool)
    np.fill_diagonal(balls, True)
    for i in range(num_words):
        for j in range(num_words):
            if np.sum(all_words[i] != all_words[j]) <= radius:
                balls[i, j] = True

    selected = []
    covered = np.zeros(num_words, dtype=bool)
    while not covered.all():
        uncovered_count = balls[:, ~covered].sum(axis=1)
        best_idx = np.argmax(uncovered_count)
        if uncovered_count[best_idx] == 0:
            break
        selected.append(best_idx)
        covered |= balls[best_idx, :]

    return [all_words[i].tolist() for i in selected]
