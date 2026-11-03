import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n

    # ---- index <-> tuple helpers ------------------------------------------
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

    # ---- precompute Hamming ball (radius) of every word -------------------
    # radius == 1  ->  1 + n*(q-1)
    if radius == 1:
        cover_count = 1 + n * (q - 1)
    else:
        from itertools import product
        cover_count = 1
        for r in range(1, radius + 1):
            from math import comb
            cover_count += comb(n, r) * (q - 1) ** r

    neighbors = np.zeros((total_words, cover_count), dtype=np.int32)
    for i in range(total_words):
        w = idx_to_tuple(i)
        j = 0
        neighbors[i, j] = i
        j += 1
        for pos in range(n):
            ov = w[pos]
            for nv in range(q):
                if nv == ov:
                    continue
                w[pos] = nv
                neighbors[i, j] = tuple_to_idx(w)
                j += 1
            w[pos] = ov

    # ---- reverse cover: which words' balls contain each word --------------
    reverse_cover = [[] for _ in range(total_words)]
    for i in range(total_words):
        for k in range(cover_count):
            reverse_cover[neighbors[i, k]].append(i)
    reverse_cover_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    reverse_cover_counts = np.zeros(total_words, dtype=np.int32)
    for w in range(total_words):
        rc = reverse_cover[w]
        for k, ci in enumerate(rc):
            reverse_cover_arr[w, k] = ci
        reverse_cover_counts[w] = len(rc)

    # ---- greedy construction (vectorized) --------------------------------
    def greedy_construct(random_seed, smart_tiebreak):
        if random_seed is not None:
            random.seed(random_seed)
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        while uncovered.any():
            gains = uncovered[neighbors].sum(axis=1)
            valid = gains > 0
            if not valid.any():
                break
            max_gain = gains[valid].max()
            best = np.where((gains == max_gain) & valid)[0]
            if smart_tiebreak and len(best) > 1:
                # prefer words that overlap least with already-chosen words
                chosen_set = set(code)
                overlaps = np.array(
                    [len(set(neighbors[c].tolist()) & chosen_set) for c in best.tolist()],
                    dtype=np.int32)
                min_ov = overlaps.min()
                best = best[overlaps == min_ov]
            chosen = int(best[random.randrange(len(best))])
            code.append(chosen)
            uncovered[neighbors[chosen]] = False
        return code

    # ---- fast reduction ---------------------------------------------------
    def reduce_code(code):
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbors[c]] -= 1
                if not np.any(counts == 0):
                    # removable -> drop it
                    changed = True
                else:
                    # try to swap for a word that covers all its unique words
                    keep = True
                    covered = counts[neighbors[c]] > 0
                    unique = neighbors[c][~covered]
                    if len(unique) > 0:
                        cand = None
                        for cw in unique:
                            rc = reverse_cover_arr[cw, :reverse_cover_counts[cw]]
                            if cand is None:
                                cand = set(rc.tolist())
                            else:
                                cand &= set(rc.tolist())
                        if cand:
                            cand.discard(c)
                            if cand:
                                # pick the candidate with the largest total coverage
                                best = max(cand, key=lambda x: counts[neighbors[x]].sum())
                                counts[neighbors[c]] -= 1
                                counts[neighbors[best]] += 1
                                new_code.append(best)
                                changed = True
                                continue
                    # cannot remove nor swap -> keep c
                    counts[neighbors[c]] += 1
                    new_code.append(c)
                    keep = False
            code = new_code
        return code

    # ---- local search (remove or swap to fix) ----------------------------
    def local_search(code, deadline):
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors[c]] += 1
        improved = True
        while improved and time.time() < deadline:
            improved = False
            order = list(range(len(code)))
            random.shuffle(order)
            for i in order:
                if time.time() >= deadline:
                    break
                c = code[i]
                counts[neighbors[c]] -= 1
                if not np.any(counts == 0):
                    code.pop(i)
                    improved = True
                    break
                # find replacement covering the words that become uncovered
                covered = counts[neighbors[c]] > 0
                unique = neighbors[c][~covered]
                cand = None
                for cw in unique:
                    rc = reverse_cover_arr[cw, :reverse_cover_counts[cw]]
                    if cand is None:
                        cand = set(rc.tolist())
                    else:
                        cand &= set(rc.tolist())
                if cand:
                    cand.discard(c)
                    if cand:
                        # pick replacement that minimizes new uncovered words
                        best = min(cand, key=lambda x: counts[neighbors[x]].sum())
                        counts[neighbors[c]] -= 1
                        counts[neighbors[best]] += 1
                        code[i] = best
                        improved = True
                        break
                else:
                    counts[neighbors[c]] += 1
        return code

    # ---- driver with explicit deadline guard -----------------------------
    start_time = time.time()
    deadline = start_time + 140
    best_code = None
    best_size = float('inf')
    seed = 0
    while time.time() < deadline:
        if seed == 0:
            code = greedy_construct(seed, False)
        else:
            code = greedy_construct(seed, True)
        code = reduce_code(code)
        if len(code) < best_size or best_code is None:
            if time.time() < deadline - 25:
                code = local_search(code, deadline - 5)
                code = reduce_code(code)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
        seed += 1

    # final polish with whatever time remains
    if best_code is not None and time.time() < deadline - 1:
        best_code = local_search(best_code, deadline - 1)
        best_code = reduce_code(best_code)

    return [list(idx_to_tuple(c)) for c in best_code]
