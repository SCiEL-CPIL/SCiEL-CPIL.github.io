"""Structural rewrite: exact coverage-count local search for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _precompute_neighbors(q, n, radius):
    """Return (neighbor_index_array, neighbor_count_array).
    neighbor_index_array[i, k] = index of the k-th neighbor of word i (within radius),
    with -1 padding for rows that have fewer than max_n neighbors."""
    total = q ** n
    digits = np.zeros((total, n), dtype=np.int32)
    for pos in range(n - 1, -1, -1):
        digits[:, pos] = np.mod(np.floor(np.arange(total) / (q ** pos)), q)

    max_n = 1 + n * (q - 1) if radius >= 1 else 1
    idx_arr = np.full((total, max_n), -1, dtype=np.int64)
    cnt_arr = np.zeros(total, dtype=np.int32)

    if radius >= 1:
        # For each position and each alternative value, the set of words differing
        # only at that position. Vectorize over all words at once.
        for pos in range(n):
            base = q ** pos
            for v in range(q):
                dv = digits[:, pos]
                mask = dv != v
                if not mask.any():
                    continue
                # new index = idx + (v - dv) * base
                new_idx = np.arange(total)[mask] + (v - dv[mask]) * base
                k = cnt_arr[mask]
                idx_arr[mask, k] = new_idx
                cnt_arr[mask] = k + 1
    return idx_arr, cnt_arr


def _count_from_code(code, idx_arr, cnt_arr, total):
    """coverage-count array: cov[j] = number of codewords covering word j."""
    cov = np.zeros(total, dtype=np.int32)
    for c in code:
        cnt = cnt_arr[c]
        if cnt > 0:
            cov[idx_arr[c, :cnt]] += 1
    return cov


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 132  # safety margin under 150s

    total = q ** n

    # Precompute neighbor structure (vectorized)
    idx_arr, cnt_arr = _precompute_neighbors(q, n, radius)

    # ---- Build a good initial seed ----
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        seed_words = [h + list(t) for h in tetracode
                      for t in _itertools_product(range(3), repeat=n - 4)]
    else:
        seed_words = [list(range(n))]

    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    code = list(dict.fromkeys(word_to_idx(w) for w in seed_words))

    cov = _count_from_code(code, idx_arr, cnt_arr, total)

    def is_full(c):
        return np.all(c > 0)

    if not is_full(cov):
        # Greedy fill: add codeword covering most uncovered words
        while not is_full(cov) and time.time() < deadline:
            uncovered = cov == 0
            gains = np.zeros(total, dtype=np.int32)
            for c in code:
                gains[c] = -1
            for j in range(total):
                if not uncovered[j]:
                    continue
                cnt = cnt_arr[j]
                if cnt > 0:
                    gains[idx_arr[j, :cnt]] += 1
            gmax = gains.max()
            if gmax <= 0:
                break
            best = int(np.argmax(gains))
            code.append(best)
            cnt = cnt_arr[best]
            if cnt > 0:
                cov[idx_arr[best, :cnt]] += 1

    # ---- Local search: exact redundancy removal + exact swap search ----
    def rebuild():
        return _count_from_code(code, idx_arr, cnt_arr, total)

    def coverage_of(c):
        cnt = cnt_arr[c]
        return idx_arr[c, :cnt] if cnt > 0 else np.array([], dtype=np.int64)

    # Phase A: repeatedly remove redundant codewords (exact)
    def remove_redundant():
        changed = True
        while changed and time.time() < deadline:
            changed = False
            cov = _count_from_code(code, idx_arr, cnt_arr, total)
            m = len(code)
            for i in range(m - 1, -1, -1):
                if time.time() > deadline:
                    return
                c = code[i]
                cov2 = cov.copy()
                cnt = cnt_arr[c]
                if cnt > 0:
                    cov2[idx_arr[c, :cnt]] -= 1
                if np.all(cov2 > 0):
                    code.pop(i)
                    changed = True
                    break

    # Phase B: swap search — replace one codeword with a better one (exact)
    def swap_search(max_iters=4000):
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            cov = _count_from_code(code, idx_arr, cnt_arr, total)
            m = len(code)
            best_gain = 0
            best_r = -1
            best_a = -1

            # sample candidate removals
            sample_size = min(m, 6)
            for ri in random.sample(range(m), sample_size):
                if time.time() > deadline:
                    break
                c = code[ri]
                cov2 = cov.copy()
                cnt = cnt_arr[c]
                if cnt > 0:
                    cov2[idx_arr[c, :cnt]] -= 1
                lost = cov2 == 0
                if not lost.any():
                    best_gain = 1
                    best_r = ri
                    best_a = -2  # pure removal
                    break

                # candidates: words that cover the lost set
                lost_idx = np.where(lost)[0]
                if len(lost_idx) > 25:
                    cand = random.sample(lost_idx.tolist(), 25)
                else:
                    cand = lost_idx.tolist()

                code_set = set(code)
                for a in cand:
                    if a in code_set:
                        continue
                    acov = coverage_of(a)
                    if len(acov) == 0:
                        continue
                    if not np.all(cov2[acov] > 0):
                        continue
                    # a covers all lost words -> valid swap; gain = new words covered
                    new_gain = int(np.sum(cov[acov] == 0))
                    if new_gain > best_gain:
                        best_gain = new_gain
                        best_r = ri
                        best_a = a

            if best_r >= 0:
                if best_a == -2:
                    code.pop(best_r)
                else:
                    code[best_r] = best_a
            else:
                break

    # Alternate removal and swap phases until no improvement / deadline
    prev_len = -1
    while time.time() < deadline:
        remove_redundant()
        if len(code) < prev_len:
            prev_len = len(code)
            continue
        swap_search(max_iters=3000)
        if len(code) == prev_len:
            # try one more removal pass after swaps
            remove_redundant()
            if len(code) == prev_len:
                break
            prev_len = len(code)

    # Final safety: ensure full coverage
    cov = _count_from_code(code, idx_arr, cnt_arr, total)
    if not np.all(cov > 0):
        # greedy top-up
        while not np.all(cov > 0) and time.time() < deadline:
            uncovered = cov == 0
            gains = np.zeros(total, dtype=np.int32)
            code_set = set(code)
            for j in range(total):
                if not uncovered[j]:
                    continue
                cnt = cnt_arr[j]
                if cnt > 0:
                    gains[idx_arr[j, :cnt]] += 1
            for c in code_set:
                gains[c] = -1
            gmax = gains.max()
            if gmax <= 0:
                break
            best = int(np.argmax(gains))
            code.append(best)
            cnt = cnt_arr[best]
            if cnt > 0:
                cov[idx_arr[best, :cnt]] += 1

    # Convert to words
    def idx_to_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return w[::-1]

    seen = set()
    result = []
    for c in code:
        w = idx_to_word(c)
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            result.append(w)
    return result


def _itertools_product(*args, repeat=1):
    """Minimal product to avoid importing itertools (kept for clarity)."""
    pools = [tuple(p) for p in args]
    if repeat:
        pools = pools * repeat
    if not pools:
        yield ()
        return
    result = [[]]
    for pool in pools:
        result = [r + [x] for r in result for x in pool]
    for r in result:
        yield tuple(r)
# EVOLVE-BLOCK-END
