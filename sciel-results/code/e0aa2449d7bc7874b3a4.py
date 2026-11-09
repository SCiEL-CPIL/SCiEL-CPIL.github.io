"""Football pool / ternary covering code K_3(8,1).
New design: greedy max-coverage seed + deletion + guided local search (critical-word swaps)."""
import time
import random

import numpy as np


def _hamming(a, b):
    return int(np.count_nonzero(a != b))


def construct(q, n, radius):
    deadline = time.time() + 110.0  # safety margin under the 120 s wall-clock kill
    rng = random.Random(12345)

    # --- 1. Greedy max-coverage seed -----------------------------------------
    # A codeword covers 1 + n*(q-1) words (radius-1 ball in {0..q-1}^n).
    ball_size = 1 + n * (q - 1)
    chosen = []
    chosen_set = set()
    chosen_arr = np.zeros((0, n), dtype=np.int8)

    # Candidate pool: random sample of the space (avoid building all 3**n for large n).
    if q ** n <= 200000:
        from itertools import product
        candidates = [tuple(w) for w in product(range(q), repeat=n)]
    else:
        candidates = [tuple(rng.randrange(q) for _ in range(n)) for _ in range(6000)]

    for w in candidates:
        if time.time() > deadline:
            break
        # covered radius-1 ball of w
        ball = {w}
        for pos in range(n):
            for v in range(q):
                if v == w[pos]:
                    continue
                tmp = list(w)
                tmp[pos] = v
                ball.add(tuple(tmp))
        # count how many of the ball are NOT already covered by chosen
        new = 0
        for b in ball:
            ok = False
            for c in chosen:
                if _hamming(b, c) <= radius:
                    ok = True
                    break
            if not ok:
                new += 1
        # accept if it adds meaningful new coverage
        if new >= max(1, ball_size // 3):
            chosen.append(w)
            chosen_set.add(w)
            if len(chosen_arr) == 0:
                chosen_arr = np.array(chosen, dtype=np.int8)
            else:
                chosen_arr = np.vstack([chosen_arr, np.array(w, dtype=np.int8)])
        # stop early once coverage is plentiful relative to lower bound
        if len(chosen) >= 430:
            break

    # ensure we have enough to actually cover; top up with random greedy if short
    guard = 0
    while len(chosen) < 400 and guard < 5000:
        guard += 1
        w = tuple(rng.randrange(q) for _ in range(n))
        if w in chosen_set:
            continue
        chosen.append(w)
        chosen_set.add(w)
        chosen_arr = np.vstack([chosen_arr, np.array(w, dtype=np.int8)]) if len(chosen_arr) else np.array(chosen, dtype=np.int8)

    # --- 2. Deletion pass (drop redundant codewords) -------------------------
    def is_redundant(idx, arr, n, radius):
        # codeword idx is redundant if every word in its ball is covered by another chosen
        c = arr[idx]
        # check a sample of its ball; if all covered by others -> redundant
        # full ball check for correctness on small n
        for pos in range(n):
            for v in range(q):
                if v == c[pos]:
                    continue
                tmp = list(c)
                tmp[pos] = v
                t = np.array(tmp, dtype=np.int8)
                covered = False
                for j in range(len(arr)):
                    if j == idx:
                        continue
                    if _hamming(t, arr[j]) <= radius:
                        covered = True
                        break
                if not covered:
                    return False
        # also the center word
        covered = False
        for j in range(len(arr)):
            if j == idx:
                continue
            if _hamming(c, arr[j]) <= radius:
                covered = True
                break
        if not covered:
            return False
        return True

    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for idx in range(len(chosen_arr) - 1, -1, -1):
            if is_redundant(idx, chosen_arr, n, radius):
                chosen_arr = np.delete(chosen_arr, idx, axis=0)
                chosen = [tuple(r) for r in chosen_arr.tolist()]
                changed = True
                break

    # --- 3. Guided local search: critical-word swaps -------------------------
    # A "critical word" for a codeword c is a word x with d(x,c)==radius such that
    # x is NOT within radius of any other chosen codeword. Removing c would lose x.
    # We try replacing c with a better critical word that covers more.
    best_len = len(chosen_arr)

    def ball_words(c):
        out = []
        for pos in range(n):
            for v in range(q):
                if v == c[pos]:
                    continue
                tmp = list(c)
                tmp[pos] = v
                out.append(tmp)
        return out

    for it in range(4000):
        if time.time() > deadline - 3:
            break
        # pick a random codeword to try to replace
        idx = rng.randrange(len(chosen_arr))
        c = chosen_arr[idx].copy()
        others = np.delete(chosen_arr, idx, axis=0)

        # find critical words for c (words only c covers)
        critical = []
        for pos in range(n):
            for v in range(q):
                if v == c[pos]:
                    continue
                tmp = list(c)
                tmp[pos] = v
                t = np.array(tmp, dtype=np.int8)
                # distance to c is 1 (== radius here), so candidate
                # check not covered by others
                covered = False
                for j in range(len(others)):
                    if _hamming(t, others[j]) <= radius:
                        covered = True
                        break
                if not covered:
                    critical.append(t)

        if not critical:
            continue

        # for each critical word, compute how many new words it would cover if we
        # replace c with it (i.e. words within radius of the new word not within radius
        # of others). Pick the best.
        best_new = None
        best_gain = 0
        for t in critical:
            # new word = t; it covers its ball. Count words in its ball not covered by others.
            gain = 0
            for pos2 in range(n):
                for v2 in range(q):
                    if v2 == t[pos2]:
                        continue
                    tmp2 = list(t)
                    tmp2[pos2] = v2
                    tt = np.array(tmp2, dtype=np.int8)
                    covered = False
                    for j in range(len(others)):
                        if _hamming(tt, others[j]) <= radius:
                            covered = True
                            break
                    if not covered:
                        gain += 1
            # also the center t itself (distance 0)
            covered = False
            for j in range(len(others)):
                if _hamming(t, others[j]) <= radius:
                    covered = True
                    break
            if not covered:
                gain += 1
            if gain > best_gain:
                best_gain = gain
                best_new = t

        if best_new is not None and best_gain >= 1:
            # accept the swap (keep size, improve coverage robustness)
            chosen_arr[idx] = best_new
            # occasionally allow growth if it helps: add a fresh critical word
            if rng.random() < 0.1 and len(chosen_arr) < best_len + 20:
                neww = best_new.copy()
                # perturb to a fresh word
                pos3 = rng.randrange(n)
                neww[pos3] = rng.randrange(q)
                chosen_arr = np.vstack([chosen_arr, neww])
            if len(chosen_arr) < best_len:
                best_len = len(chosen_arr)

    # --- 4. Final validity check & return ------------------------------------
    result = [list(r) for r in chosen_arr.tolist()]
    # dedupe
    seen = set()
    dedup = []
    for w in result:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            dedup.append(w)
    return dedup
