"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 140.0  # safety margin before the 150s kill

    # ---- small-n fallback (exhaustive greedy) ----
    if n <= 4:
        all_words = list(itertools.product(range(q), repeat=n))
        uncovered = set(all_words)
        code = []
        for w in all_words:
            if not uncovered:
                break
            if w in uncovered:
                code.append(list(w))
                for u in list(uncovered):
                    if sum(1 for i in range(n) if u[i] != w[i]) <= radius:
                        uncovered.remove(u)
        return code

    # ---- mapping helpers ----
    def word_to_int(w):
        val = 0
        for x in w:
            val = val * q + x
        return val

    def int_to_word(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % q
            val //= q
        return w

    total = q ** n

    # ---- precompute radius balls for every word ----
    neighbor_lists = []
    for i in range(total):
        w = int_to_word(i)
        ball = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    w[pos] = new_val
                    ball.append(word_to_int(w))
            w[pos] = orig
        neighbor_lists.append(ball)

    # ---- initial code: direct sum of the perfect ternary tetracode ----
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base_tails = list(itertools.product(range(3), repeat=n - 4))
    code = [h + list(t) for h in tetracode for t in base_tails]
    code_set = set(map(tuple, code))

    counts = np.zeros(total, dtype=np.int32)
    for c in code:
        ci = word_to_int(c)
        for nb in neighbor_lists[ci]:
            counts[nb] += 1

    # ---- greedy removal ----
    while time.time() < deadline - 5.0:
        found = False
        for idx in range(len(code)):
            ci = word_to_int(code[idx])
            ball = neighbor_lists[ci]
            if np.all(counts[ball] >= 2):
                for nb in ball:
                    counts[nb] -= 1
                code.pop(idx)
                found = True
                break
        if not found:
            break

    # ---- swap local search (escape the removal local optimum) ----
    def ball_mask(ci):
        m = np.zeros(total, dtype=bool)
        m[neighbor_lists[ci]] = True
        return m

    rng = random.Random(12345)

    def try_swaps():
        nonlocal counts
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            order = list(range(len(code)))
            rng.shuffle(order)
            for i in order:
                if time.time() >= deadline - 5.0:
                    return
                ci = word_to_int(code[i])
                mask_i = ball_mask(ci)
                for j in range(len(code)):
                    if j == i:
                        continue
                    cj = word_to_int(code[j])
                    # combined ball of both words
                    union = mask_i | ball_mask(cj)
                    # after swap: counts[union] - 1 (losing both) + 1 (gaining both)
                    # net change on union: 0; but coverage of words only in i or only in j
                    # must stay >= 1. Compute new counts on union.
                    newc = counts.copy()
                    newc[union] = newc[union] - 1
                    if np.any(newc[union] < 1):
                        continue
                    # accept only if it reduces some "slack" heuristic: accept if valid
                    # and prefer swaps that keep min count high; simple: accept any valid swap
                    # that is not identity (i != j always) -- but that could loop.
                    # To make progress, accept a swap only if it lowers the total
                    # sum of (counts - 1) over the union (reduces redundancy).
                    old_slack = int(np.sum(np.maximum(counts[union] - 1, 0)))
                    new_slack = int(np.sum(np.maximum(newc[union] - 1, 0)))
                    if new_slack < old_slack:
                        counts = newc
                        code[i], code[j] = code[j], code[i]
                        improved = True
                        break
                if improved:
                    break

    try_swaps()

    # ---- final greedy removal pass after swaps ----
    while time.time() < deadline - 5.0:
        found = False
        for idx in range(len(code)):
            ci = word_to_int(code[idx])
            ball = neighbor_lists[ci]
            if np.all(counts[ball] >= 2):
                for nb in ball:
                    counts[nb] -= 1
                    code.pop(idx)
                    found = True
                    break
                break
        if not found:
            break

    return code
# EVOLVE-BLOCK-END
