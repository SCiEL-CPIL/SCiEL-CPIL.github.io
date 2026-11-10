# EVOLVE-BLOCK-START
import time
import numpy as np

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin vs 150s kill
    if q == 3 and n == 7 and radius == 1:
        return _ternary7_r1(deadline)
    return _generic_greedy(q, n, radius, deadline)


def _ternary7_r1(deadline):
    n = 7
    q = 3
    total = q ** n  # 2187
    ball_size = 1 + n * (q - 1)  # 15

    # All words as base-3 digit arrays
    all_words = np.array(np.unravel_index(np.arange(total), [3] * n)).T  # (2187,7)

    # Precompute ball index lists (each 15)
    ball_lists = []
    for c in range(total):
        w = all_words[c]
        ball = [c]
        for i in range(n):
            orig = w[i]
            for nv in range(q):
                if nv != orig:
                    w2 = w.copy()
                    w2[i] = nv
                    ball.append(int(np.dot(w2, [3 ** (n - 1 - j) for j in range(n)])))
        ball_lists.append(ball)
    ball_idx = np.array(ball_lists, dtype=np.int32)  # (2187,15)

    # ---------- Greedy set cover ----------
    cov = np.zeros(total, dtype=np.int32)
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    chosen_set = set()

    while uncovered.any() and time.time() < deadline - 30.0:
        # marginal gain = number of currently-uncovered words in each ball
        gains = uncovered[ball_idx].sum(axis=1)
        for c in chosen_set:
            gains[c] = -1
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        chosen.append(best)
        chosen_set.add(best)
        b = ball_idx[best]
        cov[b] += 1
        uncovered[b] = False

    if not uncovered.any():
        chosen = _local_search(chosen, ball_idx, total, deadline)
        return [all_words[c].tolist() for c in chosen]

    # Fallback fill (should not happen)
    while uncovered.any() and time.time() < deadline - 30.0:
        for c in range(total):
            if c not in chosen_set:
                chosen.append(c); chosen_set.add(c)
                b = ball_idx[c]; cov[b] += 1; uncovered[b] = False
                break
    chosen = _local_search(chosen, ball_idx, total, deadline)
    return [all_words[c].tolist() for c in chosen]


def _local_search(chosen, ball_idx, total, deadline):
    # cov[w] = number of chosen codewords whose ball contains word w
    cov = np.zeros(total, dtype=np.int32)
    for c in chosen:
        cov[ball_idx[c]] += 1
    chosen_set = set(chosen)
    n = 7
    q = 3

    def gain_of(c):
        return int((cov[ball_idx[c]] == 0).sum())

    improved = True
    while improved and time.time() < deadline - 4.0:
        improved = False
        m = len(chosen)
        # ---- 2-opt pass: remove one, insert another ----
        for i in range(m):
            if time.time() > deadline - 4.0:
                break
            c = chosen[i]
            # temporarily remove c
            bc = ball_idx[c]
            cov[bc] -= 1
            # uncovered words = those with cov == 0
            # find best insertion c2
            gains = (cov[ball_idx] == 0).sum(axis=1)
            for x in chosen:
                if x != c:
                    gains[x] = -1
            c2 = int(np.argmax(gains))
            if c2 != c and gains[c2] >= 1:
                # accept if it fixes all uncovered (keeps valid) and reduces or ties
                # after removing c, uncovered count:
                unc = int((cov == 0).sum())
                if gains[c2] >= unc:  # covers all currently uncovered -> still valid
                    # only accept if it strictly reduces total? m stays same.
                    # We want to reduce future redundancy; accept if gain strictly improves
                    # the "slack". Use: accept if it reduces the number of words with cov==1
                    # (more robust coverage) OR simply accept any valid move to escape local min.
                    # To guarantee progress & termination, accept only if it reduces
                    # a monotone potential: total over words of (cov==1) count.
                    before_slack = int((cov == 1).sum())
                    bc2 = ball_idx[c2]
                    cov[bc2] += 1
                    after_slack = int((cov == 1).sum())
                    if after_slack < before_slack:
                        chosen[i] = c2
                        chosen_set.discard(c)
                        chosen_set.add(c2)
                        improved = True
                    else:
                        cov[bc2] -= 1  # revert
            # restore c
            cov[bc] += 1

        # ---- pure removal pass (redundant codewords) ----
        i = 0
        while i < len(chosen):
            if time.time() > deadline - 4.0:
                break
            c = chosen[i]
            bc = ball_idx[c]
            if (cov[bc] > 1).all():
                cov[bc] -= 1
                chosen.pop(i)
                chosen_set.discard(c)
                improved = True
            else:
                i += 1
    return chosen


def _generic_greedy(q, n, radius, deadline):
    total = q ** n
    all_words = np.array(np.unravel_index(np.arange(total), [q] * n)).T
    from itertools import combinations, product
    powq = [q ** (n - 1 - j) for j in range(n)]

    def ball_of(c):
        w = all_words[c]
        ball = [c]
        for r in range(1, radius + 1):
            for positions in combinations(range(n), r):
                for values in product(range(q), repeat=r):
                    w2 = w.copy()
                    for pos, val in zip(positions, values):
                        if val != w2[pos]:
                            w2[pos] = val
                    ball.append(int(np.dot(w2, powq)))
        return ball

    ball_lists = [ball_of(c) for c in range(total)]
    ball_idx = np.array(ball_lists, dtype=np.int32)
    cov = np.zeros(total, dtype=np.int32)
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    chosen_set = set()
    while uncovered.any() and time.time() < deadline - 10.0:
        gains = uncovered[ball_idx].sum(axis=1)
        for c in chosen_set:
            gains[c] = -1
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        chosen.append(best); chosen_set.add(best)
        b = ball_idx[best]; cov[b] += 1; uncovered[b] = False
    if time.time() < deadline - 4.0:
        chosen = _local_search(chosen, ball_idx, total, deadline)
    return [all_words[c].tolist() for c in chosen]
# EVOLVE-BLOCK-END
