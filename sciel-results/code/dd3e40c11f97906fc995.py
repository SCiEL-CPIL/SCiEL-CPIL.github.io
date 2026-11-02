"""Structural redesign: greedy set-cover + local-search removal for K_q(n, radius)."""
import time
import numpy as np


def _decode(idx, q, n):
    out = [0] * n
    for i in range(n - 1, -1, -1):
        out[i] = idx % q
        idx //= q
    return out


def _encode(digits, q, n):
    idx = 0
    for i in range(n):
        idx = idx * q + digits[i]
    return idx


def construct(q, n, radius):
    deadline = time.time() + 110.0
    total = q ** n

    # Precompute covered sets: for each center, the array of word indices it covers.
    covered_sets = []
    for c in range(total):
        digits = _decode(c, q, n)
        idxs = [c]
        if radius >= 1:
            for p in range(n):
                orig = digits[p]
                for v in range(q):
                    if v == orig:
                        continue
                    d2 = digits[:]
                    d2[p] = v
                    idxs.append(_encode(d2, q, n))
        if radius >= 2:
            for p1 in range(n):
                for p2 in range(p1 + 1, n):
                    for v1 in range(q):
                        if v1 == digits[p1]:
                            continue
                        for v2 in range(q):
                            if v2 == digits[p2]:
                                continue
                            d2 = digits[:]
                            d2[p1] = v1
                            d2[p2] = v2
                            idxs.append(_encode(d2, q, n))
        covered_sets.append(np.array(idxs, dtype=np.int64))

    # Greedy set cover
    cover_count = np.zeros(total, dtype=np.int32)
    uncovered = np.ones(total, dtype=bool)
    remaining = total
    chosen = []
    while remaining > 0 and time.time() < deadline:
        best_c = -1
        best_gain = 0
        for c in range(total):
            gain = uncovered[covered_sets[c]].sum()
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_gain == 0:
            break
        chosen.append(best_c)
        cs = covered_sets[best_c]
        mask = uncovered[cs]
        remaining -= mask.sum()
        uncovered[cs] = False
        cover_count[cs] += 1

    best = list(set(chosen))

    # Local search: repeated removal of redundant centers
    def compute_cover_count(center_list):
        cc = np.zeros(total, dtype=np.int32)
        for c in center_list:
            cc[covered_sets[c]] += 1
        return cc

    improved = True
    while improved and time.time() < deadline:
        improved = False
        cc = compute_cover_count(best)
        for i in range(len(best)):
            if time.time() > deadline:
                break
            cs = covered_sets[best[i]]
            cc[cs] -= 1
            if (cc > 0).all():
                best = best[:i] + best[i+1:]
                improved = True
                break
            else:
                cc[cs] += 1

    result = [_decode(c, q, n) for c in best]
    return result
