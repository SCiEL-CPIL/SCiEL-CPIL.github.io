"""Covering design C(v,k,t): minimize number of k-blocks covering all t-subsets."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _to_bitset(combs, v, t):
    """Map each t-subset (row) to a bit index; rows not present get -1."""
    rows = np.asarray(combs, dtype=np.int64)
    if rows.size == 0:
        return np.zeros((0, t), dtype=np.int64), -1
    order = np.argsort(rows, axis=1, kind="stable")
    sorted_rows = rows[np.arange(rows.shape[0])[:, None], order]
    uniq, inv = np.unique(sorted_rows, axis=0, return_inverse=True)
    lut = np.full(v, -1, dtype=np.int64)
    lut[uniq] = np.arange(uniq.shape[0], dtype=np.int64)
    return sorted_rows, lut


def _greedy(v, k, t, deadline):
    """Fast vectorized greedy: open block on least-covered t-subset, add the point
    maximizing newly-covered t-subsets, until the block has k points."""
    if t == 0:
        return []
    if t >= k:
        # every block covers exactly its own (k-subset); need all of them
        return [list(c) for c in itertools.combinations(range(v), k)]

    combs = np.fromiter(
        (x for c in itertools.combinations(range(v), t) for x in c),
        dtype=np.int64, count=v * (v - 1) * (v - 2) // 6 * t)
    sorted_rows, lut = _to_bitset(combs, v, t)
    n = sorted_rows.shape[0]
    if n == 0:
        return []

    counts = np.zeros(n, dtype=np.int64)
    blocks = []
    t1 = max(t - 1, 0)
    while True:
        if time.time() > deadline:
            break
        uncovered = counts == 0
        if not uncovered.any():
            break
        # least-covered t-subset (tie -> first)
        idx = int(np.argmax(uncovered))
        block = [int(x) for x in sorted_rows[idx]]
        while len(block) < k:
            inb = set(block)
            cands = [x for x in range(v) if x not in inb]
            best, bestg = None, -1
            for x in cands:
                if t1 == 0:
                    g = n
                else:
                    tmp = [int(y) for y in sorted_rows[idx]]
                    tmp[t1 - 1] = x
                    tmp.sort()
                    m = np.searchsorted(sorted_rows[:, t1 - 1], tmp[t1 - 1])
                    g = 0
                    while m < n:
                        if np.array_equal(sorted_rows[m], tmp):
                            g += int(counts[m] == 0)
                            m += 1
                        else:
                            break
                if g > bestg:
                    bestg, best = g, x
            block.append(best)
        b = sorted(block)
        blocks.append(b)
        # update counts for all t-subsets inside b
        for c in itertools.combinations(b, t):
            key = list(c)
            key.sort()
            m = np.searchsorted(sorted_rows[:, t - 1], key[t - 1])
            while m < n:
                if np.array_equal(sorted_rows[m], key):
                    counts[m] += 1
                    m += 1
                else:
                    break
    return blocks


def _is_valid(v, k, t, blocks):
    if t == 0:
        return True
    if t >= k:
        return len(blocks) == v * (v - 1) * (v - 2) // 6
    combs = np.fromiter(
        (x for c in itertools.combinations(range(v), t) for x in c),
        dtype=np.int64, count=v * (v - 1) * (v - 2) // 6 * t)
    sorted_rows, lut = _to_bitset(combs, v, t)
    n = sorted_rows.shape[0]
    if n == 0:
        return True
    present = np.zeros(n, dtype=bool)
    for b in blocks:
        for c in itertools.combinations(b, t):
            key = list(c)
            key.sort()
            m = np.searchsorted(sorted_rows[:, t - 1], key[t - 1])
            while m < n:
                if np.array_equal(sorted_rows[m], key):
                    present[m] = True
                    m += 1
                else:
                    break
    return bool(present.all())


def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0  # safety margin before the 120s kill

    # 1) fast greedy to get a valid covering
    blocks = _greedy(v, k, t, deadline)
    if not _is_valid(v, k, t, blocks):
        # fall back to exhaustive t-subsets (guaranteed valid) if greedy failed
        return [list(c) for c in itertools.combinations(range(v), k)][: (v * (v - 1) * (v - 2) // 6)]

    # 2) redundancy reduction: drop any block whose t-subsets are all covered by others
    def _all_covered_by(blocks, skip):
        # counts of each t-subset across all blocks except index `skip`
        combs = np.fromiter(
            (x for c in itertools.combinations(range(v), t) for x in c),
            dtype=np.int64, count=v * (v - 1) * (v - 2) // 6 * t)
        sorted_rows, lut = _to_bitset(combs, v, t)
        n = sorted_rows.shape[0]
        if n == 0:
            return True
        cnt = np.zeros(n, dtype=np.int64)
        for i, b in enumerate(blocks):
            if i == skip:
                continue
            for c in itertools.combinations(b, t):
                key = list(c); key.sort()
                m = np.searchsorted(sorted_rows[:, t - 1], key[t - 1])
                while m < n:
                    if np.array_equal(sorted_rows[m], key):
                        cnt[m] += 1
                        m += 1
                    else:
                        break
        return bool((cnt > 0).all())

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            if _all_covered_by(blocks, i):
                del blocks[i]
                changed = True
                break

    # 3) swap improvement: replace a block with one covering more "rare" t-subsets
    def _coverage_of(b):
        s = set()
        for c in itertools.combinations(b, t):
            s.add(tuple(sorted(c)))
        return s

    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        # find the block whose removal leaves the most uncovered (the "weak" block)
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            # t-subsets covered only by block i
            others = [j for j in range(len(blocks)) if j != i]
            covered_by_others = set()
            for j in others:
                covered_by_others |= _coverage_of(blocks[j])
            only_i = _coverage_of(blocks[i]) - covered_by_others
            if len(only_i) <= 1:
                continue
            # try to find a better block covering more of only_i
            best_b, best_gain = None, len(only_i)
            # candidate: greedy on the only_i subsets
            cand = []
            unc = set(only_i)
            if unc:
                cand = list(min(unc))
                while len(cand) < k and time.time() < deadline:
                    bestx, bestg = None, -1
                    for x in range(v):
                        if x in cand:
                            continue
                        g = 0
                        for rest in itertools.combinations(cand, t - 1):
                            if tuple(sorted(rest + (x,))) in unc:
                                g += 1
                        if g > bestg:
                            bestg, bestx = g, x
                    if bestx is None:
                        break
                    cand.append(bestx)
                    if bestg == 0:
                        break
                cand.sort()
                cov = _coverage_of(cand)
                gain = len(only_i & cov)
                if gain > best_gain:
                    best_gain, best_b = gain, cand
            if best_b is not None:
                blocks[i] = best_b
                improved = True
                break
        if not improved:
            break

    # 4) final repair: ensure validity; if broken, re-greedy from scratch
    if not _is_valid(v, k, t, blocks):
        blocks = _greedy(v, k, t, deadline)
        if not _is_valid(v, k, t, blocks):
            return [list(c) for c in itertools.combinations(range(v), k)][: (v * (v - 1) * (v - 2) // 6)]

    return blocks
# EVOLVE-BLOCK-END
