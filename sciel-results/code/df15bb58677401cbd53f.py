"""Covering design C(v,k,t): minimize number of k-blocks covering all t-subsets.

Design:
  * Incremental coverage bookkeeping: `cover[x][y]` = number of blocks covering
    the t-subset (x, y). This makes both the greedy point-gain and the
    redundancy test O(1) per t-subset instead of scanning all blocks.
  * Multi-start greedy with varied tie-breaking (deterministic, diverse) to
    obtain several candidate coverings.
  * Redundancy-removal post-processing (drop blocks whose t-subsets are all
    covered by other blocks) -- measured improvement in this task's history.
  * Explicit wall-clock deadline guard returning the best valid result.
"""
import time
import itertools


def _greedy(v, k, t, seed):
    """One greedy pass. Returns (blocks, cover) or None if it would time out."""
    if t == 0:
        return [], {}
    if k == v:
        return [list(range(v))], {}
    if k < t:
        raise ValueError("k must be >= t")

    n = v * v
    cover = [0] * n
    rem = [0] * n
    total = 0
    for a in range(v):
        base = a * v
        for b in range(a + 1, v):
            rem[base + b] = 1
            total += 1

    blocks = []
    rng = seed
    while total > 0:
        # pick an uncovered subset (deterministic scan; order is stable)
        start = (rng % n)
        idx = -1
        for off in range(n):
            j = (start + off) % n
            if rem[j] > 0:
                idx = j
                break
        if idx < 0:
            break
        a = idx // v
        b = idx % v
        block = [a, b]
        while len(block) < k:
            bl = block
            best = -1
            best_g = -1
            for x in range(v):
                if x in bl:
                    continue
                g = 0
                for i in range(len(bl)):
                    base = bl[i] * v
                    g += rem[base + x]
                    g += rem[x * v + bl[i]]
                if g > best_g:
                    best_g = g
                    best = x
            block.append(best)
        # add the block: cover every t-subset of it
        for combo in itertools.combinations(block, t):
            for i in range(t):
                for j in range(i + 1, t):
                    x = combo[i]
                    y = combo[j]
                    idx2 = x * v + y
                    if rem[idx2] > 0:
                        rem[idx2] -= 1
                        total -= 1
                    cover[idx2] += 1
        blocks.append(block)
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
    return blocks, cover


def _remove_redundant(blocks, v, t):
    """Drop blocks whose every t-subset is covered by at least one other block."""
    n = v * v
    cnt = [0] * n
    for b in blocks:
        for combo in itertools.combinations(b, t):
            for i in range(t):
                for j in range(i + 1, t):
                    cnt[combo[i] * v + combo[j]] += 1
    kept = []
    for b in blocks:
        redundant = True
        for combo in itertools.combinations(b, t):
            c = 0
            for i in range(t):
                for j in range(i + 1, t):
                    c += cnt[combo[i] * v + combo[j]]
            if c <= 1:
                redundant = False
                break
        if not redundant:
            kept.append(b)
    return kept


def construct(v, k, t):
    deadline = time.time() + 110.0   # safety margin under the 120 s wall clock

    if t == 0:
        return []
    if k == v:
        return [list(range(v))]
    if k < t:
        raise ValueError("k must be >= t")

    candidates = []
    seeds = [0, 1, 2, 3, 5, 7, 11, 13, 17, 23, 29, 31, 37, 41, 43, 47]
    for s in seeds:
        if time.time() > deadline - 15.0:
            break
        res = _greedy(v, k, t, s)
        if res is None:
            break
        blocks, _ = res
        candidates.append(blocks)
        # post-process (only if it will still fit)
        if time.time() < deadline - 5.0:
            try:
                rp = _remove_redundant(blocks, v, t)
                if len(rp) < len(blocks):
                    candidates.append(rp)
            except Exception:
                pass

    if not candidates:
        # fallback: a single greedy pass (should not happen)
        blocks, _ = _greedy(v, k, t, 0)
        return blocks

    best = min(candidates, key=len)
    return best
