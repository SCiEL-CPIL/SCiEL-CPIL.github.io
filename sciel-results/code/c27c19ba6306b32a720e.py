"""Covering design C(v,k,t) via vectorized greedy + deletion + local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _encode(combs, v):
    """Encode t-combinations (array of ints, shape (N,t)) into int64 codes."""
    base = v
    c = np.asarray(combs, dtype=np.int64)
    codes = np.zeros(c.shape[0], dtype=np.int64)
    for j in range(c.shape[1]):
        codes = codes * base + c[:, j]
    return codes


def _membership(codes, lookup):
    """Boolean array: is each code in lookup (a set of int64 codes)?"""
    return np.fromiter((int(c) in lookup for c in codes), dtype=bool, count=codes.shape[0])


def _all_tsubsets(v, t):
    return list(itertools.combinations(range(v), t))


def _precompute_point_sets(v, k, t):
    """point_sets[x] = int64 array of codes of all t-subsets containing x."""
    point_sets = []
    for x in range(v):
        subs = []
        for rest in itertools.combinations([y for y in range(v) if y != x], t - 1):
            subs.append(tuple(sorted((x,) + rest)))
        point_sets.append(_encode(np.array(subs, dtype=np.int64), v))
    return point_sets


def _block_codes(block, v, t):
    return _encode(np.array(list(itertools.combinations(block, t)), dtype=np.int64), v)


def _greedy_build(v, k, t, point_sets, uncovered, deadline):
    """Greedy: open block on first uncovered t-subset, add max-gain points."""
    blocks = []
    while uncovered:
        if time.time() > deadline:
            break
        # first uncovered t-subset (smallest)
        first = min(uncovered)
        block = list(first)
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # 4-subsets newly covered by adding x to block
                cands = []
                for rest in itertools.combinations(block, t - 1):
                    cands.append(tuple(sorted(rest + (x,))))
                codes = _encode(np.array(cands, dtype=np.int64), v)
                gain = int(_membership(codes, uncovered).sum())
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        block.sort()
        blocks.append(tuple(block))
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
    return blocks


def _deletion_pass(blocks, v, t):
    """Remove blocks whose t-subsets are all covered by other blocks."""
    n = len(blocks)
    if n == 0:
        return blocks
    block_code_lists = [_block_codes(list(b), v, t) for b in blocks]
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        # check each t-subset of block i is covered by some other kept block
        covered = np.zeros(len(block_code_lists[i]), dtype=bool)
        for j in range(n):
            if j == i or not keep[j]:
                continue
            covered |= _membership(block_code_lists[i], set(block_code_lists[j]))
        if covered.all():
            keep[i] = False
    return [b for b, kk in zip(blocks, keep) if kk]


def _local_search(blocks, v, k, t, point_sets, deadline, max_rounds=6):
    """Replace one point per block to maximize newly-covered t-subsets."""
    all_subs = _all_tsubsets(v, t)
    covered = np.zeros(len(all_subs), dtype=bool)
    for b in blocks:
        covered |= _membership(_block_codes(list(b), v, t), set(range(len(all_subs))))
    # map code -> index via dict for fast uncovered set
    code_to_idx = {}
    for idx, sub in enumerate(all_subs):
        code_to_idx[int(_encode(np.array([sub], dtype=np.int64), v)[0])] = idx

    def uncovered_set():
        return {idx for idx in range(len(all_subs)) if not covered[idx]}

    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for bi, b in enumerate(blocks):
            if time.time() > deadline:
                break
            bset = set(b)
            best_gain = 0
            best_new = None
            for out in b:
                for inp in range(v):
                    if inp in bset:
                        continue
                    newb = tuple(sorted((bset - {out}) | {inp}))
                    codes = _block_codes(list(newb), v, t)
                    idxs = np.array([code_to_idx[int(c)] for c in codes], dtype=np.int64)
                    gain = int((~covered[idxs]).sum())
                    if gain > best_gain:
                        best_gain = gain
                        best_new = newb
            if best_gain > 0 and best_new is not None:
                old_codes = _block_codes(list(b), v, t)
                old_idx = np.array([code_to_idx[int(c)] for c in old_codes], dtype=np.int64)
                new_codes = _block_codes(list(best_new), v, t)
                new_idx = np.array([code_to_idx[int(c)] for c in new_codes], dtype=np.int64)
                covered[old_idx] = False
                covered[new_idx] = True
                blocks[bi] = best_new
                improved = True
        if not improved:
            break
    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    point_sets = _precompute_point_sets(v, k, t)
    uncovered = set(_all_tsubsets(v, t))

    blocks = _greedy_build(v, k, t, point_sets, uncovered, deadline)

    # ensure full coverage (greedy may stop early on deadline)
    while uncovered and time.time() < deadline:
        first = min(uncovered)
        block = list(first)
        while len(block) < k:
            best_x, best_gain = None, -1
            for x in range(v):
                if x in block:
                    continue
                cands = [tuple(sorted(rest + (x,))) for rest in itertools.combinations(block, t - 1)]
                codes = _encode(np.array(cands, dtype=np.int64), v)
                gain = int(_membership(codes, uncovered).sum())
                if gain > best_gain:
                    best_gain, best_x = gain, x
            block.append(best_x)
        block.sort()
        blocks.append(tuple(block))
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)

    if time.time() < deadline:
        blocks = _deletion_pass(blocks, v, t)

    if time.time() < deadline - 15.0:
        blocks = _local_search(blocks, v, k, t, point_sets, deadline, max_rounds=6)
        if time.time() < deadline:
            blocks = _deletion_pass(blocks, v, t)

    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
