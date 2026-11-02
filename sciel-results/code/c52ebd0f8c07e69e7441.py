# EVOLVE-BLOCK-START
import time
import random
import itertools

import numpy as np

_DEADLINE = 150.0
_MARGIN = 15.0


def _prep(v, t):
    tsubs = list(itertools.combinations(range(v), t))
    tsubs_arr = np.array(tsubs, dtype=np.int64)
    return tsubs, tsubs_arr, len(tsubs)


def _covers_mask(block, t, tsubs_arr):
    """Boolean mask: which t-subsets are contained in `block`."""
    B = sorted(block)
    mask = np.zeros(len(tsubs_arr), dtype=bool)
    for comb in itertools.combinations(range(len(B)), t):
        vals = np.sort(np.asarray(B[list(comb)], dtype=np.int64))
        mask |= (tsubs_arr == vals).all(axis=1)
    return mask


def _greedy(v, k, t, tsubs_arr, rng):
    N = len(tsubs_arr)
    uncovered = np.ones(N, dtype=bool)
    blocks = []
    while uncovered.any():
        first = int(np.argmax(uncovered))
        block = list(tsubs_arr[first])
        bset = set(block)
        while len(block) < k:
            cands = [x for x in range(v) if x not in bset]
            if not cands:
                break
            best_x, best_gain = cands[0], -1
            for x in cands:
                nb = block + [x]
                gain = int(_covers_mask(nb, t, tsubs_arr)[uncovered].sum())
                if gain > best_gain:
                    best_gain, best_x = gain, x
                    if gain == N:
                        break
            block.append(best_x)
            bset.add(best_x)
        block.sort()
        blocks.append(block)
        uncovered[_covers_mask(block, t, tsubs_arr)] = False
    return blocks


def _valid(blocks, t, tsubs_arr):
    cov = np.zeros(len(tsubs_arr), dtype=bool)
    for b in blocks:
        cov |= _covers_mask(b, t, tsubs_arr)
    return cov.all()


def _remove_redundant(blocks, t, tsubs_arr, deadline):
    N = len(tsubs_arr)
    covered = np.zeros(N, dtype=bool)
    for b in blocks:
        covered |= _covers_mask(b, t, tsubs_arr)
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            b = blocks[i]
            if (covered & ~_covers_mask(b, t, tsubs_arr)).all():
                blocks.pop(i)
                changed = True
                covered = np.zeros(N, dtype=bool)
                for bb in blocks:
                    covered |= _covers_mask(bb, t, tsubs_arr)
                break
    return blocks


def _local_search(blocks, v, k, t, tsubs_arr, deadline):
    """Drop a block, re-greedy the gaps; keep if it shrinks the set."""
    N = len(tsubs_arr)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        count = np.zeros(N, dtype=int)
        for b in blocks:
            count += _covers_mask(b, t, tsubs_arr).astype(int)
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            b = blocks[i]
            # gaps if we remove b
            gaps = count == 0
            gap_mask = (count == 1) & _covers_mask(b, t, tsubs_arr)
            if not gap_mask.any():
                continue  # fully redundant, handled elsewhere
            # try: remove b, then greedily cover only the gap t-subsets
            trial = blocks[:i] + blocks[i+1:]
            # build a covering of just gap_mask
            unc = gap_mask.copy()
            add = 0
            while unc.any() and add < 3:
                first = int(np.argmax(unc))
                nb = list(tsubs_arr[first])
                bset = set(nb)
                while len(nb) < k:
                    cands = [x for x in range(v) if x not in bset]
                    if not cands:
                        break
                    bx, bg = cands[0], -1
                    for x in cands:
                        g = int(_covers_mask(nb + [x], t, tsubs_arr)[unc].sum())
                        if g > bg:
                            bg, bx = g, x
                            if g == int(unc.sum()):
                                break
                    nb.append(bx)
                    bset.add(bx)
                nb.sort()
                trial.append(nb)
                unc[_covers_mask(nb, t, tsubs_arr)] = False
                add += 1
            if not unc.any() and len(trial) < len(blocks):
                blocks = trial
                improved = True
                break
    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + (_DEADLINE - _MARGIN)
    tsubs, tsubs_arr, N = _prep(v, t)

    best, best_len = None, float('inf')

    def consider(blocks):
        nonlocal best, best_len
        if _valid(blocks, t, tsubs_arr):
            blocks = _remove_redundant(blocks, t, tsubs_arr, deadline)
            blocks = _local_search(blocks, v, k, t, tsubs_arr, deadline)
            if len(blocks) < best_len:
                best, best_len = blocks, len(blocks)

    consider(_greedy(v, k, t, tsubs_arr, random.Random(0)))
    seed = 1
    while time.time() < deadline:
        consider(_greedy(v, k, t, tsubs_arr, random.Random(seed)))
        seed += 1

    if best is None:
        best = _greedy(v, k, t, tsubs_arr, random.Random(0))
    return best
# EVOLVE-BLOCK-END
