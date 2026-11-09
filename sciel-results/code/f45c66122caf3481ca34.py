"""Bitmask-accelerated greedy covering with randomized restarts + redundancy polish."""
import time
import random
import itertools

import numpy as np

# Hard wall-clock limit (safety margin under the 150s kill).
_LIMIT = 140.0


def _build_masks(v, t):
    """Return (masks, n_t) where masks[i] is the bitmask of all t-subsets containing point i."""
    n_t = 0
    masks = [0] * v
    for comb in itertools.combinations(range(v), t):
        m = 1 << n_t
        n_t += 1
        for x in comb:
            masks[x] |= m
    return masks, n_t


def _greedy(v, k, t, masks, n_t, rng, use_random, max_blocks):
    """Greedy set-cover. Returns list of blocks (sorted tuples) or None if it hit max_blocks."""
    uncovered = (1 << n_t) - 1
    blocks = []
    while uncovered:
        if time.time() > _LIMIT - 2.0:
            return None
        if len(blocks) >= max_blocks:
            return None
        # Choose a seed point.
        if use_random:
            seed = rng.randrange(v)
        else:
            seed = 0
        block = [seed]
        while len(block) < k:
            best = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = (uncovered & masks[x]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best = x
            block.append(best)
        block.sort()
        blocks.append(tuple(block))
        # Remove newly covered t-subsets.
        m = 0
        for x in block:
            m |= masks[x]
        uncovered &= ~m
    return blocks


def _redundancy_remove(blocks, v, k, t, masks, n_t, rng):
    """Drop blocks that are fully redundant (their t-subsets are covered by other blocks)."""
    if not blocks:
        return blocks
    n = len(blocks)
    # Coverage count per t-subset.
    counts = np.zeros(n_t, dtype=np.int32)
    for b in blocks:
        m = 0
        for x in b:
            m |= masks[x]
        counts |= 0  # no-op; keep vector
        counts += 0
    # Simpler: use bitsets per block.
    bmasks = []
    for b in blocks:
        m = 0
        for x in b:
            m |= masks[x]
        bmasks.append(m)
    # count how many blocks cover each t-subset
    # accumulate via integer bit_count per subset is expensive; do per block OR into counts
    for m in bmasks:
        # add 1 to every bit set in m
        y = m
        while y:
            lb = y & (-y)
            idx = (lb.bit_length() - 1)
            counts[idx] += 1
            y ^= lb
    keep = list(range(n))
    changed = True
    while changed:
        changed = False
        for i in range(len(keep)):
            bi = keep[i]
            m = bmasks[bi]
            # is every t-subset in m covered by at least one other kept block?
            # check counts for bits in m
            y = m
            ok = True
            while y:
                lb = y & (-y)
                idx = (lb.bit_length() - 1)
                if counts[idx] <= 1:
                    ok = False
                    break
                y ^= lb
            if ok:
                # remove block bi
                y = m
                while y:
                    lb = y & (-y)
                    idx = (lb.bit_length() - 1)
                    counts[idx] -= 1
                    y ^= lb
                keep.pop(i)
                changed = True
                break
    return [blocks[i] for i in keep]


def construct(v, k, t):
    t0 = time.time()
    rng = random.Random(12345)
    masks, n_t = _build_masks(v, t)
    max_blocks = 10 ** 6

    best = None
    # Deterministic first pass.
    res = _greedy(v, k, t, masks, n_t, rng, use_random=False, max_blocks=max_blocks)
    if res is not None:
        res = _redundancy_remove(res, v, k, t, masks, n_t, rng)
        best = res

    # Randomized restarts until deadline.
    while time.time() < _LIMIT - 3.0:
        res = _greedy(v, k, t, masks, n_t, rng, use_random=True, max_blocks=max_blocks)
        if res is None:
            continue
        res = _redundancy_remove(res, v, k, t, masks, n_t, rng)
        if best is None or len(res) < len(best):
            best = res

    if best is None:
        # Fallback: pure deterministic greedy (should not happen within budget).
        best = _greedy(v, k, t, masks, n_t, rng, use_random=False, max_blocks=max_blocks)
        if best is None:
            best = []

    return [list(b) for b in best]
