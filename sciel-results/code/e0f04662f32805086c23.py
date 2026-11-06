"""Structural rewrite: build a valid network, then greedily delete redundant
comparators (verified via the 0-1 principle) to minimize comparator count.
"""
import time
import random

N = 17
NUM_BOOL = 1 << N
DEADLINE_MARGIN = 8.0          # stop searching this far before the 90s kill
MAX_COMPARATORS = 4096


def _precompute_masks():
    """For each (i, j) and each of the 2^N boolean inputs, the two resulting
    states after a compare-exchange (min->i, max->j)."""
    masks = []
    for i in range(N):
        for j in range(i + 1, N):
            lo = 1 << i
            hi = 1 << j
            hi_mask = hi - lo
            arr = [0] * NUM_BOOL
            for s in range(NUM_BOOL):
                if (s & hi) and not (s & lo):
                    arr[s] = (s & ~hi_mask) | lo
                else:
                    arr[s] = s
            masks.append(arr)
    return masks


def _apply(arr, state):
    return arr[state]


def _is_valid(comps, masks):
    """True iff the network sorts every 0-1 input (final state == 0)."""
    for s in range(1, NUM_BOOL):
        cur = s
        for i, j in comps:
            cur = masks[i * N + j][cur]
            if cur == 0:
                break
        if cur != 0:
            return False
    return True


def _build_seed():
    """Odd-even transposition (136 comparators)."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _build_random(depth):
    """Random network of given depth (each stage has all 8 disjoint pairs)."""
    comps = []
    for _ in range(depth):
        perm = list(range(N))
        random.shuffle(perm)
        for a in range(0, N - 1, 2):
            i, j = perm[a], perm[a + 1]
            if i > j:
                i, j = j, i
            comps.append((i, j))
    return comps


def construct():
    start = time.time()
    deadline = start + 90.0 - DEADLINE_MARGIN
    masks = _precompute_masks()
    rng = random.Random(12345)

    # 1) Find an initial valid network.
    comps = _build_seed()
    if not _is_valid(comps, masks):
        comps = _build_random(22)
    while not _is_valid(comps, masks):
        comps = _build_random(24)
        if time.time() > deadline:
            return comps[:MAX_COMPARATORS]

    # 2) Greedy deletion of redundant comparators (verified by 0-1 principle).
    comps = list(comps)
    while time.time() < deadline:
        changed = False
        for k in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                break
            trial = comps[:k] + comps[k + 1:]
            if _is_valid(trial, masks):
                comps = trial
                changed = True
                break
        if not changed:
            # No single removal helps; try a random short shuffle to escape.
            if len(comps) < MAX_COMPARATORS and rng.random() < 0.3:
                a = rng.randrange(len(comps))
                b = rng.randrange(len(comps))
                comps[a], comps[b] = comps[b], comps[a]
            else:
                break
        if len(comps) <= 71:
            break

    return comps[:MAX_COMPARATORS]
