"""Rewritten program: greedy + randomized restarts with incremental checks."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64

def _try_insert(sums, diffs, s, x, other_diffs):
    """Return (new_sums, new_diffs) if x can be added, else None.
    sums: sorted list of pair sums (a<=b). diffs: sorted list of nonzero diffs (a<b).
    s: current sorted set. x: candidate (x > s[-1] guaranteed by caller)."""
    # new sums: 2x and x + a for each a in s
    new_sums = [x + a for a in s]
    new_sums.append(2 * x)
    # check sum uniqueness
    if len(set(new_sums)) != len(new_sums):
        return None
    if sums:
        # merge two sorted lists, check for collision with existing sums
        i = j = 0
        combined = []
        while i < len(sums) and j < len(new_sums):
            if sums[i] < new_sums[j]:
                combined.append(sums[i]); i += 1
            elif sums[i] > new_sums[j]:
                combined.append(new_sums[j]); j += 1
            else:
                return None  # collision
        combined.extend(sums[i:])
        combined.extend(new_sums[j:])
    else:
        combined = new_sums
    # new diffs: x - a for each a in s (all positive since x > a)
    new_diffs = [x - a for a in s]
    if diffs:
        # check collision with existing diffs
        i = j = 0
        merged = []
        while i < len(diffs) and j < len(new_diffs):
            if diffs[i] < new_diffs[j]:
                merged.append(diffs[i]); i += 1
            elif diffs[i] > new_diffs[j]:
                merged.append(new_diffs[j]); j += 1
            else:
                return None  # collision
        merged.extend(diffs[i:])
        merged.extend(new_diffs[j:])
    else:
        merged = new_diffs
    # check disjointness with other_diffs (both sorted)
    if other_diffs:
        i = j = 0
        while i < len(merged) and j < len(other_diffs):
            if merged[i] < other_diffs[j]:
                i += 1
            elif merged[i] > other_diffs[j]:
                j += 1
            else:
                return None  # intersection
    return combined, merged

def _greedy(order, deadline, limit_each):
    """Build two sets greedily following the candidate order."""
    A, B = [], []
    sumA, diffA = [], []
    sumB, diffB = [], []
    for x in order:
        if len(A) < limit_each:
            r = _try_insert(sumA, diffA, A, x, diffB)
            if r is not None:
                sumA, diffA = r
                A.append(x)
        if len(B) < limit_each:
            r = _try_insert(sumB, diffB, B, x, diffA)
            if r is not None:
                sumB, diffB = r
                B.append(x)
        if len(A) >= limit_each and len(B) >= limit_each:
            break
        if time.time() > deadline:
            break
    return A, B

def construct():
    start = time.time()
    deadline = start + 80  # safety margin before 90s kill
    best = (0, [], [])
    rng = random.Random(12345)

    # Deterministic baseline: ascending
    A, B = _greedy(list(range(1, N + 1)), deadline, MAX_EACH)
    if len(A) + len(B) > best[0]:
        best = (len(A) + len(B), A, B)

    # Randomized restarts with various orderings
    while time.time() < deadline - 2:
        r = rng.random()
        if r < 0.4:
            order = list(range(1, N + 1))
            rng.shuffle(order)
        elif r < 0.7:
            # reverse
            order = list(range(N, 0, -1))
        elif r < 0.85:
            # biased: more from upper half
            order = list(range(1, N + 1))
            # swap some random pairs to perturb
            for _ in range(rng.randint(20, 80)):
                i, j = rng.randrange(N), rng.randrange(N)
                order[i], order[j] = order[j], order[i]
        else:
            # random permutation
            order = list(range(1, N + 1))
            rng.shuffle(order)
        A, B = _greedy(order, deadline, MAX_EACH)
        if len(A) + len(B) > best[0]:
            best = (len(A) + len(B), A, B)

    return [best[1], best[2]]
# EVOLVE-BLOCK-END
