# EVOLVE-BLOCK-START
import time
import itertools
import random

_DEADLINE = 140.0


def _bitmask(block):
    m = 0
    for x in block:
        m |= (1 << x)
    return m


def _greedy(v, k, t, rng):
    # coverage[x] = number of currently-uncovered t-subsets that contain point x
    coverage = [0] * v
    for s in itertools.combinations(range(v), t):
        for x in s:
            coverage[x] += 1
    blocks = []
    while True:
        # pick the first (lexicographically) uncovered t-subset
        found = None
        for s in itertools.combinations(range(v), t):
            covered = False
            for bm in blocks:
                if all((bm >> x) & 1 for x in s):
                    covered = True
                    break
            if not covered:
                found = s
                break
        if found is None:
            break
        block = list(found)
        while len(block) < k:
            best = -1
            best_gain = -1
            for x in range(v):
                if (1 << x) & 0:
                    pass
                if any(y == x for y in block):
                    continue
                gain = coverage[x]
                if gain > best_gain:
                    best_gain = gain
                    best = x
            if best < 0:
                break
            block.append(best)
        block.sort()
        blocks.append(block)
        # update coverage for the t-subsets this block covers
        for s in itertools.combinations(block, t):
            for x in s:
                coverage[x] -= 1
    return blocks


def _local_search(v, k, t, blocks, rng, deadline):
    masks = [_bitmask(b) for b in blocks]

    def is_covered(s, masks):
        for m in masks:
            if all((m >> x) & 1 for x in s):
                return True
        return False

    def try_improve():
        nonlocal masks
        while True:
            improved = False
            n = len(masks)
            for i in range(n):
                if time.time() > deadline:
                    return False
                bi = masks[i]
                # candidate additions
                add_opts = [x for x in range(v) if not ((bi >> x) & 1)]
                # candidate removals
                rem_opts = [x for x in range(v) if (bi >> x) & 1]
                best_move = None
                best_gain = 0
                for x in add_opts:
                    if time.time() > deadline:
                        return False
                    newbi = bi | (1 << x)
                    for y in rem_opts:
                        cand = newbi & ~(1 << y)
                        # must remain a k-subset
                        if bin(cand).count('1') != k:
                            continue
                        # evaluate: does cand cover any t-subset that bi did not?
                        # and does it still cover everything bi covered?
                        gain = 0
                        ok = True
                        # check every t-subset of cand
                        for s in itertools.combinations(range(v), t):
                            if all((cand >> x2) & 1 for x2 in s):
                                # was it covered by bi?
                                if not all((bi >> x2) & 1 for x2 in s):
                                    gain += 1
                        # ensure cand covers all that bi covered
                        for s in itertools.combinations(range(v), t):
                            if all((bi >> x2) & 1 for x2 in s):
                                if not all((cand >> x2) & 1 for x2 in s):
                                    ok = False
                                    break
                        if ok and gain > best_gain:
                            best_gain = gain
                            best_move = (x, y, cand)
                if best_move is not None:
                    x, y, cand = best_move
                    masks[i] = cand
                    improved = True
                    break
            if not improved:
                break
        return True

    # remove redundant blocks
    def remove_redundant():
        nonlocal masks
        changed = True
        while changed:
            changed = False
            for i in range(len(masks)):
                if time.time() > deadline:
                    return
                masks2 = masks[:i] + masks[i+1:]
                # check if all t-subsets of masks[i] are covered by masks2
                redundant = True
                for s in itertools.combinations(range(v), t):
                    if all((masks[i] >> x) & 1 for x in s):
                        covered = False
                        for m in masks2:
                            if all((m >> x) & 1 for x in s):
                                covered = True
                                break
                        if not covered:
                            redundant = False
                            break
                if redundant:
                    masks = masks2
                    changed = True
                    break

    while time.time() < deadline:
        try_improve()
        remove_redundant()

    # convert back to lists
    result = []
    for m in masks:
        b = [x for x in range(v) if (m >> x) & 1]
        b.sort()
        result.append(b)
    return result


def _is_valid(v, k, t, blocks):
    masks = [_bitmask(b) for b in blocks]
    for s in itertools.combinations(range(v), t):
        ok = False
        for m in masks:
            if all((m >> x) & 1 for x in s):
                ok = True
                break
        if not ok:
            return False
    return True


def construct(v, k, t):
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(12345)

    # Build initial greedy cover
    blocks = _greedy(v, k, t, rng)

    # If we already beat the known best, great; else improve via local search
    if time.time() < deadline:
        improved = _local_search(v, k, t, blocks, rng, deadline)
        if improved:
            blocks = improved

    # Ensure validity; if local search broke it, fall back to greedy
    if not _is_valid(v, k, t, blocks):
        blocks = _greedy(v, k, t, rng)

    # Deduplicate
    seen = set()
    out = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out
# EVOLVE-BLOCK-END
