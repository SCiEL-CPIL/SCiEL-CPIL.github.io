import time
import random
import itertools
from math import comb

# Precompute all t-subsets and their bit positions for the given (v, t)
def _precompute(v, t):
    subs = list(itertools.combinations(range(v), t))
    n = len(subs)
    # map each t-subset to an index
    index = {}
    for i, c in enumerate(subs):
        index[c] = i
    return subs, n, index


def _bitsets_for_point(v, t, index):
    # For each point x and each (t-1)-subset of the other points, the triplet
    # formed is (t-1)-subset + x. Precompute, for each x, a list of bit positions
    # of triplets that contain x (as a list of (rest_tuple, bitpos)).
    # Actually we want, given current block B, the triplets completed by adding x:
    #   for each (t-1)-subset R of B, triplet = R + x.
    # We'll compute gain on the fly using combinations of B.
    pass


def _popcount(x):
    return x.bit_count()


def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s / 150s kill

    subs, n, index = _precompute(v, t)
    # full mask of all triplets
    full_mask = (1 << n) - 1

    # For fast gain computation: precompute for each point x a dict:
    #   point_mask[x][rest_tuple_of_size_t-1] = bitmask of triplets containing x
    # where rest_tuple is a (t-1)-tuple of points not including x.
    # Then gain of adding x to block B = popcount(OR over R in C(B, t-1) of point_mask[x][R] & uncovered)
    # Precompute point_masks[x] as dict mapping rest_tuple -> bitmask.
    point_masks = [None] * v
    for x in range(v):
        d = {}
        for rest in itertools.combinations([p for p in range(v) if p != x], t - 1):
            tri = tuple(sorted(rest + (x,)))
            d[rest] = 1 << index[tri]
        point_masks[x] = d

    def _gain(x, block, uncovered):
        # block is a list of points (size < k). Compute new triplets completed by x.
        mask = 0
        for rest in itertools.combinations(block, t - 1):
            m = point_masks[x].get(rest)
            if m:
                mask |= m
        return _popcount(mask & uncovered)

    def _greedy(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        blocks = []
        # initial block: pick a random uncovered triplet to anchor
        # Find an uncovered triplet
        def _pick_uncovered_triplet():
            # find a set bit
            u = uncovered
            if u == 0:
                return None
            # lowest set bit
            lsb = u & (-u)
            pos = lsb.bit_length() - 1
            return subs[pos]
        # start with first uncovered triplet (randomized order via seed on point order)
        first = _pick_uncovered_triplet()
        if first is None:
            return []
        block = list(first)
        # randomize starting point order
        rng.shuffle(block)
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            # compute gains
            best_gain = -1
            best_list = []
            for x in candidates:
                g = _gain(x, block, uncovered)
                if g > best_gain:
                    best_gain = g
                    best_list = [x]
                elif g == best_gain:
                    best_list.append(x)
            # pick randomly among ties (randomized greedy)
            block.append(rng.choice(best_list))
        # finalize block
        block.sort()
        blocks.append(block)
        # update uncovered: remove all triplets of this block
        mask = 0
        for tri in itertools.combinations(block, t):
            mask |= 1 << index[tri]
        uncovered &= ~mask
        # continue until all covered
        while uncovered:
            # anchor on a random uncovered triplet
            first = _pick_uncovered_triplet()
            if first is None:
                break
            block = list(first)
            rng.shuffle(block)
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                best_gain = -1
                best_list = []
                for x in candidates:
                    g = _gain(x, block, uncovered)
                    if g > best_gain:
                        best_gain = g
                        best_list = [x]
                    elif g == best_gain:
                        best_list.append(x)
                block.append(rng.choice(best_list))
            block.sort()
            blocks.append(block)
            mask = 0
            for tri in itertools.combinations(block, t):
                mask |= 1 << index[tri]
            uncovered &= ~mask
            if time.time() > deadline:
                break
        # check if fully covered
        if uncovered != 0:
            return None  # incomplete
        return blocks

    best = None
    seed = 0
    while time.time() < deadline:
        blocks = _greedy(seed)
        if blocks is not None:
            # verify distinctness and validity quickly
            if len(set(tuple(b) for b in blocks)) == len(blocks):
                if best is None or len(blocks) < len(best):
                    best = blocks
        seed += 1
        if time.time() > deadline - 5.0:
            break

    if best is None:
        # fallback: deterministic greedy
        uncovered = full_mask
        blocks = []
        while uncovered:
            u = uncovered
            lsb = u & (-u)
            pos = lsb.bit_length() - 1
            block = list(subs[pos])
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                best_gain = -1
                best_x = candidates[0]
                for x in candidates:
                    g = _gain(x, block, uncovered)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            mask = 0
            for tri in itertools.combinations(block, t):
                mask |= 1 << index[tri]
            uncovered &= ~mask
        best = blocks

    return best
