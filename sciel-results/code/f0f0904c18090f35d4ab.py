<<<<<<< SEARCH
import itertools
import random
import time
from collections import defaultdict
import math


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    uncovered = set(frozenset(c) for c in itertools.combinations(range(v), t))
    blocks = []
    block_set = set()

    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed = rng.choice(tuple(uncovered))
        block = list(seed)
        rng.shuffle(block)

        # Greedily add points to maximize coverage
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)

            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)

        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            blocks.append(b_tuple)
            block_set.add(b_tuple)
            for c in itertools.combinations(block, t):
                uncovered.discard(frozenset(c))

    return blocks
=======
import itertools
import random
import time
from collections import defaultdict
import math


def _subset_to_idx(v, t, subset):
    """Convert a t-subset (sorted tuple) to a unique index in [0, C(v,t))."""
    # Use combinatorial number system
    idx = 0
    for i, x in enumerate(subset):
        # C(x, i)
        idx += math.comb(x, i)
    return idx


def _idx_to_subset(v, t, idx):
    """Convert an index back to a t-subset."""
    subset = []
    for i in range(t - 1, -1, -1):
        x = v - 1
        while math.comb(x, i) > idx:
            x -= 1
        subset.append(x)
        idx -= math.comb(x, i)
    return tuple(sorted(subset))


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    total_t = math.comb(v, t)
    covered = [0] * total_t
    blocks = []
    block_set = set()

    # Precompute all t-subsets and their indices
    all_ts = []
    for c in itertools.combinations(range(v), t):
        idx = _subset_to_idx(v, t, c)
        all_ts.append((idx, c))

    while True:
        # Find an uncovered t-subset
        uncovered_indices = [i for i, (idx, c) in enumerate(all_ts) if covered[idx] == 0]
        if not uncovered_indices:
            break

        # Pick a random uncovered t-subset as seed
        seed_idx = rng.choice(uncovered_indices)
        seed = all_ts[seed_idx][1]
        block = list(seed)
        rng.shuffle(block)

        # Greedily add points to maximize coverage
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)

            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = tuple(sorted(list(rest) + [x]))
                    fs_idx = _subset_to_idx(v, t, fs)
                    if covered[fs_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)

        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            blocks.append(b_tuple)
            block_set.add(b_tuple)
            for c in itertools.combinations(block, t):
                idx = _subset_to_idx(v, t, c)
                covered[idx] += 1

    return blocks
>>>>>>> REPLACE

