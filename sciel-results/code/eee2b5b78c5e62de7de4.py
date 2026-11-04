"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets."""
# EVOLVE-BLOCK-START
import time
import itertools
from math import comb
import numpy as np

_TIME_BUDGET = 140.0          # safety margin below the 150s hard kill
_DEADLINE = None


def _time_left():
    if _DEADLINE is None:
        return _TIME_BUDGET
    return max(0.0, _DEADLINE - time.time())


def construct(v, k, t):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_BUDGET

    n_t = comb(v, t)

    def tsub_to_idx(sub):
        idx = 0
        for x in sub:
            idx = idx * v + x
        return idx

    all_arr = np.empty(n_t, dtype=np.int64)
    for i, sub in enumerate(itertools.combinations(range(v), t)):
        all_arr[i] = tsub_to_idx(sub)

    covered = np.zeros(n_t, dtype=bool)
    membership = [[] for _ in range(n_t)]

    blocks = []
    blocks_set = set()

    def gain_of_point(point, block):
        if t == 1:
            return 1 if not covered[point] else 0
        rest = list(block)
        if len(rest) < (t - 1):
            return 0
        cand = []
        for sub in itertools.combinations(rest, t - 1):
            cand.append(tsub_to_idx(tuple(sorted((point,) + sub))))
        cand_arr = np.array(cand, dtype=np.int64)
        return int(np.sum(~covered[cand_arr]))

    while not covered.all():
        if _time_left() < 0.5:
            break
        uncovered_idx = np.where(~covered)[0]
        if len(uncovered_idx) == 0:
            break
        # prioritize the uncovered subset covered by the fewest existing blocks
        counts = np.array([len(membership[i]) for i in uncovered_idx], dtype=np.int64)
        minc = int(counts.min())
        candidates = uncovered_idx[counts == minc]
        anchor_idx = int(candidates[0])
        anchor = all_arr[anchor_idx]
        block = sorted(int(x) for x in anchor.tolist())

        while len(block) < k:
            if _time_left() < 0.3:
                break
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                g = gain_of_point(x, block)
                if g > best_gain:
                    best_gain = g
                    best_point = x
            if best_point < 0 or best_gain <= 0:
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
                break
            block.append(best_point)

        block.sort()
        bkey = tuple(block)
        if bkey in blocks_set:
            continue
        blocks_set.add(bkey)
        blocks.append(block)

        for sub in itertools.combinations(block, t):
            i = tsub_to_idx(sub)
            if not covered[i]:
                covered[i] = True
            membership[i].append(len(blocks) - 1)

    return blocks
# EVOLVE-BLOCK-END
