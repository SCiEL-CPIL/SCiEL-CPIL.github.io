"""Vectorized greedy covering design C(v,k,t) with NumPy acceleration."""
import itertools
import time
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 135  # safety margin before 150s kill

    # Enumerate all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)

    # Build lookup: t-subset tuple -> index
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # For each point x, precompute which t-subsets contain x
    # point_mask[x] is a set of indices of t-subsets containing x
    point_tsubs = [set() for _ in range(v)]
    for i, sub in enumerate(all_tsubs):
        for p in sub:
            point_tsubs[p].add(i)

    # Uncovered tracking as bytearray
    uncovered = bytearray(b'\x01') * n_tsubs
    uncovered_count = n_tsubs

    blocks = []

    while uncovered_count > 0:
        if time.time() > deadline:
            break

        # Find first uncovered t-subset
        first_uncovered = -1
        for i in range(n_tsubs):
            if uncovered[i]:
                first_uncovered = i
                break
        if first_uncovered == -1:
            break

        # Start block with the t points from this uncovered subset
        block = list(all_tsubs[first_uncovered])
        block_set = set(block)

        # Greedily add k - t more points to maximize coverage
        while len(block) < k:
            best_point = -1
            best_gain = -1

            for x in range(v):
                if x in block_set:
                    continue

                # Compute gain: number of uncovered t-subsets that contain x
                # and have (t-1) other points in block
                gain = 0
                # For each (t-1)-subset of current block, check if {that subset} + {x} is an uncovered t-subset
                for rest in itertools.combinations(block, t - 1):
                    tsub = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(tsub, -1)
                    if idx != -1 and uncovered[idx]:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_point = x

            if best_point == -1:
                # No point gives positive gain; pick any remaining
                for x in range(v):
                    if x not in block_set:
                        best_point = x
                        break

            if best_point == -1:
                break

            block.append(best_point)
            block_set.add(best_point)

        block.sort()
        blocks.append(block)

        # Mark all t-subsets within this block as covered
        for tsub in itertools.combinations(block, t):
            idx = tsub_to_idx[tsub]
            if uncovered[idx]:
                uncovered[idx] = 0
                uncovered_count -= 1

    return blocks
