"""Greedy covering with bitmask acceleration and multi-restart for C(v,k,t)."""
import itertools
import random
import time
from functools import lru_cache

import numpy as np


def construct(v, k, t):
    """Construct a covering of t-subsets of {0..v-1} using k-subsets (blocks).

    Returns a list of blocks (each a sorted list of k distinct integers in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Precompute all t-subsets as bitmasks
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_mask_set = set()
    for comb in all_t_subsets:
        mask = 0
        for x in comb:
            mask |= (1 << x)
        t_mask_set.add(mask)
    total_t_subsets = len(t_mask_set)

    # Precompute for each pair of points, which t-subsets contain both (for gain computation)
    # Actually, let's use a different approach: for each (t-1)-subset, precompute all t-subsets containing it
    # This allows fast gain computation when adding a new point x to a partial block B:
    # gain(x) = number of uncovered t-subsets that contain x and (t-1) elements from B

    # Precompute: for each (t-1)-subset S, list of t-subsets containing S
    # Map: tuple of (t-1) points -> set of t-subset masks that contain all of them
    t_minus_1_sets = list(itertools.combinations(range(v), t - 1))
    # For each (t-1)-subset, we'll track which t-subsets contain it
    # t_superset_map[tm1_tuple] = set of t-mask values
    t_superset_map = {}
    for tm1 in t_minus_1_sets:
        t_superset_map[tm1] = set()

    for comb in all_t_subsets:
        mask = 0
        for x in comb:
            mask |= (1 << x)
        # For each (t-1)-subset of this t-subset, add mask
        for i in range(t):
            tm1 = tuple(comb[:i] + comb[i+1:])
            t_superset_map[tm1].add(mask)

    # We'll use a random restart approach with the best greedy found
    best_blocks = None
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Number of restarts to attempt
    max_restarts = 50

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        rng = random.Random(42 + restart)

        # Greedy covering with bitmasks
        uncovered = set(t_mask_set)
        blocks = []

        while uncovered:
            if time.time() > deadline:
                break

            # Pick the first uncovered t-subset (or random one for diversity)
            if restart == 0:
                first_uncovered = min(uncovered)
            else:
                # Pick a random uncovered t-subset for diversity
                first_uncovered = rng.choice(tuple(uncovered))

            # Find the (t)-subset corresponding to this mask
            # Convert mask to sorted tuple of points
            first_points = []
            for i in range(v):
                if first_uncovered & (1 << i):
                    first_points.append(i)
            block = list(first_points)

            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block) < k:
                best_point = -1
                best_gain = -1

                for x in range(v):
                    if x in block:
                        continue

                    # Compute gain: number of uncovered t-subsets that contain x and (t-1) elements from block
                    gain = 0
                    # For each (t-1)-subset of block, count uncovered t-subsets containing it and x
                    for tm1 in itertools.combinations(block, t - 1):
                        tm1_tuple = tuple(tm1)
                        # The t-subset would be tm1_tuple + (x,)
                        target_mask = 0
                        for pt in tm1_tuple:
                            target_mask |= (1 << pt)
                        target_mask |= (1 << x)
                        if target_mask in uncovered:
                            gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_point = x

                if best_point == -1:
                    # No point adds coverage; just pick any remaining point
                    remaining = [x for x in range(v) if x not in block]
                    if remaining:
                        best_point = rng.choice(remaining)
                    else:
                        break

                block.append(best_point)

            block.sort()
            blocks.append(block)

            # Remove all t-subsets covered by this block from uncovered
            for comb in itertools.combinations(block, t):
                mask = 0
                for x in comb:
                    mask |= (1 << x)
                uncovered.discard(mask)

        if not uncovered and (best_blocks is None or len(blocks) < len(best_blocks)):
            best_blocks = blocks
            # If we've found a good solution, we can try a few more restarts
            # but if time is tight, stop early
            if time.time() > deadline - 10:
                break

    # If we never completed a full cover (shouldn't happen for reasonable params),
    # fall back to a simple deterministic greedy
    if best_blocks is None:
        uncovered = set(t_mask_set)
        best_blocks = []
        while uncovered:
            first_uncovered = min(uncovered)
            first_points = []
            for i in range(v):
                if first_uncovered & (1 << i):
                    first_points.append(i)
            block = list(first_points)
            while len(block) < k:
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for tm1 in itertools.combinations(block, t - 1):
                        tm1_tuple = tuple(tm1)
                        target_mask = 0
                        for pt in tm1_tuple:
                            target_mask |= (1 << pt)
                        target_mask |= (1 << x)
                        if target_mask in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    remaining = [x for x in range(v) if x not in block]
                    if remaining:
                        best_point = remaining[0]
                    else:
                        break
                block.append(best_point)
            block.sort()
            best_blocks.append(block)
            for comb in itertools.combinations(block, t):
                mask = 0
                for x in comb:
                    mask |= (1 << x)
                uncovered.discard(mask)

    return best_blocks
