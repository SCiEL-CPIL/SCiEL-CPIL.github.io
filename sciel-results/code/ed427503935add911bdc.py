"""Multi-start vectorized greedy covering with bitmask marginal gain."""
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 140.0
    tsubs = list(itertools.combinations(range(v), t))
    n = len(tsubs)
    tsub_to_idx = {s: i for i, s in enumerate(tsubs)}

    # Point -> list of t-subset indices containing it
    point_tsubs = [[] for _ in range(v)]
    for i, s in enumerate(tsubs):
        for p in s:
            point_tsubs[p].append(i)

    # For each point x, precompute list of (tsub_idx, other_mask) where
    # other_mask is a v-bit int with bits set for points in tsub other than x.
    x_other_masks = []
    for x in range(v):
        pairs = []
        for idx in point_tsubs[x]:
            s = tsubs[idx]
            mask = 0
            for p in s:
                if p != x:
                    mask |= (1 << p)
            pairs.append((idx, mask))
        x_other_masks.append(pairs)

    def greedy_once(seed, dl):
        rng = random.Random(seed)
        covered = np.zeros(n, dtype=bool)
        blocks = []

        while True:
            if time.time() > dl:
                break

            # Find first uncovered t-subset
            anchor = None
            for i in range(n):
                if not covered[i]:
                    anchor = i
                    break
            if anchor is None:
                break

            # Build block anchored on tsubs[anchor]
            block = list(tsubs[anchor])
            block_set = set(block)
            block_mask = 0
            for p in block:
                block_mask |= (1 << p)

            # Grow to size k using bitmask marginal gain
            while len(block) < k:
                if time.time() > dl:
                    break
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for (idx, om) in x_other_masks[x]:
                        if covered[idx]:
                            continue
                        if (om & ~block_mask) == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x is None or best_gain <= 0:
                    for x in range(v):
                        if x not in block_set:
                            block.append(x)
                            block_set.add(x)
                            block_mask |= (1 << x)
                        if len(block) == k:
                            break
                    break

                block.append(best_x)
                block_set.add(best_x)
                block_mask |= (1 << best_x)

            block.sort()
            for c in itertools.combinations(block, t):
                covered[tsub_to_idx[c]] = True
            blocks.append(block)

            if covered.all():
                break

        return blocks, covered

    # Multi-start: run greedy with different seeds, keep best
    best_blocks = None
    best_count = float('inf')
    seed = 0
    while time.time() < deadline - 10.0:
        blocks, covered = greedy_once(seed, deadline - 5.0)
        if covered.all() and len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        seed += 1

    if best_blocks is None:
        blocks, covered = greedy_once(0, deadline)
        best_blocks = blocks
        best_count = len(blocks)

    # Local improvement: drop redundant blocks
    if time.time() < deadline - 5.0 and best_blocks:
        cov_sets = [set(tsub_to_idx[c] for c in itertools.combinations(b, t)) for b in best_blocks]
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            m = len(best_blocks)
            for i in range(m):
                if time.time() > deadline - 2.0:
                    break
                others = set()
                for j in range(m):
                    if j != i:
                        others |= cov_sets[j]
                if cov_sets[i] <= others:
                    best_blocks.pop(i)
                    cov_sets.pop(i)
                    changed = True
                    break

    # Final validity check and top-up
    covered = np.zeros(n, dtype=bool)
    for b in best_blocks:
        for c in itertools.combinations(b, t):
            covered[tsub_to_idx[c]] = True
    if not covered.all():
        while not covered.all() and time.time() < deadline:
            anchor = None
            for i in range(n):
                if not covered[i]:
                    anchor = i
                    break
            if anchor is None:
                break
            block = list(tsubs[anchor])
            block_set = set(block)
            block_mask = 0
            for p in block:
                block_mask |= (1 << p)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for (idx, om) in x_other_masks[x]:
                        if covered[idx]:
                            continue
                        if (om & ~block_mask) == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None or best_gain <= 0:
                    for x in range(v):
                        if x not in block_set:
                            block.append(x)
                            block_set.add(x)
                            block_mask |= (1 << x)
                        if len(block) == k:
                            break
                    break
                block.append(best_x)
                block_set.add(best_x)
                block_mask |= (1 << best_x)
            block.sort()
            for c in itertools.combinations(block, t):
                covered[tsub_to_idx[c]] = True
            best_blocks.append(block)

    # Deduplicate
    seen = set()
    uniq = []
    for b in best_blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            uniq.append(list(b))
    return uniq
