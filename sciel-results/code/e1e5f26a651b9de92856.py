import itertools
import random
import time
import sys
import numpy as np

def construct(v, k, t):
    """
    Covering design C(v,k,t): find few k-subsets of {0,...,v-1}
    such that every t-subset is contained in at least one block.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    start_time = time.time()

    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)

    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    best_blocks = None
    best_count = float('inf')

    # Precompute all k-subsets? Too many for C(19,7)=50388, but manageable.
    # Actually C(19,7) = 50388, which is fine to enumerate.
    all_ksubsets = list(itertools.combinations(range(v), k))
    n_ksubsets = len(all_ksubsets)

    # For each k-subset, precompute the set of t-subsets it covers
    # This is a boolean matrix of shape (n_ksubsets, n_tsubsets)
    # 50388 * 3876 = ~195M booleans = ~200MB, might be too much.
    # Alternative: for each k-subset, compute list of t-subset indices it covers.

    # Let's use a different approach: greedy with random restarts
    # For each restart, shuffle the uncovered t-subsets and greedily pick blocks.

    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = set(range(n_tsubsets))
        blocks = []

        # Order t-subsets randomly for this restart
        order = list(range(n_tsubsets))
        rng.shuffle(order)

        # We'll process uncovered t-subsets in random order
        # For each uncovered t-subset, find the best k-subset that covers it
        # and maximizes the number of other uncovered t-subsets covered.

        # Build a mapping: for each point x, list of t-subsets containing x
        point_to_tsubs = [[] for _ in range(v)]
        for i, ts in enumerate(all_tsubsets):
            for x in ts:
                point_to_tsubs[x].append(i)

        while uncovered:
            if time.time() > deadline:
                return None  # timeout, abort this restart

            # Pick the first uncovered t-subset in our random order
            # Find it efficiently
            target = None
            for idx in order:
                if idx in uncovered:
                    target = idx
                    break
            if target is None:
                break

            target_tsub = all_tsubsets[target]

            # Find all k-subsets that contain target_tsub
            # A k-subset contains a t-subset if all t points are in the k-subset
            # We need to find k-subsets that include all points of target_tsub
            # and k-t additional points from the remaining v-t points

            remaining_points = [x for x in range(v) if x not in target_tsub]
            best_gain = -1
            best_block = None

            # Enumerate all ways to add (k-t) points from remaining_points
            for extra in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_tsub + extra))
                # Count how many uncovered t-subsets this block covers
                gain = 0
                for tsub in itertools.combinations(block, t):
                    idx = tsub_to_idx[tsub]
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = list(block)

            if best_block is None:
                # Fallback: just use a k-subset containing target_tsub
                extra = list(remaining_points[:k-t])
                best_block = sorted(list(target_tsub) + extra)

            blocks.append(best_block)

            # Update uncovered
            for tsub in itertools.combinations(best_block, t):
                idx = tsub_to_idx[tsub]
                uncovered.discard(idx)

        return blocks

    # Try multiple random restarts
    n_restarts = 0
    for seed in range(200):
        if time.time() > deadline:
            break
        blocks = greedy_cover(seed)
        if blocks is not None:
            n_restarts += 1
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                # Try to polish
                if time.time() < deadline - 20:
                    polished = polish(blocks, v, k, t, tsub_to_idx, all_tsubsets, deadline)
                    if polished is not None and len(polished) < best_count:
                        best_blocks = polished
                        best_count = len(polished)

    if best_blocks is None:
        # Fallback: pure deterministic greedy
        blocks = greedy_cover(0)
        if blocks is None:
            # Very slow fallback
            uncovered = set(itertools.combinations(range(v), t))
            blocks = []
            while uncovered and time.time() < deadline:
                block = list(min(uncovered))
                while len(block) < k:
                    best_x = 0
                    best_g = -1
                    for x in range(v):
                        if x in block:
                            continue
                        g = 0
                        for rest in itertools.combinations(block, t-1):
                            ts = tuple(sorted(rest + (x,)))
                            if ts in uncovered:
                                g += 1
                        if g > best_g:
                            best_g = g
                            best_x = x
                    block.append(best_x)
                block.sort()
                blocks.append(block)
                for ts in itertools.combinations(block, t):
                    uncovered.discard(tuple(ts))
            best_blocks = blocks
            best_count = len(blocks)

    return best_blocks


def polish(blocks, v, k, t, tsub_to_idx, all_tsubsets, deadline):
    """
    Try to reduce block count by:
    1. Removing redundant blocks (if a block's t-subsets are all covered by others)
    2. Swapping points in blocks to improve coverage
    """
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    if n == 0:
        return blocks

    # Compute coverage count for each t-subset
    cover_count = [0] * len(all_tsubsets)
    for block in blocks:
        for tsub in itertools.combinations(block, t):
            cover_count[tsub_to_idx[tsub]] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # Try removing each block if it's redundant
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            block = blocks[i]
            # Check if all t-subsets in this block are covered by other blocks
            redundant = True
            for tsub in itertools.combinations(block, t):
                if cover_count[tsub_to_idx[tsub]] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove this block
                for tsub in itertools.combinations(block, t):
                    cover_count[tsub_to_idx[tsub]] -= 1
                blocks.pop(i)
                improved = True
                break  # restart the loop since indices changed

        if time.time() > deadline:
            break

        # Try swapping: for each block, try replacing one point with another
        # to increase coverage of currently singly-covered t-subsets
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            block = blocks[i]
            for pos in range(len(block)):
                orig_point = block[pos]
                for new_point in range(v):
                    if new_point == orig_point or new_point in block:
                        continue
                    # Try the swap
                    new_block = block[:]
                    new_block[pos] = new_point
                    new_block.sort()

                    # Count change in coverage
                    gain = 0
                    for tsub in itertools.combinations(block, t):
                        idx = tsub_to_idx[tsub]
                        if idx in tsub_to_idx and cover_count[idx] == 1:
                            # This t-subset would become uncovered unless in new_block
                            if new_point not in tsub or orig_point in tsub:
                                # t-subset doesn't contain new_point, or still contains orig_point
                                # Actually need to check if t-subset is in new_block
                                pass
                    # This is getting complex; skip swaps for now
                    break
                break

    return blocks


if __name__ == "__main__":
    # Quick test
    v, k, t = 19, 7, 4
    start = time.time()
    result = construct(v, k, t)
    elapsed = time.time() - start
    print(f"C({v},{k},{t}): {len(result)} blocks in {elapsed:.1f}s")
    # Verify
    covered = set()
    for block in result:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    expected = set(itertools.combinations(range(v), t))
    print(f"Covered {len(covered)}/{len(expected)} t-subsets")
    print(f"Valid: {covered == expected}")
