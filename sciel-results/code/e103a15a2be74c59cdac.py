"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 145.0  # safety margin before 150s kill
    rng = random.Random(42)

    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # For each point, list of t-subset indices containing that point
    point_tsubs = [[] for _ in range(v)]
    for ts in all_tsubs:
        idx = tsub_to_idx[ts]
        for p in ts:
            point_tsubs[p].append(idx)

    # For each t-subset, list of points in it
    tsub_points = [list(ts) for ts in all_tsubs]
    
    # Precompute for t=3: for each pair of points, the list of t-subsets containing both
    # This allows O(1) gain computation when adding a third point
    if t == 3:
        pair_to_tsubs = {}
        for ts in all_tsubs:
            a, b, c = ts
            idx = tsub_to_idx[ts]
            for pair in [(a,b), (a,c), (b,c)]:
                if pair not in pair_to_tsubs:
                    pair_to_tsubs[pair] = []
                pair_to_tsubs[pair].append(idx)
    else:
        pair_to_tsubs = None

    def greedy_cover(seed_order=None):
        """Randomized greedy set cover with optimized gain computation."""
        coverage = np.zeros(n_tsubs, dtype=np.int32)
        blocks = []

        if seed_order is None:
            seed_order = list(range(n_tsubs))
            rng.shuffle(seed_order)

        # Track uncovered count for quick check
        uncovered_count = n_tsubs

        for seed_idx in seed_order:
            if time.time() > deadline:
                return None

            if coverage[seed_idx] > 0:
                continue

            # Build a block that covers this t-subset and maximizes new coverage
            seed_tsub = all_tsubs[seed_idx]
            block_set = set(seed_tsub)
            block_list = list(seed_tsub)

            # Greedily add points
            for _ in range(k - t):
                if time.time() > deadline:
                    return None

                best_point = None
                best_gain = -1
                candidates = [p for p in range(v) if p not in block_set]
                rng.shuffle(candidates)

                for p in candidates:
                    # Calculate gain: how many new t-subsets this point covers
                    # given the current block
                    gain = 0
                    if t == 3 and pair_to_tsubs is not None:
                        # For t=3: a t-subset containing p and two points from block
                        # is covered iff the pair (from block) + p forms a t-subset
                        for other in block_list:
                            pair = (min(p, other), max(p, other))
                            if pair in pair_to_tsubs:
                                for tsub_idx in pair_to_tsubs[pair]:
                                    if coverage[tsub_idx] == 0:
                                        gain += 1
                    else:
                        for tsub_idx in point_tsubs[p]:
                            if coverage[tsub_idx] == 0:
                                ts = all_tsubs[tsub_idx]
                                others = [x for x in ts if x != p]
                                if all(o in block_set for o in others):
                                    gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p

                if best_point is not None:
                    block_set.add(best_point)
                    block_list.append(best_point)
                else:
                    break

            block = sorted(block_set)
            blocks.append(block)

            # Update coverage
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                if coverage[idx] == 0:
                    uncovered_count -= 1
                coverage[idx] += 1

            if uncovered_count == 0:
                break

        return blocks if uncovered_count == 0 else None

    def local_search_remove(blocks):
        """Try to remove redundant blocks."""
        if not blocks:
            return blocks

        coverage_count = np.zeros(n_tsubs, dtype=np.int32)
        block_tsubs = []

        for block in blocks:
            tsubs_in_block = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage_count[idx] += 1
                tsubs_in_block.append(idx)
            block_tsubs.append(tsubs_in_block)

        keep = [True] * len(blocks)

        for i in range(len(blocks)):
            if not keep[i]:
                continue
            can_remove = True
            for tsub_idx in block_tsubs[i]:
                if coverage_count[tsub_idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                keep[i] = False
                for tsub_idx in block_tsubs[i]:
                    coverage_count[tsub_idx] -= 1

        return [blocks[i] for i in range(len(blocks)) if keep[i]]

    def local_search_replace(blocks):
        """Try to replace a block with a smaller set of blocks."""
        if not blocks:
            return blocks

        coverage_count = np.zeros(n_tsubs, dtype=np.int32)
        block_tsubs = []

        for block in blocks:
            tsubs_in_block = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage_count[idx] += 1
                tsubs_in_block.append(idx)
            block_tsubs.append(tsubs_in_block)

        improved = True
        while improved:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks

                # Temporarily remove block i
                for tsub_idx in block_tsubs[i]:
                    coverage_count[tsub_idx] -= 1

                uncovered = np.where(coverage_count == 0)[0]

                if len(uncovered) == 0:
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    improved = True
                    break

                # Try to cover uncovered t-subsets with new blocks
                new_blocks = []
                covered_set = set(uncovered.tolist())

                while covered_set:
                    if time.time() > deadline:
                        for tsub_idx in block_tsubs[i]:
                            coverage_count[tsub_idx] += 1
                        return blocks

                    seed_idx = min(covered_set)
                    seed_tsub = all_tsubs[seed_idx]
                    block_set = set(seed_tsub)
                    block_list = list(seed_tsub)

                    for _ in range(k - t):
                        best_point = None
                        best_gain = -1
                        candidates = [p for p in range(v) if p not in block_set]
                        rng.shuffle(candidates)

                        for p in candidates:
                            gain = 0
                            if t == 3 and pair_to_tsubs is not None:
                                for other in block_list:
                                    pair = (min(p, other), max(p, other))
                                    if pair in pair_to_tsubs:
                                        for tsub_idx in pair_to_tsubs[pair]:
                                            if tsub_idx in covered_set:
                                                gain += 1
                            else:
                                for tsub_idx in point_tsubs[p]:
                                    if tsub_idx in covered_set:
                                        ts = all_tsubs[tsub_idx]
                                        others = [x for x in ts if x != p]
                                        if all(o in block_set for o in others):
                                            gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_point = p

                        if best_point is not None:
                            block_set.add(best_point)
                            block_list.append(best_point)
                        else:
                            break

                    new_block = sorted(block_set)
                    new_blocks.append(new_block)

                    for ts in itertools.combinations(new_block, t):
                        idx = tsub_to_idx[ts]
                        if idx in covered_set:
                            covered_set.discard(idx)

                if len(new_blocks) < 1:
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    improved = True
                    break
                else:
                    # Revert
                    for tsub_idx in block_tsubs[i]:
                        coverage_count[tsub_idx] += 1

            if improved:
                coverage_count = np.zeros(n_tsubs, dtype=np.int32)
                block_tsubs = []
                for block in blocks:
                    tsubs_in_block = []
                    for ts in itertools.combinations(block, t):
                        idx = tsub_to_idx[ts]
                        coverage_count[idx] += 1
                        tsubs_in_block.append(idx)
                    block_tsubs.append(tsubs_in_block)

        return blocks

    best_blocks = None
    best_count = float('inf')

    # Run greedy with randomization and local search
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        blocks = greedy_cover()
        if blocks is None:
            continue

        blocks = local_search_remove(blocks)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b.copy() for b in blocks]

    # Try local search replace on the best solution
    if best_blocks is not None and time.time() < deadline:
        improved_blocks = local_search_replace([b.copy() for b in best_blocks])
        if improved_blocks is not None and len(improved_blocks) < best_count:
            best_count = len(improved_blocks)
            best_blocks = [b.copy() for b in improved_blocks]

    if best_blocks is not None:
        return best_blocks

    # Fallback: simple greedy without local search
    blocks = greedy_cover()
    if blocks is not None:
        return blocks

    # Last resort: original greedy
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
# EVOLVE-BLOCK-END
