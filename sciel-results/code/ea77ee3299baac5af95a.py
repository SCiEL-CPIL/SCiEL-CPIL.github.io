"""Improved covering design C(19,8,5) with vectorized greedy and local search."""
import itertools
import time
import random
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    # Enumerate all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute for each point, which t-subsets contain it
    point_t_subs = [[] for _ in range(v)]
    for i, sub in enumerate(all_t_subs):
        for p in sub:
            point_t_subs[p].append(i)

    # Precompute for each pair of points, which t-subsets contain both
    # This helps in computing gains when we have a partial block
    # For a partial block B of size b, adding point x covers t-subsets that:
    #   - contain x
    #   - have all other t-1 elements in B
    # We can precompute: for each point x, and for each subset of size t-1,
    #   the list of t-subsets containing x and that subset

    # Better approach: precompute a mapping from (frozen_set, point) to list of t-subset indices
    # where t-subset = frozen_set ∪ {point} and len(frozen_set) = t-1
    # But that's too large. Instead, let's use a different strategy.

    # For the greedy algorithm, we need to compute for a partial block B (size b) and candidate x:
    #   gain(x) = number of uncovered t-subsets s such that x ∈ s and s \ {x} ⊆ B
    # This is equivalent to: for each (t-1)-subset R of B, count uncovered t-subsets of form R ∪ {x}

    # Precompute: for each (t-1)-subset, a dict mapping point -> t-subset index
    t_minus_1_sub_to_point_to_idx = defaultdict(dict)
    for i, sub in enumerate(all_t_subs):
        for j in range(t):
            # Remove element j to get a (t-1)-subset
            rest = tuple(sorted(list(sub[:j]) + list(sub[j+1:])))
            point = sub[j]
            t_minus_1_sub_to_point_to_idx[rest][point] = i

    # Convert to a more efficient structure: for each (t-1)-subset, a dict point->idx
    # We'll use this to quickly look up gains

    def greedy_cover(seed=None):
        if seed is not None:
            random.seed(seed)

        covered = bytearray(n_t)
        blocks = []
        n_covered = 0

        while n_covered < n_t:
            # Find first uncovered t-subset
            first_uncovered = -1
            for i in range(n_t):
                if not covered[i]:
                    first_uncovered = i
                    break
            if first_uncovered == -1:
                break

            # Check deadline
            if time.time() > deadline - 5:
                break

            # Start block with the points from the first uncovered t-subset
            block = list(all_t_subs[first_uncovered])
            block_set = set(block)

            # Mark t-subsets covered by the initial t points
            # The initial t points form exactly one t-subset: all_t_subs[first_uncovered]
            if not covered[first_uncovered]:
                covered[first_uncovered] = 1
                n_covered += 1

            # Build the block greedily
            while len(block) < k:
                best_x = -1
                best_gain = -1

                # For each candidate point not in block, compute gain
                for x in range(v):
                    if x in block_set:
                        continue

                    # Gain = number of uncovered t-subsets s such that x ∈ s and s \ {x} ⊆ block
                    # s \ {x} is a (t-1)-subset of block
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        rest_frozen = tuple(sorted(rest))
                        if x in t_minus_1_sub_to_point_to_idx[rest_frozen]:
                            idx = t_minus_1_sub_to_point_to_idx[rest_frozen][x]
                            if not covered[idx]:
                                gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x == -1 or best_gain <= 0:
                    # Can't improve with coverage, add a random point
                    candidates = [x for x in range(v) if x not in block_set]
                    if candidates:
                        best_x = random.choice(candidates)
                    else:
                        break

                block.append(best_x)
                block_set.add(best_x)

                # Mark newly covered t-subsets
                for sub in itertools.combinations(block, t):
                    idx = t_sub_to_idx[sub]
                    if not covered[idx]:
                        covered[idx] = 1
                        n_covered += 1

            block.sort()
            blocks.append(block)

        return blocks

    # Run greedy with a fixed seed first
    blocks = greedy_cover(seed=42)

    # Check if complete
    def is_complete(blocks):
        covered = bytearray(n_t)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                covered[t_sub_to_idx[sub]] = 1
        return all(covered)

    if not is_complete(blocks):
        # Try different seeds
        for seed in range(1, 30):
            if time.time() > deadline - 10:
                break
            b = greedy_cover(seed=seed)
            if is_complete(b) and len(b) < len(blocks):
                blocks = b

    # Reduction pass: try to remove redundant blocks
    def try_reduce(blocks):
        if not blocks:
            return blocks
        n = len(blocks)

        # Precompute coverage per block as sets of t-subset indices
        block_cover = []
        for block in blocks:
            cov = set()
            for sub in itertools.combinations(block, t):
                cov.add(t_sub_to_idx[sub])
            block_cover.append(cov)

        changed = True
        while changed:
            changed = False
            n = len(blocks)
            for i in range(n):
                if time.time() > deadline - 5:
                    return blocks
                # Check if block i is redundant
                redundant = True
                for idx in block_cover[i]:
                    found = False
                    for j in range(n):
                        if j == i:
                            continue
                        if idx in block_cover[j]:
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    block_cover.pop(i)
                    changed = True
                    break  # restart loop since indices changed
        return blocks

    if time.time() < deadline - 15:
        blocks = try_reduce(blocks)

    # Local search: try to replace a block with a better one
    def local_search(blocks):
        if not blocks:
            return blocks

        # Precompute coverage counts
        cov_count = np.zeros(n_t, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cov_count[t_sub_to_idx[sub]] += 1

        # Find blocks that can be improved
        # For each block, try to find a replacement that covers more uncovered subsets
        # while maintaining coverage of all subsets that are only covered by this block

        n = len(blocks)
        for i in range(n):
            if time.time() > deadline - 5:
                return blocks

            # Find t-subsets only covered by block i
            unique_to_block = set()
            for sub in itertools.combinations(blocks[i], t):
                idx = t_sub_to_idx[sub]
                if cov_count[idx] == 1:
                    unique_to_block.add(idx)

            if not unique_to_block:
                continue

            # Try to find a new block that covers all unique_to_block and maximizes additional coverage
            # This is expensive, so we'll try a heuristic approach
            # Start with the points that must be in the new block (from unique_to_block)
            must_points = set()
            for idx in unique_to_block:
                must_points.update(all_t_subs[idx])

            if len(must_points) > k:
                continue  # Can't fit all must-points in one block

            # Build a candidate block starting with must_points
            candidate = list(must_points)
            candidate_set = set(candidate)

            # Compute current coverage of candidate
            candidate_cov = set()
            for sub in itertools.combinations(candidate, t):
                candidate_cov.add(t_sub_to_idx[sub])

            # Greedily add points to maximize coverage
            while len(candidate) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in candidate_set:
                        continue
                    # Compute how many new t-subsets this would cover
                    gain = 0
                    for sub in itertools.combinations(candidate + [x], t):
                        if sub[-1] != x:  # Only count subsets containing x
                            continue
                        idx = t_sub_to_idx[sub]
                        if idx not in candidate_cov:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    candidates = [x for x in range(v) if x not in candidate_set]
                    if candidates:
                        best_x = candidates[0]
                    else:
                        break
                candidate.append(best_x)
                candidate_set.add(best_x)
                for sub in itertools.combinations(candidate, t):
                    if sub[-1] == best_x:
                        candidate_cov.add(t_sub_to_idx[sub])

            # Check if candidate covers all unique_to_block
            if not unique_to_block.issubset(candidate_cov):
                continue

            # Count how many additional t-subsets the candidate covers
            # that are not covered by the old block
            old_cov = set()
            for sub in itertools.combinations(blocks[i], t):
                old_cov.add(t_sub_to_idx[sub])

            new_gain = len(candidate_cov - old_cov)
            if new_gain > 0:
                # Replace the block
                # Update coverage counts
                for idx in old_cov:
                    cov_count[idx] -= 1
                for idx in candidate_cov:
                    if idx not in old_cov:
                        cov_count[idx] += 1
                blocks[i] = sorted(candidate)

        return blocks

    if time.time() < deadline - 10:
        blocks = local_search(blocks)

    # Final verification - if incomplete, fall back to a fresh run
    if not is_complete(blocks):
        # Emergency: run simple greedy without deadline
        covered = bytearray(n_t)
        blocks = []
        while True:
            first_uncovered = -1
            for i in range(n_t):
                if not covered[i]:
                    first_uncovered = i
                    break
            if first_uncovered == -1:
                break
            block = list(all_t_subs[first_uncovered])
            block_set = set(block)
            if not covered[first_uncovered]:
                covered[first_uncovered] = 1
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        rest_frozen = tuple(sorted(rest))
                        if x in t_minus_1_sub_to_point_to_idx[rest_frozen]:
                            idx = t_minus_1_sub_to_point_to_idx[rest_frozen][x]
                            if not covered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    candidates = [x for x in range(v) if x not in block_set]
                    best_x = candidates[0] if candidates else 0
                block.append(best_x)
                block_set.add(best_x)
                for sub in itertools.combinations(block, t):
                    if sub[-1] == best_x:
                        idx = t_sub_to_idx[sub]
                        if not covered[idx]:
                            covered[idx] = 1
            block.sort()
            blocks.append(block)

    return blocks
