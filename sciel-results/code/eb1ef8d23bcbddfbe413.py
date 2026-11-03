"""Optimized covering design search for C(24,7,3).

Strategy:
1. Fast greedy construction (multiple seeds) with vectorized gain.
2. Fast reduction (remove redundant blocks).
3. Local search: block swap and replacement to reduce block count.
"""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def construct(v, k, t):
    deadline = time.time() + 110.0  # Safety margin

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}

    # Precompute: for each point x, for each (t-1)-subset of other points, the t-subset index
    point_tmap = [dict() for _ in range(v)]
    for idx, subset in enumerate(all_t_subsets):
        for i in range(t):
            p = subset[i]
            rest = frozenset(subset[:i] + subset[i+1:])
            point_tmap[p][rest] = idx

    # Precompute block->t-subset indices mapping for faster coverage updates
    # block_t_subsets[block_tuple] = list of t-subset indices covered by that block
    block_t_subsets_cache = {}

    def get_block_t_subsets(block):
        """Get or compute the list of t-subset indices covered by a block."""
        key = tuple(sorted(block))
        if key not in block_t_subsets_cache:
            indices = []
            for subset in itertools.combinations(block, t):
                indices.append(t_subset_to_idx[subset])
            block_t_subsets_cache[key] = indices
        return block_t_subsets_cache[key]

    def greedy_cover(seed, rng=None):
        """Greedy covering with randomization and vectorized gain."""
        if rng is None:
            rng = random.Random(seed)
        uncovered = np.ones(n_t_subsets, dtype=bool)
        uncovered_set = set(range(n_t_subsets))
        blocks = []

        while uncovered_set:
            if time.time() > deadline - 15:
                return None

            # Pick a random uncovered t-subset
            start_idx = rng.choice(list(uncovered_set))
            uncovered_set.discard(start_idx)
            uncovered[start_idx] = False

            block = list(all_t_subsets[start_idx])
            rng.shuffle(block)
            block_set = set(block)

            # Add points to the block greedily
            while len(block) < k:
                if time.time() > deadline - 15:
                    return None

                # Compute gain for each candidate point using vectorized approach
                best_x = -1
                best_gain = -1
                best_gain_exact = False

                # Precompute all (t-1)-subsets of current block
                block_t1_subs = list(itertools.combinations(block, t - 1))

                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in block_t1_subs:
                        idx = point_tmap[x].get(frozenset(rest), -1)
                        if idx != -1 and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_gain_exact = True
                    elif gain == best_gain and best_gain > 0:
                        if rng.random() < 0.3:
                            best_x = x

                if best_x < 0:
                    return None

                block.append(best_x)
                block_set.add(best_x)

            block.sort()
            blocks.append(block)

            # Update uncovered using cached t-subset indices
            for idx in get_block_t_subsets(block):
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_set.discard(idx)

        return blocks

    def reduce_blocks(blocks):
        """Remove redundant blocks iteratively with incremental coverage."""
        if not blocks:
            return []

        blocks = list(blocks)
        improved = True

        while improved:
            if time.time() > deadline - 8:
                break
            improved = False

            # Compute coverage counts using cached t-subset indices
            coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
            for block in blocks:
                for idx in get_block_t_subsets(block):
                    coverage_count[idx] += 1

            # Try removing each block (reverse order for efficiency)
            for bi in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 8:
                    break

                block = blocks[bi]
                affected_indices = get_block_t_subsets(block)
                can_remove = True

                for idx in affected_indices:
                    coverage_count[idx] -= 1
                    if coverage_count[idx] == 0:
                        can_remove = False
                        break

                if can_remove:
                    blocks.pop(bi)
                    improved = True
                    break
                else:
                    for a_idx in affected_indices:
                        coverage_count[a_idx] += 1

        return blocks

    def local_search(blocks):
        """Try to reduce block count via pair-merge local search."""
        if not blocks:
            return blocks

        blocks = list(blocks)
        improved = True
        max_iterations = 50  # Limit total iterations to avoid getting stuck

        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            improved = False

            n = len(blocks)
            if n < 2:
                break

            # Compute coverage counts
            coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
            for block in blocks:
                for idx in get_block_t_subsets(block):
                    coverage_count[idx] += 1

            # For each pair of blocks, try to merge them into one
            found_merge = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 5:
                        break

                    # Get t-subsets uniquely covered by block i or block j
                    unique_subsets = set()
                    for idx in get_block_t_subsets(blocks[i]):
                        if coverage_count[idx] == 1:
                            unique_subsets.add(idx)
                    for idx in get_block_t_subsets(blocks[j]):
                        if coverage_count[idx] == 1:
                            unique_subsets.add(idx)

                    if not unique_subsets:
                        # One or both blocks are fully redundant
                        blocks.pop(j)
                        improved = True
                        found_merge = True
                        break
                        continue

                    # Try to find a k-subset that covers all unique_subsets
                    # Start with the union of points from both blocks
                    union_points = set(blocks[i]) | set(blocks[j])

                    if len(union_points) <= k:
                        candidate = sorted(union_points)
                        if len(candidate) == k:
                            # Check if this candidate covers all unique_subsets
                            cand_set = set(candidate)
                            ok = True
                            for idx in unique_subsets:
                                sub = all_t_subsets[idx]
                                if not all(p in cand_set for p in sub):
                                    ok = False
                                    break
                            if ok:
                                # Remove both i and j, add candidate
                                new_blocks = blocks[:i] + blocks[i+1:j] + blocks[j+1:]
                                new_blocks.append(candidate)
                                # Quick verification: check coverage of unique subsets
                                # and that no other subset becomes uncovered
                                cov_check = np.zeros(n_t_subsets, dtype=np.int32)
                                for b in new_blocks:
                                    for idx in get_block_t_subsets(b):
                                        cov_check[idx] += 1
                                if np.all(cov_check > 0):
                                    blocks = new_blocks
                                    improved = True
                                    found_merge = True
                                    break
                                    continue

                    # If union is smaller than k, try adding points greedily
                    if len(union_points) < k:
                        remaining_points = [p for p in range(v) if p not in union_points]
                        candidate = list(union_points)
                        cand_set = set(candidate)
                        remaining_needed = set(unique_subsets)

                        for _ in range(k - len(union_points)):
                            if time.time() > deadline - 5:
                                break
                            best_p = -1
                            best_gain = -1
                            for p in remaining_points:
                                if p in cand_set:
                                    continue
                                new_cand_set = cand_set | {p}
                                gain = 0
                                for idx in remaining_needed:
                                    sub = all_t_subsets[idx]
                                    if all(pt in new_cand_set for pt in sub):
                                        gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_p = p
                            if best_p < 0 or best_gain == 0:
                                break
                            candidate.append(best_p)
                            cand_set.add(best_p)
                            remaining_points.remove(best_p)
                            remaining_needed = set()
                            for idx in unique_subsets:
                                sub = all_t_subsets[idx]
                                if not all(pt in cand_set for pt in sub):
                                    remaining_needed.add(idx)

                        if len(candidate) == k and not remaining_needed:
                            candidate.sort()
                            new_blocks = blocks[:i] + blocks[i+1:j] + blocks[j+1:]
                            new_blocks.append(candidate)
                            cov_check = np.zeros(n_t_subsets, dtype=np.int32)
                            for b in new_blocks:
                                for idx in get_block_t_subsets(b):
                                    cov_check[idx] += 1
                            if np.all(cov_check > 0):
                                blocks = new_blocks
                                improved = True
                                found_merge = True
                                break
                                continue

                if found_merge:
                    break

            if not improved:
                break

        return blocks

    best_blocks = None
    best_count = float('inf')

    # Phase 1: Multiple greedy constructions
    seed = 0
    while time.time() < deadline - 20:
        result = greedy_cover(seed)
        if result is not None:
            reduced = reduce_blocks(result)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_blocks = reduced
        seed += 1

    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = np.ones(n_t_subsets, dtype=bool)
        uncovered_set = set(range(n_t_subsets))
        blocks = []

        while uncovered_set:
            start_idx = min(uncovered_set)
            uncovered_set.discard(start_idx)
            uncovered[start_idx] = False

            block = list(all_t_subsets[start_idx])
            block_set = set(block)

            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_t1_subs = list(itertools.combinations(block, t - 1))
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in block_t1_subs:
                        idx = point_tmap[x].get(frozenset(rest), -1)
                        if idx != -1 and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x < 0:
                    break
                block.append(best_x)
                block_set.add(best_x)

            if len(block) < k:
                break

            block.sort()
            blocks.append(block)

            for idx in get_block_t_subsets(block):
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_set.discard(idx)

        best_blocks = blocks

    # Phase 2: Local search on best result
    if best_blocks is not None and time.time() < deadline - 5:
        local = local_search(best_blocks)
        if len(local) < len(best_blocks):
            best_blocks = local

    return best_blocks

# EVOLVE-BLOCK-END
