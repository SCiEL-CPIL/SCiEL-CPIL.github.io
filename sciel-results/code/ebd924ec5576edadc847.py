import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip

    # Precompute all blocks and their masks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Precompute mask for each block
    # mask[b] is an integer with bits set for each t-subset in block b
    block_masks = np.zeros(n_blocks, dtype=object)
    block_triplet_ids = [None] * n_blocks  # list of triplet ids for each block

    for bi, block in enumerate(all_blocks):
        mask = 0
        ids = []
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
            ids.append(tid)
        block_masks[bi] = mask
        block_triplet_ids[bi] = ids

    # For fast lookup: given a set of points, find blocks containing them
    # Build index: for each point, list of block indices containing it
    point_to_blocks = defaultdict(list)
    for bi, block in enumerate(all_blocks):
        for p in block:
            point_to_blocks[p].append(bi)

    def get_blocks_containing(points):
        """Return list of block indices containing all points in the set."""
        if len(points) == 0:
            return list(range(n_blocks))
        if len(points) > k:
            return []
        # Start with blocks containing the first point, intersect with others
        first = next(iter(points))
        candidate_set = set(point_to_blocks[first])
        for p in points:
            if p == first:
                continue
            candidate_set &= set(point_to_blocks[p])
            if not candidate_set:
                return []
        return list(candidate_set)

    # Greedy Construction with ratio-based selection
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        blocks = []
        used_block_indices = set()

        while uncovered:
            # Pick a random uncovered triplet to guide the search
            idx = -1
            for _ in range(20):
                cand = random.randint(0, n_triplets - 1)
                if (uncovered >> cand) & 1:
                    idx = cand
                    break
            if idx == -1:
                lsb = uncovered & -uncovered
                idx = lsb.bit_length() - 1

            start_triplet = id_to_triplet[idx]

            # Find best block containing start_triplet using ratio: gain / cost
            best_block_idx = -1
            best_ratio = -1
            best_mask = 0

            # Get blocks containing all 3 points of start_triplet
            candidate_indices = get_blocks_containing(set(start_triplet))

            for bi in candidate_indices:
                if bi in used_block_indices:
                    continue
                b_mask = block_masks[bi]
                gain = (b_mask & uncovered).bit_count()
                if gain == 0:
                    continue
                cost = b_mask.bit_count()  # C(k,t) = 35
                ratio = gain / cost

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_block_idx = bi
                    best_mask = b_mask

            if best_block_idx == -1:
                # Fallback: allow reuse or pick any block covering the triplet
                for bi in candidate_indices:
                    b_mask = block_masks[bi]
                    gain = (b_mask & uncovered).bit_count()
                    if gain > 0:
                        best_block_idx = bi
                        best_mask = b_mask
                        break

            if best_block_idx == -1:
                # Should not happen if covering is possible
                break

            blocks.append(all_blocks[best_block_idx])
            used_block_indices.add(best_block_idx)
            uncovered &= ~best_mask

        return blocks

    # Local Search Optimizer
    def optimize(blocks, time_limit):
        counts = np.zeros(n_triplets, dtype=np.int32)

        def recalc_counts(b_list):
            counts[:] = 0
            for b in b_list:
                for triplet in itertools.combinations(b, t):
                    counts[triplet_to_id[triplet]] += 1

        recalc_counts(blocks)

        no_improve_count = 0
        max_no_improve = 30

        while time.time() < time_limit:
            improved = False
            n = len(blocks)

            # 1. Try Remove Redundant
            for i in range(n):
                b = blocks[i]
                redundant = True
                for triplet in itertools.combinations(b, t):
                    if counts[triplet_to_id[triplet]] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    for triplet in itertools.combinations(b, t):
                        counts[triplet_to_id[triplet]] -= 1
                    improved = True
                    break

            # 2. Try 2-for-1 replacement
            if not improved and n >= 2:
                # Compute unique load for each block
                unique_loads = []
                for i in range(n):
                    b = blocks[i]
                    load = 0
                    for triplet in itertools.combinations(b, t):
                        if counts[triplet_to_id[triplet]] == 1:
                            load += 1
                    unique_loads.append((load, i))

                unique_loads.sort(reverse=True)

                # Try pairs from top-loaded blocks
                tried_pairs = set()
                for load1, i in unique_loads[:15]:
                    if load1 == 0:
                        continue
                    for load2, j in unique_loads[:15]:
                        if i == j or i > j:
                            continue
                        if (i, j) in tried_pairs:
                            continue
                        tried_pairs.add((i, j))

                        b1 = blocks[i]
                        b2 = blocks[j]

                        # Get triplet ids
                        ids_b1 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b1, t))
                        ids_b2 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b2, t))
                        all_involved = ids_b1 | ids_b2

                        needed_triplets = []
                        points_needed = set()
                        for tid in all_involved:
                            if counts[tid] == 1:
                                needed_triplets.append(tid)
                                points_needed.update(id_to_triplet[tid])

                        if not needed_triplets:
                            # Both redundant
                            # Remove j first (higher index) to keep indices valid
                            blocks.pop(j)
                            for triplet in itertools.combinations(b2, t):
                                counts[triplet_to_id[triplet]] -= 1
                            blocks.pop(i)
                            for triplet in itertools.combinations(b1, t):
                                counts[triplet_to_id[triplet]] -= 1
                            improved = True
                            break

                        if len(points_needed) > k:
                            continue

                        # Find replacement block
                        candidate_indices = get_blocks_containing(points_needed)
                        found = False
                        for bi in candidate_indices:
                            new_block = all_blocks[bi]
                            if set(new_block) == set(b1) or set(new_block) == set(b2):
                                continue
                            new_ids = set(block_triplet_ids[bi])
                            covers = all(tid in new_ids for tid in needed_triplets)
                            if covers:
                                # Perform swap
                                blocks.pop(j)
                                for triplet in itertools.combinations(b2, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                blocks.pop(i)
                                for triplet in itertools.combinations(b1, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                blocks.append(new_block)
                                for tid in block_triplet_ids[bi]:
                                    counts[tid] += 1
                                improved = True
                                found = True
                                break

                        if improved:
                            break
                        if time.time() > time_limit - 3:
                            break

            # 3. Try 1-for-1 replacement
            if not improved and n >= 1:
                i = random.randint(0, n - 1)
                b1 = blocks[i]
                ids_b1 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b1, t))

                needed_triplets = []
                points_needed = set()
                for tid in ids_b1:
                    if counts[tid] == 1:
                        needed_triplets.append(tid)
                        points_needed.update(id_to_triplet[tid])

                if needed_triplets and len(points_needed) <= k:
                    candidate_indices = get_blocks_containing(points_needed)
                    for bi in candidate_indices:
                        new_block = all_blocks[bi]
                        if set(new_block) == set(b1):
                            continue
                        new_ids = set(block_triplet_ids[bi])
                        covers = all(tid in new_ids for tid in needed_triplets)
                        if covers:
                            blocks.pop(i)
                            for triplet in itertools.combinations(b1, t):
                                counts[triplet_to_id[triplet]] -= 1
                            blocks.append(new_block)
                            for tid in block_triplet_ids[bi]:
                                counts[tid] += 1
                            improved = True
                            break

            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            else:
                no_improve_count = 0

        return blocks

    best_blocks = None
    best_len = float('inf')

    # Run multiple restarts
    n_restarts = 8
    for seed in range(n_restarts):
        if time.time() > deadline - 10:
            break

        # Greedy init
        init_blocks = greedy_construct(seed)

        # Optimize with remaining time
        opt_blocks = optimize(init_blocks, deadline - 5)

        # Verify validity
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in opt_blocks:
            for triplet in itertools.combinations(b, t):
                counts[triplet_to_id[triplet]] += 1

        if np.min(counts) > 0:
            if len(opt_blocks) < best_len:
                best_len = len(opt_blocks)
                best_blocks = opt_blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0)

    return list(best_blocks)


if __name__ == "__main__":
    pass
