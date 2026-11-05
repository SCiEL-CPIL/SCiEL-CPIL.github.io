import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = list(all_triplets)

    # Precompute: for each point, which triplets contain it
    point_to_triplets = defaultdict(list)
    for tid, trip in enumerate(all_triplets):
        for p in trip:
            point_to_triplets[p].append(tid)

    # Precompute masks for all possible blocks? Too many: C(24,7)=346104
    # But we can compute masks on the fly efficiently using numpy

    # Build a lookup: for each block (as a frozenset or sorted tuple), its mask
    # We'll use a cache
    block_mask_cache = {}

    def get_block_mask(block):
        """Get the bitmask of triplets covered by a block."""
        key = tuple(sorted(block))
        if key in block_mask_cache:
            return block_mask_cache[key]
        mask = 0
        for trip in itertools.combinations(block, t):
            mask |= (1 << triplet_to_id[trip])
        block_mask_cache[key] = mask
        return mask

    def blocks_containing(points):
        """Yield all k-subsets containing the given set of points."""
        points = set(points)
        if len(points) > k:
            return
        remaining = [p for p in range(v) if p not in points]
        need = k - len(points)
        if need < 0:
            return
        for comb in itertools.combinations(remaining, need):
            yield tuple(sorted(points.union(comb)))

    # Global greedy set cover
    def greedy_set_cover(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        blocks = []

        while uncovered:
            # Pick a random uncovered triplet
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

            # Find best block containing start_triplet
            best_block = None
            best_gain = 0
            best_mask = 0

            for block in blocks_containing(start_triplet):
                b_mask = get_block_mask(block)
                gain = (b_mask & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_mask = b_mask

            blocks.append(best_block)
            uncovered &= ~best_mask

        return blocks

    # Optimization: remove redundant blocks and try replacements
    def optimize(blocks, time_limit):
        # Build counts array
        counts = np.zeros(n_triplets, dtype=np.int32)

        def rebuild_counts():
            counts[:] = 0
            for b in blocks:
                for trip in itertools.combinations(b, t):
                    counts[triplet_to_id[trip]] += 1

        rebuild_counts()

        no_improve = 0
        max_no_improve = 50

        while time.time() < time_limit:
            improved = False
            n = len(blocks)

            # 1. Remove redundant blocks
            for i in range(n):
                b = blocks[i]
                redundant = True
                for trip in itertools.combinations(b, t):
                    if counts[triplet_to_id[trip]] == 1:
                        redundant = False
                        break
                if redundant:
                    for trip in itertools.combinations(b, t):
                        counts[triplet_to_id[trip]] -= 1
                    blocks.pop(i)
                    improved = True
                    break

            # 2. Try 2-for-1 replacement
            if not improved:
                n = len(blocks)
                # Compute unique loads
                unique_loads = []
                for i in range(n):
                    load = 0
                    b = blocks[i]
                    for trip in itertools.combinations(b, t):
                        if counts[triplet_to_id[trip]] == 1:
                            load += 1
                    unique_loads.append((load, i))
                unique_loads.sort(reverse=True)

                tried = set()
                max_tries = 20
                for load1, i in unique_loads[:10]:
                    if load1 == 0:
                        continue
                    for load2, j in unique_loads[:10]:
                        if i == j:
                            continue
                        key = (min(i, j), max(i, j))
                        if key in tried:
                            continue
                        tried.add(key)
                        if len(tried) >= max_tries:
                            break

                        b1 = blocks[i]
                        b2 = blocks[j]
                        ids_b1 = set(triplet_to_id[trip] for trip in itertools.combinations(b1, t))
                        ids_b2 = set(triplet_to_id[trip] for trip in itertools.combinations(b2, t))
                        all_involved = ids_b1 | ids_b2

                        needed = []
                        pts_needed = set()
                        for tid in all_involved:
                            if counts[tid] == 1:
                                needed.append(tid)
                                pts_needed.update(id_to_triplet[tid])

                        if not needed:
                            # Both redundant
                            for trip in itertools.combinations(b2, t):
                                counts[triplet_to_id[trip]] -= 1
                            blocks.pop(j)
                            for trip in itertools.combinations(b1, t):
                                counts[triplet_to_id[trip]] -= 1
                            blocks.pop(i)
                            improved = True
                            break

                        if len(pts_needed) > k:
                            continue

                        found = False
                        for new_block in blocks_containing(pts_needed):
                            if set(new_block) == set(b1) or set(new_block) == set(b2):
                                continue
                            new_ids = set(triplet_to_id[trip] for trip in itertools.combinations(new_block, t))
                            if all(tid in new_ids for tid in needed):
                                # Apply
                                for trip in itertools.combinations(b2, t):
                                    counts[triplet_to_id[trip]] -= 1
                                blocks.pop(j)
                                for trip in itertools.combinations(b1, t):
                                    counts[triplet_to_id[trip]] -= 1
                                blocks.pop(i)
                                blocks.append(new_block)
                                for trip in itertools.combinations(new_block, t):
                                    counts[triplet_to_id[trip]] += 1
                                improved = True
                                found = True
                                break
                        if improved:
                            break
                        if time.time() > time_limit - 2:
                            break

            # 3. Try 1-for-1 replacement
            if not improved:
                n = len(blocks)
                if n == 0:
                    break
                # Pick block with highest unique load
                best_i = -1
                best_load = -1
                for i in range(n):
                    load = 0
                    b = blocks[i]
                    for trip in itertools.combinations(b, t):
                        if counts[triplet_to_id[trip]] == 1:
                            load += 1
                    if load > best_load:
                        best_load = load
                        best_i = i

                if best_i == -1 or best_load == 0:
                    continue

                i = best_i
                b1 = blocks[i]
                ids_b1 = set(triplet_to_id[trip] for trip in itertools.combinations(b1, t))
                needed = []
                pts_needed = set()
                for tid in ids_b1:
                    if counts[tid] == 1:
                        needed.append(tid)
                        pts_needed.update(id_to_triplet[tid])

                if not needed or len(pts_needed) > k:
                    continue

                for new_block in blocks_containing(pts_needed):
                    if set(new_block) == set(b1):
                        continue
                    new_ids = set(triplet_to_id[trip] for trip in itertools.combinations(new_block, t))
                    if all(tid in new_ids for tid in needed):
                        for trip in itertools.combinations(b1, t):
                            counts[triplet_to_id[trip]] -= 1
                        blocks[i] = new_block
                        for trip in itertools.combinations(new_block, t):
                            counts[triplet_to_id[trip]] += 1
                        improved = True
                        break

            # 4. Shift move: replace one point in a block
            if not improved:
                n = len(blocks)
                if n == 0:
                    break
                i = random.randint(0, n - 1)
                b = list(blocks[i])
                for pos in range(k):
                    old_pt = b[pos]
                    for new_pt in range(v):
                        if new_pt == old_pt or new_pt in b:
                            continue
                        candidate = b.copy()
                        candidate[pos] = new_pt
                        candidate.sort()
                        candidate = tuple(candidate)
                        if candidate in [tuple(bl) for bl in blocks]:
                            continue
                        old_ids = set(triplet_to_id[trip] for trip in itertools.combinations(b, t))
                        unique_old = [tid for tid in old_ids if counts[tid] == 1]
                        if not unique_old:
                            continue
                        new_ids = set(triplet_to_id[trip] for trip in itertools.combinations(candidate, t))
                        if all(tid in new_ids for tid in unique_old):
                            for trip in itertools.combinations(b, t):
                                counts[triplet_to_id[trip]] -= 1
                            blocks[i] = list(candidate)
                            for trip in itertools.combinations(candidate, t):
                                counts[triplet_to_id[trip]] += 1
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
            else:
                no_improve = 0

        return blocks

    best_blocks = None
    best_len = float('inf')

    # Run multiple restarts
    n_restarts = 12
    for seed in range(n_restarts):
        if time.time() > deadline - 8:
            break

        init_blocks = greedy_set_cover(seed)
        opt_blocks = optimize(init_blocks, deadline - 8)

        # Verify
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in opt_blocks:
            for trip in itertools.combinations(b, t):
                counts[triplet_to_id[trip]] += 1

        if np.min(counts) > 0:
            if len(opt_blocks) < best_len:
                best_len = len(opt_blocks)
                best_blocks = opt_blocks

    if best_blocks is None:
        best_blocks = greedy_set_cover(0)

    return best_blocks

if __name__ == "__main__":
    pass
