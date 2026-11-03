import itertools
import time
import random
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Bitset representation
    tsub_mask = [0] * n_blocks
    tsub_blocks_mask = [0] * n_tsubsets
    block_tsubs = [[] for _ in range(n_blocks)]

    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks_mask[idx] |= (1 << b_idx)
            block_tsubs[b_idx].append(idx)
        tsub_mask[b_idx] = mask

    full_tsub_mask = (1 << n_tsubsets) - 1

    # Precompute point frequencies in each block for frequency-biased greedy
    # point_in_blocks[p] = set of block indices containing point p
    point_in_blocks = [set() for _ in range(v)]
    for b_idx, block in enumerate(all_blocks):
        for p in block:
            point_in_blocks[p].add(b_idx)

    def greedy_cover(seed):
        """
        Frequency-biased greedy: maximize newly-covered t-subsets,
        with tie-break favoring blocks that contain under-covered points.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        
        # Track point coverage counts for tie-breaking
        point_cover_count = [0] * v

        while uncovered_mask:
            if time.time() > deadline - 5.0:
                break

            best_block = -1
            best_gain = 0
            best_tiebreak = 0

            for b_idx in range(n_blocks):
                if not (tsub_mask[b_idx] & uncovered_mask):
                    continue
                gain = (tsub_mask[b_idx] & uncovered_mask).bit_count()
                
                # Tie-break: sum of inverse point coverage counts (favor under-covered points)
                tiebreak = 0
                for p in all_blocks[b_idx]:
                    tiebreak += 1.0 / (point_cover_count[p] + 1)
                
                if gain > best_gain or (gain == best_gain and tiebreak > best_tiebreak):
                    best_gain = gain
                    best_block = b_idx
                    best_tiebreak = tiebreak
                    if best_gain == n_tsubsets:
                        break

            if best_block == -1:
                break

            selected.append(best_block)
            uncovered_mask &= ~tsub_mask[best_block]
            for p in all_blocks[best_block]:
                point_cover_count[p] += 1

        return selected

    def local_search(blocks, seed, max_passes=5):
        """
        Local search with swaps and deletions, multiple passes.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks

        block_set = set(blocks)

        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline - 2.0:
            improved = False
            passes += 1

            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in block_set:
                redundant = True
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)

            if to_remove:
                for b in to_remove:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue

            # Phase 2: Try swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for b in current_blocks:
                if time.time() > deadline - 2.0:
                    break
                if b not in block_set:
                    continue

                unique_tsubs = []
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                cand_mask = (1 << n_blocks) - 1
                for ts_idx in unique_tsubs:
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break

                if cand_mask == 0:
                    continue

                cand_mask &= ~(1 << b)

                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    candidates.append(idx)
                    m ^= lsb
                    if len(candidates) > 30:
                        break

                if not candidates:
                    continue

                rng.shuffle(candidates)

                swapped = False
                for r in candidates:
                    if r in block_set:
                        continue

                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1

                    block_set.add(r)
                    for ts_idx in block_tsubs[r]:
                        cover_count[ts_idx] += 1

                    improved = True
                    swapped = True
                    break

                if swapped:
                    break

        return list(block_set)

    def two_for_one_search(blocks, seed):
        """
        Try to replace two blocks with one.
        """
        rng = random.Random(seed)
        if len(blocks) < 2:
            return blocks

        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            n_checks = 150
            for _ in range(n_checks):
                if time.time() > deadline - 2.0:
                    break
                if len(block_set) < 2:
                    break

                i, j = rng.sample(range(len(current_blocks)), 2)
                b1 = current_blocks[i]
                b2 = current_blocks[j]

                if b1 not in block_set or b2 not in block_set:
                    continue

                required_mask = 0
                for ts_idx in block_tsubs[b1]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)
                for ts_idx in block_tsubs[b2]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)

                if required_mask == 0:
                    continue

                cand_mask = (1 << n_blocks) - 1
                m = required_mask
                while m:
                    lsb = m & (-m)
                    ts_idx = lsb.bit_length() - 1
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break
                    m ^= lsb

                if cand_mask == 0:
                    continue

                cand_mask &= ~(1 << b1)
                cand_mask &= ~(1 << b2)

                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    candidates.append(idx)
                    m ^= lsb
                    if len(candidates) > 10:
                        break

                if not candidates:
                    continue

                rng.shuffle(candidates)
                r = candidates[0]

                block_set.remove(b1)
                for ts_idx in block_tsubs[b1]:
                    cover_count[ts_idx] -= 1

                block_set.remove(b2)
                for ts_idx in block_tsubs[b2]:
                    cover_count[ts_idx] -= 1

                block_set.add(r)
                for ts_idx in block_tsubs[r]:
                    cover_count[ts_idx] += 1

                improved = True
                break

        return list(block_set)

    def final_polish(blocks, seed):
        """
        High-precision polish: aggressive local search with more iterations
        and tighter candidate selection.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks

        block_set = set(blocks)

        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False

            # Remove redundant
            to_remove = []
            for b in block_set:
                redundant = True
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)

            if to_remove:
                for b in to_remove:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue

            # Try swaps with more candidates
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for b in current_blocks:
                if time.time() > deadline - 1.0:
                    break
                if b not in block_set:
                    continue

                unique_tsubs = []
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                cand_mask = (1 << n_blocks) - 1
                for ts_idx in unique_tsubs:
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break

                if cand_mask == 0:
                    continue

                cand_mask &= ~(1 << b)

                # Score candidates by how many additional t-subsets they cover
                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    extra = 0
                    for ts_idx in block_tsubs[idx]:
                        if cover_count[ts_idx] == 0:
                            extra += 1
                    candidates.append((extra, idx))
                    m ^= lsb
                    if len(candidates) > 50:
                        break

                if not candidates:
                    continue

                candidates.sort(reverse=True)
                top_candidates = [c[1] for c in candidates[:10]]
                rng.shuffle(top_candidates)

                swapped = False
                for r in top_candidates:
                    if r in block_set:
                        continue

                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1

                    block_set.add(r)
                    for ts_idx in block_tsubs[r]:
                        cover_count[ts_idx] += 1

                    improved = True
                    swapped = True
                    break

                if swapped:
                    break

        return list(block_set)

    best_solution = None
    best_size = float('inf')

    # Run multiple restarts with frequency-biased greedy
    num_restarts = 15
    for i in range(num_restarts):
        if time.time() > deadline - 20.0:
            break

        seed = 1000 + i
        greedy_sol = greedy_cover(seed)

        if not greedy_sol:
            continue

        # Local Search
        ls_sol = local_search(greedy_sol, seed, max_passes=6)

        # 2-for-1 Search
        t2o_sol = two_for_one_search(ls_sol, seed)

        # Final polish
        if time.time() < deadline - 10.0:
            polished_sol = final_polish(t2o_sol, seed + 10000)
            if len(polished_sol) < len(t2o_sol):
                t2o_sol = polished_sol

        if len(t2o_sol) < best_size:
            best_size = len(t2o_sol)
            best_solution = t2o_sol.copy()

    # Final pass: try more aggressive local search on the best solution
    if best_solution is not None and time.time() < deadline - 5.0:
        final_ls = local_search(best_solution, 9999, max_passes=8)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_solution = final_ls

        # Another polish
        if time.time() < deadline - 3.0:
            final_polished = final_polish(best_solution, 12345)
            if len(final_polished) < best_size:
                best_size = len(final_polished)
                best_solution = final_polished

    # If no solution found (should not happen), fallback to basic greedy
    if best_solution is None:
        best_solution = greedy_cover(0)

    return [list(all_blocks[idx]) for idx in best_solution]
