import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 150s kill

    # Precomputation
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}

    # Generate all k-subsets
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)

    def nCr(n, r):
        if r > n:
            return 0
        if r == 0:
            return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res

    t_count = nCr(k, t)

    # Precompute block_t_indices: for each k-subset, which t-subsets it covers
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]

    best_solution = None
    best_score = float('inf')

    def run_greedy(seed):
        """Run one greedy construction with given random seed."""
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts

        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None

            # Calculate gains: how many new t-subsets each block would cover
            uncovered_flags = ~covered[block_t_indices]
            gains = uncovered_flags.sum(axis=1)

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            # Pick from top candidates randomly to diversify
            if n_ks > 50:
                top_k = min(50, n_ks)
                top_indices = np.argpartition(gains, -top_k)[-top_k:]
            else:
                top_indices = np.arange(n_ks)

            best_idx = top_indices[rng.randint(0, len(top_indices) - 1)]

            blocks_indices.append(best_idx)

            # Update covered
            subs_to_cover = block_t_indices[best_idx]
            new_covers = np.sum(~covered[subs_to_cover])
            remaining -= new_covers
            covered[subs_to_cover] = True

        current_blocks = list(blocks_indices)

        # Local search phase 1: Remove redundant blocks
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                return current_blocks, len(current_blocks)

            improved = False
            new_blocks = []

            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current_blocks:
                coverage_count[block_t_indices[b_idx]] += 1

            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                if np.min(coverage_count[subs_in_block]) > 1:
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)

            current_blocks = new_blocks

        # Local search phase 2: Try swaps to reduce block count
        # For each block, try replacing it with another block that covers
        # the same "critical" t-subsets (those only covered by this block)
        # while potentially covering more.
        for swap_pass in range(3):
            if time.time() - start_time > time_limit - 2.0:
                break

            improved_swap = False
            new_blocks = list(current_blocks)

            for i in range(len(new_blocks)):
                if time.time() - start_time > time_limit - 2.0:
                    break

                b_idx = new_blocks[i]
                subs_in_block = block_t_indices[b_idx]

                # Find t-subsets only covered by this block
                coverage_count = np.zeros(n_ts, dtype=np.int32)
                for b_idx2 in new_blocks:
                    coverage_count[block_t_indices[b_idx2]] += 1

                critical = coverage_count[subs_in_block] == 1
                if not np.any(critical):
                    # This block is redundant, remove it
                    new_blocks.pop(i)
                    improved_swap = True
                    i -= 1
                    continue

                # Try to find a replacement block that covers all critical subsets
                # and has at least as many total covered subsets
                critical_mask = np.zeros(n_ts, dtype=bool)
                critical_mask[subs_in_block[critical]] = True

                # For each candidate block, check if it covers all critical subsets
                # and count how many total subsets it covers
                for cand_idx in range(n_ks):
                    if cand_idx == b_idx:
                        continue
                    cand_subs = block_t_indices[cand_idx]
                    # Check if candidate covers all critical subsets
                    if not np.all(critical_mask[cand_subs]):
                        continue

                    # Count total coverage of candidate
                    cand_coverage = np.sum(coverage_count[cand_subs] == 0)

                    # Count how many of the current block's subsets are not covered by candidate
                    # (these would become uncovered)
                    lost = np.sum((coverage_count[subs_in_block] == 1) & (~critical_mask[subs_in_block]))

                    # Net change in covered count
                    if cand_coverage - lost >= 0:
                        # Accept the swap
                        new_blocks[i] = cand_idx
                        improved_swap = True
                        break

            current_blocks = new_blocks
            if not improved_swap:
                break

        return current_blocks, len(current_blocks)

    # Run multiple random restarts
    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        blocks, score = run_greedy(restart_count)

        if blocks is None:
            break

        if score < best_score:
            best_score = score
            best_solution = blocks

        restart_count += 1

    if best_solution is None:
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]

    return final_blocks
