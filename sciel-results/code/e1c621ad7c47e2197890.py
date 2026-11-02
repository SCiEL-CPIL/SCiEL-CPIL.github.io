import numpy as np
import itertools
import time
import random
from collections import Counter
import bisect

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin

    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}

    # Generate all k-subsets
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)

    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res

    t_count = nCr(k, t)

    # Precompute block_t_indices
    # Shape: (n_ks, t_count)
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]

    best_solution = None
    best_score = float('inf')

    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)

        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts

        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None

            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            # Stochastic Greedy: Pick from top candidates
            # Use a larger pool for better diversity in local search
            if n_ks > 50:
                top_indices = np.argpartition(gains, -50)[-50:]
            else:
                top_indices = np.arange(n_ks)

            best_idx = top_indices[random.randint(0, len(top_indices) - 1)]

            blocks_indices.append(best_idx)

            subs_to_cover = block_t_indices[best_idx]
            prev_state = covered[subs_to_cover]
            new_covers = np.sum(~prev_state)
            remaining -= new_covers
            covered[subs_to_cover] = True

        current_blocks = list(blocks_indices)

        if not current_blocks:
            return current_blocks, 0

        # Local Search Phase
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1

        improved = True
        # Limit local search iterations to ensure we finish in time
        max_ls_iters = 200
        ls_iter = 0

        while improved and ls_iter < max_ls_iters and (time.time() - start_time < time_limit - 2.0):
            improved = False
            ls_iter += 1
            new_blocks = []

            # Pass 1: Deletion
            # Shuffle to break symmetries
            random.shuffle(current_blocks)
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                if np.min(coverage_count[subs_in_block]) > 1:
                    coverage_count[subs_in_block] -= 1
                    improved = True
                else:
                    new_blocks.append(b_idx)

            current_blocks = new_blocks

            # If no deletion happened, try swaps to escape local minima
            if not improved and len(current_blocks) > 0:
                swaps_tried = 0
                max_swaps = 100
                swap_applied = False

                while swaps_tried < max_swaps and not swap_applied and (time.time() - start_time < time_limit - 2.0):
                    swaps_tried += 1
                    b_pos = random.randint(0, len(current_blocks) - 1)
                    b_idx = current_blocks[b_pos]
                    block = all_ks[b_idx]
                    block_set = set(block)

                    x = random.choice(block)
                    y = random.randint(0, v - 1)
                    if y in block_set:
                        continue

                    new_block_list = [p for p in block if p != x]
                    new_block_list.append(y)
                    new_block_list.sort()
                    new_block_tuple = tuple(new_block_list)

                    # Check if this block exists in our precomputed list
                    pos = bisect.bisect_left(all_ks, new_block_tuple)
                    if pos < n_ks and all_ks[pos] == new_block_tuple:
                        new_b_idx = pos

                        # Calculate gain/loss efficiently
                        old_triples = block_t_indices[b_idx]
                        new_triples = block_t_indices[new_b_idx]

                        # We need to check if removing old and adding new maintains coverage.
                        # Lost: in old, not in new, and coverage == 1
                        # Gained: in new, not in old (coverage increases, always safe)

                        # Since sets are small (35), use sets for difference
                        old_set = set(old_triples)
                        new_set = set(new_triples)

                        lost_indices = list(old_set - new_set)
                        # gained_indices = list(new_set - old_set) # Not strictly needed for validity check, but for update

                        valid = True
                        for idx in lost_indices:
                            if coverage_count[idx] <= 1:
                                valid = False
                                break

                        if valid:
                            # Apply swap
                            for idx in lost_indices:
                                coverage_count[idx] -= 1
                            for idx in new_set:
                                if idx not in old_set:
                                    coverage_count[idx] += 1

                            current_blocks[b_pos] = new_b_idx
                            swap_applied = True
                            improved = True # Trigger deletion pass again

        return current_blocks, len(current_blocks)

    restart_count = 0
    # Run multiple restarts to find a better solution
    # Allocate time for restarts
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks, score = run_greedy(seed)

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
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]

    return final_blocks
