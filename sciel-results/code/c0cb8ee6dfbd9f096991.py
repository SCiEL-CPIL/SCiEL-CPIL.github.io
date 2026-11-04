import itertools
import time
import random
import numpy as np
from collections import defaultdict, deque

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin for 150s limit

    # Precompute all t-subsets and blocks
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Precompute which t-subset indices each block covers
    block_covered_ts_indices = []
    for blk in all_blocks:
        covered_indices = []
        for ts in itertools.combinations(blk, t):
            idx = tsubset_to_idx[ts]
            covered_indices.append(idx)
        block_covered_ts_indices.append(covered_indices)

    # Precompute ts_to_blocks
    ts_to_blocks = [[] for _ in range(n_tsubsets)]
    for b_idx, ts_list in enumerate(block_covered_ts_indices):
        for ts_idx in ts_list:
            ts_to_blocks[ts_idx].append(b_idx)

    # Convert to numpy arrays for vectorized operations
    # block_covered_ts_indices is a list of lists. Let's make it a list of np arrays.
    block_covered_ts_np = [np.array(lst, dtype=np.int32) for lst in block_covered_ts_indices]

    def run_greedy_reduce(seed_val):
        random.seed(seed_val)
        np.random.seed(seed_val)

        # Greedy Phase
        counts = np.zeros(n_tsubsets, dtype=np.int8)
        selected_block_indices = []
        remaining = np.ones(n_tsubsets, dtype=bool)
        n_remaining = n_tsubsets

        # gains[b_idx] = number of remaining ts covered by b_idx
        gains = np.array([len(lst) for lst in block_covered_ts_indices], dtype=np.int32)

        # Precompute a set of selected blocks for fast lookup?
        # We need to check if a block is already selected.
        # Since n_blocks is ~75k, a boolean array is efficient.
        selected_mask = np.zeros(n_blocks, dtype=bool)

        while n_remaining > 0:
            if time.time() > deadline:
                break

            # Vectorized gain calculation is not strictly necessary if we update gains incrementally.
            # But let's stick to the incremental update which is faster per step.
            # We need to find the block with max gain.
            # However, gains might be stale? No, we update them.
            # But we must ensure we don't pick a block that is already selected.
            # We can set gains[selected] = -1 or similar.

            # Find max gain among unselected blocks
            # To avoid picking selected blocks, we can temporarily mask them.
            # Or just check after selection.
            # Better: maintain gains correctly and use a mask.

            # Actually, the "remaining" mask is the key.
            # gain[b] = sum(remaining[block_covered_ts[b]])
            # If we update gains incrementally, we just need to pick max.

            # Let's pick the block with the maximum gain.
            # To add randomness/tie-breaking, we can add small noise.
            # noise = np.random.uniform(0, 1, n_blocks) * 0.1
            # scores = gains.copy()
            # scores[selected_mask] = -1
            # best_idx = np.argmax(scores)

            # Simpler: np.argmax(gains) might pick a selected block if its gain wasn't set to 0.
            # So we must set gains[selected_block] = 0 (or -1) when selected.

            best_block_idx = np.argmax(gains)
            best_gain = gains[best_block_idx]

            if best_gain == 0:
                break

            selected_block_indices.append(best_block_idx)
            selected_mask[best_block_idx] = True
            gains[best_block_idx] = 0  # Mark as used

            # Update counts and remaining/gains
            ts_indices = block_covered_ts_np[best_block_idx]
            # Update counts
            counts[ts_indices] += 1

            # Update remaining
            newly_covered = ts_indices[remaining[ts_indices]]
            n_newly_covered = len(newly_covered)
            if n_newly_covered > 0:
                remaining[newly_covered] = False
                n_remaining -= n_newly_covered

                # Update gains for all blocks covering these newly covered ts
                # For each newly covered ts, iterate over blocks covering it and decrement gain
                for ts_idx in newly_covered:
                    for b_idx in ts_to_blocks[ts_idx]:
                        if not selected_mask[b_idx]:
                            gains[b_idx] -= 1

        # Reduction Phase
        # A block is redundant if all ts it covers have count > 1
        # We can do this iteratively.
        # To speed up, we can check in reverse order (last added are most likely redundant? Not necessarily).
        # Let's just iterate through the list.

        changed = True
        while changed:
            changed = False
            for i in range(len(selected_block_indices) - 1, -1, -1):
                if time.time() > deadline:
                    return selected_block_indices

                b_idx = selected_block_indices[i]
                is_redundant = True
                ts_indices = block_covered_ts_np[b_idx]
                # Check if any ts in this block has count == 1
                if np.any(counts[ts_indices] == 1):
                    is_redundant = False

                if is_redundant:
                    # Remove block
                    selected_block_indices.pop(i)
                    selected_mask[b_idx] = False
                    for ts_idx in ts_indices:
                        counts[ts_idx] -= 1
                    changed = True

        return selected_block_indices

    def local_search_swap(start_time, time_limit):
        # This function takes a solution and tries to improve it via swaps
        # We will integrate this into the main loop or call it after greedy.
        # For now, let's stick to the multi-restart greedy + reduction.
        # The inspiration suggests swaps can help.
        # Let's implement a simple swap pass.

        pass

    best_solution = None
    best_len = float('inf')

    # Multiple restarts
    # Vectorized greedy is fast.
    # Each step: argmax (O(N_blocks)), update gains (O(sum of degrees of newly covered ts)).
    # Sum of degrees is roughly n_remaining * avg_degree.
    # Should be much faster than Python loops.
    # Let's allow more restarts.

    num_restarts = 200  # Try more restarts

    for i in range(num_restarts):
        if time.time() > deadline:
            break

        # Use different seeds for diversity
        # We can also vary the greedy strategy slightly.
        # E.g., random tie-breaking.
        sol = run_greedy_reduce(i * 12345 + 7)

        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol[:]

    # Convert block indices back to lists of points
    final_blocks = []
    if best_solution:
        for b_idx in best_solution:
            final_blocks.append(list(all_blocks[b_idx]))
    else:
        # Fallback
        sol = run_greedy_reduce(0)
        for b_idx in sol:
            final_blocks.append(list(all_blocks[b_idx]))

    return final_blocks
