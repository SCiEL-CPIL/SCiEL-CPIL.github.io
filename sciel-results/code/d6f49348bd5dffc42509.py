import itertools
import random
import time
import sys

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using:
    1. Multi-seed greedy initialization (fast, anchored on uncovered t-sets)
    2. Aggressive local search: redundant-block removal + critical-block swaps
    3. Final polish: exhaustive redundant-block sweep
    """
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin for 120s limit

    # Precomputation
    # 1. Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_set_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_tsets = len(t_subsets)

    # 2. Generate all k-subsets (blocks) and map to indices
    k_subsets = list(itertools.combinations(range(v), k))
    num_ksets = len(k_subsets)

    # 3. Map each block to the list of t-set indices it covers
    block_to_tsets = [[] for _ in range(num_ksets)]
    # 4. Map each t-set to the list of block indices that cover it
    tset_to_blocks = [[] for _ in range(num_tsets)]

    for b_idx, block in enumerate(k_subsets):
        for ts in itertools.combinations(block, t):
            ts_idx = t_set_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)

    def greedy_init(seed):
        """Run greedy initialization with a given random seed."""
        random.seed(seed)
        current_blocks = []
        tset_counts = [0] * num_tsets
        in_current = [False] * num_ksets
        uncovered = set(range(num_tsets))

        while uncovered:
            if time.time() > deadline:
                break
            # Pick a random uncovered t-set to anchor the search
            anchor_ts = random.choice(list(uncovered))
            candidates = tset_to_blocks[anchor_ts]

            best_b = -1
            max_gain = 0
            for b_idx in candidates:
                if in_current[b_idx]:
                    continue
                gain = 0
                for ts_idx in block_to_tsets[b_idx]:
                    if ts_idx in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_b = b_idx

            if best_b == -1:
                break

            # Add the block
            current_blocks.append(best_b)
            in_current[best_b] = True
            for ts_idx in block_to_tsets[best_b]:
                tset_counts[ts_idx] += 1
                if ts_idx in uncovered:
                    uncovered.discard(ts_idx)

        return current_blocks, tset_counts, in_current

    def local_search(current_blocks, tset_counts, in_current, time_budget):
        """Run local search to minimize the number of blocks."""
        while time.time() < deadline - time_budget:
            # Phase 1: Try to remove redundant blocks
            removed = False
            random.shuffle(current_blocks)

            for b_idx in list(current_blocks):
                if time.time() > deadline - time_budget:
                    break

                # Check if any t-set in this block has count 1
                is_redundant = True
                for ts_idx in block_to_tsets[b_idx]:
                    if tset_counts[ts_idx] == 1:
                        is_redundant = False
                        break

                if is_redundant:
                    # Remove the block
                    current_blocks.remove(b_idx)
                    in_current[b_idx] = False
                    for ts_idx in block_to_tsets[b_idx]:
                        tset_counts[ts_idx] -= 1
                    removed = True
                    break  # Restart loop as state changed

            if removed:
                continue

            # Phase 2: Try to swap a critical block
            # Find all critical blocks (blocks with at least one unique t-set)
            critical_blocks = []
            for b_idx in current_blocks:
                is_critical = False
                for ts_idx in block_to_tsets[b_idx]:
                    if tset_counts[ts_idx] == 1:
                        is_critical = True
                        break
                if is_critical:
                    critical_blocks.append(b_idx)

            if not critical_blocks:
                # No critical blocks found, which means all t-sets have count >= 2
                # This shouldn't happen if we've removed all redundant blocks
                break

            # Pick a random critical block
            old_b = random.choice(critical_blocks)

            # Find unique t-sets of old_b
            unique_ts = []
            for ts_idx in block_to_tsets[old_b]:
                if tset_counts[ts_idx] == 1:
                    unique_ts.append(ts_idx)

            if not unique_ts:
                continue

            # Find candidate new blocks that cover all unique_ts
            # Intersection of tset_to_blocks[ts] for ts in unique_ts
            candidates = set(tset_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(tset_to_blocks[ts_idx])
                if not candidates:
                    break

            # Filter out blocks already in current
            candidates.difference_update(current_blocks)

            if candidates:
                # Pick a random candidate
                new_b = random.choice(list(candidates))

                # Perform swap
                current_blocks.remove(old_b)
                in_current[old_b] = False
                for ts_idx in block_to_tsets[old_b]:
                    tset_counts[ts_idx] -= 1

                current_blocks.append(new_b)
                in_current[new_b] = True
                for ts_idx in block_to_tsets[new_b]:
                    tset_counts[ts_idx] += 1
            else:
                # No swap found for this block, continue to next iteration
                pass

        return current_blocks, tset_counts, in_current

    # Run multiple greedy initializations with different seeds
    # and keep the best result after local search
    best_blocks = None
    best_count = float('inf')

    # Number of restarts based on available time
    # Each greedy + local search takes ~1-2 seconds, so we can do ~50-100 restarts
    num_restarts = 0
    base_seed = 42

    for i in range(200):  # Upper bound on restarts
        if time.time() > deadline - 5.0:
            break

        seed = base_seed + i * 7919  # Large prime step for good mixing
        current_blocks, tset_counts, in_current = greedy_init(seed)

        # Check if greedy produced a valid covering
        # (it should, by construction, but verify)
        valid = True
        for ts_idx in range(num_tsets):
            if tset_counts[ts_idx] < 1:
                valid = False
                break

        if not valid:
            continue

        # Run local search with a time budget
        # Leave some time for final polish
        time_budget = 2.0
        current_blocks, tset_counts, in_current = local_search(
            current_blocks, tset_counts, in_current, time_budget
        )

        count = len(current_blocks)
        if count < best_count:
            best_count = count
            best_blocks = list(current_blocks)
            # Update best
            best_tset_counts = tset_counts
            best_in_current = in_current

        num_restarts += 1

    # Final polish: exhaustive redundant-block sweep on the best solution
    if best_blocks is not None:
        # Make a copy
        final_blocks = list(best_blocks)
        final_tset_counts = [0] * num_tsets
        final_in_current = [False] * num_ksets

        for b_idx in final_blocks:
            final_in_current[b_idx] = True
            for ts_idx in block_to_tsets[b_idx]:
                final_tset_counts[ts_idx] += 1

        # Exhaustive redundant-block removal
        for _ in range(50):  # Limit iterations
            if time.time() > deadline - 1.0:
                break
            removed = False
            for b_idx in list(final_blocks):
                if time.time() > deadline - 1.0:
                    break

                is_redundant = True
                for ts_idx in block_to_tsets[b_idx]:
                    if final_tset_counts[ts_idx] == 1:
                        is_redundant = False
                        break

                if is_redundant:
                    final_blocks.remove(b_idx)
                    final_in_current[b_idx] = False
                    for ts_idx in block_to_tsets[b_idx]:
                        final_tset_counts[ts_idx] -= 1
                    removed = True
                    break

            if not removed:
                break

        # Use the polished result if it's better
        if len(final_blocks) <= len(best_blocks):
            best_blocks = final_blocks

    # Convert block indices back to lists of points
    if best_blocks is None:
        # Fallback: should not happen, but return an empty list if something went wrong
        return []

    result = [list(k_subsets[b_idx]) for b_idx in best_blocks]
    return result

if __name__ == "__main__":
    # Test with the specific task parameters
    v, k, t = 19, 7, 4
    blocks = construct(v, k, t)
    print(f"Number of blocks: {len(blocks)}")
