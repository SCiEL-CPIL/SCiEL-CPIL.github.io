"""Bitmask-based greedy covering with random restarts and time guard."""
import time
import random
import itertools


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)

    # Precompute bitmask for each t-subset
    t_subset_masks = [0] * num_t_subsets
    for i, s in enumerate(all_t_subsets):
        mask = 0
        for x in s:
            mask |= (1 << x)
        t_subset_masks[i] = mask

    # For each point x, list of t-subset indices that contain x
    t_subsets_containing = [[] for _ in range(v)]
    for i, s in enumerate(all_t_subsets):
        for x in s:
            t_subsets_containing[x].append(i)

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = bytearray([1] * num_t_subsets)  # 1 = uncovered
        blocks = []

        while True:
            if time.time() > deadline:
                return None

            # Find any uncovered t-subset
            uncovered_indices = [i for i in range(num_t_subsets) if uncovered[i]]
            if not uncovered_indices:
                break

            # Pick randomly for diversity
            first_idx = rng.choice(uncovered_indices)
            first_subset = all_t_subsets[first_idx]

            block_mask = 0
            for x in first_subset:
                block_mask |= (1 << x)

            # Greedily add points
            while block_mask.bit_count() < k:
                if time.time() > deadline:
                    return None

                best_gain = -1
                best_points = []

                for x in range(v):
                    if (block_mask >> x) & 1:
                        continue

                    new_mask = block_mask | (1 << x)
                    gain = 0
                    for s_idx in t_subsets_containing[x]:
                        if uncovered[s_idx] and (t_subset_masks[s_idx] & ~new_mask) == 0:
                            gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain and gain > 0:
                        best_points.append(x)

                if not best_points:
                    available = [x for x in range(v) if not ((block_mask >> x) & 1)]
                    if not available:
                        break
                    x = rng.choice(available)
                else:
                    x = rng.choice(best_points)
                block_mask |= (1 << x)

            block = sorted([x for x in range(v) if (block_mask >> x) & 1])
            if len(block) != k:
                continue

            blocks.append(block)

            # Update uncovered
            for i in range(num_t_subsets):
                if uncovered[i] and (t_subset_masks[i] & ~block_mask) == 0:
                    uncovered[i] = 0

        return blocks

    best_blocks = None
    best_count = float('inf')

    # Deterministic first
    result = greedy_cover(0)
    if result is not None:
        best_blocks = result
        best_count = len(result)

    # Random restarts
    seed = 1
    while time.time() < deadline - 3.0:
        result = greedy_cover(seed)
        if result is not None:
            if len(result) < best_count:
                best_blocks = result
                best_count = len(result)
        seed += 1

    if best_blocks is None:
        return []

    return best_blocks
