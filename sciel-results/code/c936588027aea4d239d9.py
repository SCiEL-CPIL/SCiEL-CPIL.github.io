"""Covering design C(19,8,5) via Lagrangian-relaxation guided greedy with time guard."""
import itertools
import time
import random
import math


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # Precompute all t-subsets as frozensets for fast membership
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = set(all_t_subsets)
    num_t = len(all_t_subsets)

    # Map each t-subset to an index for fast lookups
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # For each point, precompute which t-subsets it belongs to (as indices)
    # points_to_tsubs[point] = set of indices of t-subsets containing point
    points_to_tsubs = [set() for _ in range(v)]
    for i, ts in enumerate(all_t_subsets):
        for p in ts:
            points_to_tsubs[p].add(i)

    # For a block (set of points), the t-subsets it covers are those t-subsets
    # fully contained in the block. We'll compute coverage incrementally.

    uncovered = set(range(num_t))  # indices of uncovered t-subsets
    blocks = []

    # Lagrangian weights: penalty for each uncovered t-subset
    # Initially uniform; updated via subgradient method
    weights = [1.0] * num_t

    def block_gain(block_points):
        """Compute the number of newly covered t-subsets if we add point x to block."""
        # For each candidate point x not in block, count how many uncovered t-subsets
        # would be completed by adding x.
        # A t-subset is completed by adding x if it contains x and all other t-1 points are in block.
        best_x = -1
        best_gain = -1
        block_set = set(block_points)
        for x in range(v):
            if x in block_set:
                continue
            # Count uncovered t-subsets that contain x and have all other t-1 points in block
            gain = 0
            # A t-subset containing x and t-1 points from block:
            # Iterate over combinations of t-1 points from block
            if len(block_points) >= t - 1:
                for combo in itertools.combinations(block_points, t - 1):
                    ts = tuple(sorted(combo + (x,)))
                    idx = t_subset_index.get(ts, -1)
                    if idx != -1 and idx in uncovered:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        return best_x, best_gain

    def build_block_with_weights():
        """Build a block greedily, using Lagrangian weights to break ties and guide selection."""
        # Start with a random uncovered t-subset's points to seed the block
        if not uncovered:
            return None

        # Pick a random uncovered t-subset to start
        seed_idx = random.choice(list(uncovered))
        seed = list(all_t_subsets[seed_idx])

        # Build block starting from seed
        block = list(seed)
        block_set = set(block)

        while len(block) < k:
            best_x = -1
            best_score = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Compute weighted gain: sum of weights of uncovered t-subsets completed
                score = 0.0
                if len(block) >= t - 1:
                    for combo in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(combo + (x,)))
                        idx = t_subset_index.get(ts, -1)
                        if idx != -1 and idx in uncovered:
                            score += weights[idx]
                if score > best_score:
                    best_score = score
                    best_x = x
            if best_x == -1 or best_score <= 0:
                # No positive gain; just add a random unused point
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    best_x = random.choice(remaining)
                else:
                    break
            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        return block

    def coverage_of_block(block):
        """Return set of t-subset indices covered by this block."""
        covered = set()
        for ts in itertools.combinations(block, t):
            idx = t_subset_index.get(ts, -1)
            if idx != -1:
                covered.add(idx)
        return covered

    def subgradient_update(blocks_list):
        """Update Lagrangian weights based on current uncovered set."""
        # Subgradient: for each t-subset, if uncovered, increase weight; if covered, decrease
        # Use a simple multiplicative update
        for i in range(num_t):
            if i in uncovered:
                weights[i] = min(weights[i] * 1.1, 100.0)
            else:
                weights[i] = max(weights[i] * 0.95, 0.1)

    # Main loop: iteratively add blocks
    num_iterations = 0
    while uncovered and time.time() < deadline:
        num_iterations += 1

        # Build a block
        block = build_block_with_weights()
        if block is None:
            break

        # Compute coverage
        covered = coverage_of_block(block)
        new_covered = covered & uncovered

        if not new_covered:
            # This block doesn't help; skip it
            continue

        blocks.append(block)
        uncovered -= new_covered

        # Periodically update weights
        if num_iterations % 10 == 0:
            subgradient_update(blocks)

        # Check time
        if time.time() > deadline - 5.0:
            break

    # Final check: if we still have uncovered subsets and time remains,
    # do a simple greedy pass to try to cover them
    if uncovered and time.time() < deadline - 2.0:
        while uncovered and time.time() < deadline - 1.0:
            # Find the t-subset with the fewest points already in some block
            # Just do simple greedy: pick first uncovered, build block around it
            seed_idx = next(iter(uncovered))
            seed = list(all_t_subsets[seed_idx])
            block = list(seed)
            block_set = set(block)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    if len(block) >= t - 1:
                        for combo in itertools.combinations(block, t - 1):
                            ts = tuple(sorted(combo + (x,)))
                            idx = t_subset_index.get(ts, -1)
                            if idx != -1 and idx in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_x = random.choice(remaining)
                    else:
                        break
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            if block not in blocks:
                blocks.append(block)
                covered = coverage_of_block(block)
                uncovered -= covered

    return blocks
