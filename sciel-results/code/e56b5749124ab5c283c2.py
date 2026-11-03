"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Pre-compute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)

    # Map each t-subset to an index for fast lookup
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # We'll represent a block as a frozenset of points
    # For each block, we want to know which t-subsets it covers

    # Helper: get all t-subsets covered by a given block (list of points)
    def block_covers(block_points):
        """Return set of t-subset indices covered by this block."""
        covered = set()
        for ts in itertools.combinations(block_points, t):
            covered.add(tsubset_to_idx[ts])
        return covered

    # Greedy construction with pre-computed coverage
    def greedy_construct():
        uncovered = set(range(num_tsubsets))
        blocks = []

        while uncovered:
            # Pick the uncovered t-subset with the most "potential"
            # Actually, pick any uncovered one and build a block around it
            # Better: pick the uncovered t-subset, then greedily add points
            # that cover the most additional uncovered t-subsets

            # Find an uncovered t-subset
            # To make it deterministic-ish but allow variety, we can rotate
            # For now, just pick the first uncovered
            first_uncovered = min(uncovered)
            initial_points = list(all_tsubsets[first_uncovered])

            block_points = list(initial_points)
            # Track which t-subsets are covered by the current block
            block_covered = block_covers(block_points)
            # Only count uncovered ones
            effective_covered = block_covered & uncovered

            while len(block_points) < k:
                # Find the point that covers the most additional uncovered t-subsets
                best_point = None
                best_gain = -1

                for x in range(v):
                    if x in block_points:
                        continue
                    # What t-subsets would be newly covered by adding x?
                    # These are t-subsets containing x and t-1 points from block_points
                    new_covered = 0
                    for rest in itertools.combinations(block_points, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsubset_to_idx.get(ts)
                        if idx is not None and idx in uncovered:
                            new_covered += 1

                    if new_covered > best_gain:
                        best_gain = new_covered
                        best_point = x

                if best_point is None:
                    break

                block_points.append(best_point)

            # Sort for canonical form
            block_points.sort()
            blocks.append(block_points)

            # Update uncovered
            new_covered = block_covers(block_points)
            uncovered -= new_covered

        return blocks

    # Local search: try to remove a block if all its t-subsets are covered by other blocks
    def try_remove_block(blocks):
        """Try to remove one block at a time if redundant."""
        changed = True
        while changed:
            changed = False
            # Compute coverage counts
            coverage_count = [0] * num_tsubsets
            for block in blocks:
                covered = block_covers(block)
                for idx in covered:
                    coverage_count[idx] += 1

            # Try to remove each block
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    return blocks

                block = blocks[i]
                covered = block_covers(block)
                # Check if all t-subsets in this block are covered by others
                redundant = True
                for idx in covered:
                    if coverage_count[idx] <= 1:
                        redundant = False
                        break

                if redundant:
                    # Remove this block
                    for idx in covered:
                        coverage_count[idx] -= 1
                    blocks.pop(i)
                    changed = True

        return blocks

    # Local search: swap a point in a block to improve coverage
    def local_search(blocks, max_iters=1000):
        """Try to improve the covering by swapping points in blocks."""
        # Pre-compute coverage for each block
        block_covered_list = [block_covers(b) for b in blocks]
        coverage_count = [0] * num_tsubsets
        for covered in block_covered_list:
            for idx in covered:
                coverage_count[idx] += 1

        best_blocks = [list(b) for b in blocks]
        best_len = len(blocks)

        for _ in range(max_iters):
            if time.time() > deadline:
                break

            # Pick a random block and a random point in it
            bi = random.randint(0, len(blocks) - 1)
            block = blocks[bi]
            if len(block) == 0:
                continue
            pi = random.randint(0, len(block) - 1)
            old_point = block[pi]

            # Try replacing with another point
            new_point = random.choice([x for x in range(v) if x not in block])
            if new_point == old_point:
                continue

            # Compute the effect of the swap
            old_covered = block_covered_list[bi]

            # New block
            new_block = list(block)
            new_block[pi] = new_point
            new_block.sort()
            new_covered = block_covers(new_block)

            # Check if swap maintains validity
            # A t-subset becomes uncovered if it was only covered by this block
            # and is no longer covered
            lost = old_covered - new_covered
            gained = new_covered - old_covered

            valid = True
            for idx in lost:
                if coverage_count[idx] <= 1:
                    valid = False
                    break

            if valid:
                # Apply the swap
                blocks[bi] = new_block
                block_covered_list[bi] = new_covered
                for idx in lost:
                    coverage_count[idx] -= 1
                for idx in gained:
                    coverage_count[idx] += 1

                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = [list(b) for b in blocks]

        return best_blocks

    # Main algorithm
    # 1. Greedy construction
    blocks = greedy_construct()
    if time.time() > deadline:
        return blocks

    # 2. Remove redundant blocks
    blocks = try_remove_block(blocks)
    if time.time() > deadline:
        return blocks

    # 3. Local search with random restarts
    best_blocks = [list(b) for b in blocks]
    best_len = len(blocks)

    # Do a few rounds of local search
    num_restarts = 0
    while time.time() < deadline - 5.0:
        # Try local search on current best
        improved = local_search([list(b) for b in best_blocks], max_iters=200)
        if len(improved) < best_len:
            best_len = len(improved)
            best_blocks = [list(b) for b in improved]
            num_restarts += 1

        # Try a fresh greedy construction with randomization
        if num_restarts % 3 == 0:
            # Randomized greedy
            uncovered = set(range(num_tsubsets))
            new_blocks = []

            while uncovered and time.time() < deadline - 5.0:
                # Pick a random uncovered t-subset
                first_uncovered = random.choice(list(uncovered))
                initial_points = list(all_tsubsets[first_uncovered])

                block_points = list(initial_points)

                while len(block_points) < k:
                    best_point = None
                    best_gain = -1

                    for x in range(v):
                        if x in block_points:
                            continue
                        new_covered = 0
                        for rest in itertools.combinations(block_points, t - 1):
                            ts = tuple(sorted(rest + (x,)))
                            idx = tsubset_to_idx.get(ts)
                            if idx is not None and idx in uncovered:
                                new_covered += 1

                        if new_covered > best_gain:
                            best_gain = new_covered
                            best_point = x

                    if best_point is None:
                        break

                    block_points.append(best_point)

                block_points.sort()
                new_blocks.append(block_points)

                new_covered = block_covers(block_points)
                uncovered -= new_covered

            if uncovered:
                # Incomplete, skip
                continue

            new_blocks = try_remove_block(new_blocks)
            if len(new_blocks) < best_len:
                best_len = len(new_blocks)
                best_blocks = [list(b) for b in new_blocks]

    return best_blocks
# EVOLVE-BLOCK-END
