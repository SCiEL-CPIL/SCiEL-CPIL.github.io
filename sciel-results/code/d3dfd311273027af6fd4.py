"""Variable-neighborhood search over block sets for C(19,8,5) covering."""
import itertools
import time
import random
import sys

# EVOLVE-BLOCK-START
import itertools
import time
import random

_DEADLINE = None
_DEADLINE_MARGIN = 5.0  # seconds before the 150s kill

def _now():
    return time.time()

def _time_left():
    if _DEADLINE is None:
        return float('inf')
    return _DEADLINE - _now() - _DEADLINE_MARGIN

def construct(v, k, t):
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = _now() + 150.0

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)

    # Map each t-subset to an index
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    # For each point, precompute which t-subsets contain it
    # point_to_ts: list of lists of t-subset indices
    point_to_ts = [[] for _ in range(v)]
    for ts in all_t_subsets:
        idx = t_subset_to_idx[ts]
        for p in ts:
            point_to_ts[p].append(idx)

    # For each block (k-subset), precompute which t-subsets it covers
    # We'll compute on-the-fly but cache for speed
    # Actually, for v=19,k=8,t=5: C(8,5)=56 t-subsets per block

    # Greedy initial solution
    def greedy_cover():
        uncovered = set(range(num_t))  # indices of uncovered t-subsets
        blocks = []
        while uncovered:
            # Pick the t-subset with fewest blocks covering it (or just first)
            # For simplicity, pick the one that appears in fewest potential blocks
            # Actually, just pick the first uncovered one
            ts_idx = min(uncovered)
            ts = all_t_subsets[ts_idx]
            block = list(ts)

            while len(block) < k:
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block:
                        continue
                    # Count how many uncovered t-subsets this point would help cover
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_ts = tuple(sorted(rest + (x,)))
                        if new_ts in t_subset_to_idx:
                            idx = t_subset_to_idx[new_ts]
                            if idx in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    break
                block.append(best_point)

            block.sort()
            blocks.append(tuple(block))

            # Remove covered t-subsets
            for ts2 in itertools.combinations(block, t):
                idx = t_subset_to_idx[tuple(ts2)]
                uncovered.discard(idx)

        return list(blocks)

    # Compute coverage count for each t-subset given a list of blocks
    def compute_coverage(blocks):
        coverage = [0] * num_t
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = t_subset_to_idx[tuple(ts)]
                coverage[idx] += 1
        return coverage

    def is_valid(blocks):
        coverage = compute_coverage(blocks)
        return all(c > 0 for c in coverage)

    def count_uncovered(blocks):
        coverage = compute_coverage(blocks)
        return sum(1 for c in coverage if c == 0)

    # Generate candidate blocks: all k-subsets
    # For v=19, k=8: C(19,8) = 75582 - manageable
    all_blocks = list(itertools.combinations(range(v), k))
    all_blocks = [tuple(b) for b in all_blocks]
    num_blocks_total = len(all_blocks)

    # For each block, precompute the set of t-subset indices it covers
    block_covers = []
    for block in all_blocks:
        covers = set()
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[tuple(ts)]
            covers.add(idx)
        block_covers.append(covers)

    # Greedy initial solution
    initial_blocks = greedy_cover()
    initial_count = len(initial_blocks)

    # Track best solution
    best_blocks = list(initial_blocks)
    best_count = initial_count

    # VNS: Variable Neighborhood Search
    # Neighborhoods: swap out 1, 2, or 3 blocks for alternatives
    rng = random.Random(42)

    def try_improve(blocks, max_iter=200):
        """Local improvement via 1-swap moves."""
        blocks = list(blocks)
        n = len(blocks)
        if n == 0:
            return blocks

        coverage = compute_coverage(blocks)
        uncovered_count = sum(1 for c in coverage if c == 0)
        if uncovered_count > 0:
            # Need to add blocks first
            pass

        improved = True
        while improved and _time_left() > 1.0:
            improved = False
            for i in range(n):
                if _time_left() < 0.5:
                    break
                # Try removing block i
                if coverage_check_without_block(coverage, blocks, i, block_covers):
                    # Block i is redundant (all its t-subsets are covered by others)
                    new_blocks = blocks[:i] + blocks[i+1:]
                    blocks = new_blocks
                    n = len(blocks)
                    coverage = compute_coverage(blocks)
                    uncovered_count = sum(1 for c in coverage if c == 0)
                    improved = True
                    break
            # Also try swapping one block for another
            if not improved and n > 0:
                # Try to find a block that can be replaced by one that covers more uncovered
                for i in range(n):
                    if _time_left() < 0.5:
                        break
                    # Compute what's only covered by block i
                    only_i = set()
                    for ts_idx in block_covers[all_blocks.index(blocks[i]) if blocks[i] in all_blocks else -1]:
                        pass
                    # This is getting complex; let's use a simpler approach
                    pass

        return blocks

    def coverage_check_without_block(coverage, blocks, i, block_covers):
        """Check if block i is redundant."""
        block = blocks[i]
        # Find the index in all_blocks
        try:
            bi = all_blocks.index(block)
        except ValueError:
            return False
        for ts_idx in block_covers[bi]:
            if coverage[ts_idx] <= 1:
                return False
        return True

    # Better approach: direct VNS with bitmasks
    # Represent solution as a set of block indices
    # For efficiency, use a different strategy

    # Let's use a simpler but effective approach:
    # 1. Start with greedy
    # 2. Repeatedly try to remove redundant blocks
    # 3. Try to swap blocks for better ones
    # 4. If stuck, perturb by removing a block and re-greedy

    def remove_redundant(blocks):
        blocks = list(blocks)
        n = len(blocks)
        changed = True
        while changed and _time_left() > 1.0:
            changed = False
            coverage = compute_coverage(blocks)
            for i in range(n):
                if _time_left() < 0.5:
                    break
                block = blocks[i]
                try:
                    bi = all_blocks.index(block)
                except ValueError:
                    continue
                redundant = True
                for ts_idx in block_covers[bi]:
                    if coverage[ts_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    n -= 1
                    changed = True
                    break
        return blocks

    def swap_improve(blocks, max_rounds=50):
        """Try swapping one block for another to reduce count or maintain validity with fewer."""
        blocks = list(blocks)
        n = len(blocks)
        if n == 0:
            return blocks

        improved = True
        rounds = 0
        while improved and rounds < max_rounds and _time_left() > 1.0:
            improved = False
            rounds += 1
            coverage = compute_coverage(blocks)
            for i in range(n):
                if _time_left() < 0.5:
                    break
                # Try removing block i and see if we can cover the gap with one existing block
                # or if it's just redundant
                block = blocks[i]
                try:
                    bi = all_blocks.index(block)
                except ValueError:
                    continue

                # Check if redundant
                redundant = True
                for ts_idx in block_covers[bi]:
                    if coverage[ts_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    n -= 1
                    improved = True
                    break

                # Try swapping with another block
                # Find blocks that cover the t-subsets uniquely covered by block i
                unique_ts = set()
                for ts_idx in block_covers[bi]:
                    if coverage[ts_idx] == 1:
                        unique_ts.add(ts_idx)

                if unique_ts:
                    # Find a block that covers all unique_ts and is not already in the solution
                    for j, other_block in enumerate(all_blocks):
                        if other_block in blocks:
                            continue
                        oj = j
                        if unique_ts.issubset(block_covers[oj]):
                            # This block covers all unique t-subsets
                            # Check if adding it and removing i keeps coverage
                            new_blocks = blocks[:i] + blocks[i+1:] + [other_block]
                            new_cov = compute_coverage(new_blocks)
                            if all(c > 0 for c in new_cov):
                                blocks = new_blocks
                                n = len(blocks)
                                improved = True
                                break
                    if improved:
                        break

        return blocks

    def perturb_and_recover(blocks, num_remove=1):
        """Remove some blocks and re-greedy to recover."""
        blocks = list(blocks)
        n = len(blocks)
        if n == 0:
            return blocks

        # Remove random blocks
        indices = list(range(n))
        rng.shuffle(indices)
        remove_set = set(indices[:min(num_remove, n)])

        new_blocks = [b for i, b in enumerate(blocks) if i not in remove_set]

        # Re-greedy to cover remaining gaps
        coverage = compute_coverage(new_blocks)
        uncovered = set(i for i, c in enumerate(coverage) if c == 0)

        while uncovered and _time_left() > 1.0:
            ts_idx = min(uncovered)
            ts = all_t_subsets[ts_idx]
            # Find a block that covers this t-subset and maximizes coverage of uncovered
            best_block = None
            best_gain = -1
            for b in all_blocks:
                if b in new_blocks:
                    continue
                bi = all_blocks.index(b)
                gain = len(block_covers[bi] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            if best_block is None or best_gain <= 0:
                break
            new_blocks.append(best_block)
            bi = all_blocks.index(best_block)
            uncovered -= block_covers[bi]

        return new_blocks

    # Main VNS loop
    current_blocks = list(initial_blocks)
    current_count = len(current_blocks)

    vns_iter = 0
    while _time_left() > 5.0:
        vns_iter += 1

        # Local improvement
        current_blocks = remove_redundant(current_blocks)
        current_blocks = swap_improve(current_blocks)
        current_count = len(current_blocks)

        if current_count < best_count:
            best_blocks = list(current_blocks)
            best_count = current_count

        # Perturb
        if vns_iter % 3 == 0:  # Perturb every 3 iterations
            num_remove = min(1 + vns_iter // 10, 5)
            perturbed = perturb_and_recover(current_blocks, num_remove)
            if len(perturbed) < current_count:
                current_blocks = perturbed
                current_count = len(perturbed)
            elif len(perturbed) == current_count:
                current_blocks = perturbed
                current_count = len(perturbed)

        # Check time
        if _time_left() < 10.0:
            break

    # Final cleanup
    best_blocks = remove_redundant(best_blocks)
    best_blocks = swap_improve(best_blocks)

    # Convert tuples to lists
    return [list(b) for b in best_blocks]

# EVOLVE-BLOCK-END
