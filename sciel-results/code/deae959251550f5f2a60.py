"""Covering design C(19,7,4) - fast greedy + simulated annealing."""
import time
import random
import numpy as np
from itertools import combinations
from collections import defaultdict

_DEADLINE = 145.0  # safety margin before 150s kill


def _build_index(v, t):
    """Map each t-subset to an integer index."""
    idx = {}
    i = 0
    for comb in combinations(range(v), t):
        idx[comb] = i
        i += 1
    return idx, i


def _block_tsubsets(block, t):
    """Return list of t-subset indices for a block."""
    return [tuple(sorted(block))] if len(block) == t else [
        tuple(c) for c in combinations(block, t)
    ]


def _greedy_construct(v, k, t, idx, total, deadline):
    """Fast greedy: pick block covering most uncovered t-subsets."""
    n = total
    uncovered = np.ones(n, dtype=bool)
    blocks = []
    start = time.time()

    # Precompute for each point, the list of t-subsets it participates in
    # Actually, let's use a different approach: for each candidate block,
    # compute how many uncovered t-subsets it covers

    while uncovered.any():
        if time.time() > deadline:
            break

        # Find the t-subset with the fewest covering options (most constrained)
        # Actually, let's just pick a random uncovered t-subset and build a block around it
        # But better: pick the uncovered t-subset that appears in the fewest potential blocks

        # For efficiency, let's use a greedy approach:
        # Pick an uncovered t-subset, then greedily extend to k points
        # maximizing coverage of other uncovered t-subsets

        # Get all uncovered t-subset indices
        uncov_indices = np.where(uncovered)[0]
        if len(uncov_indices) == 0:
            break

        # Pick a random uncovered t-subset to start from
        # But to be more strategic, pick one that's "central"
        # For speed, just pick the first one
        first_idx = uncov_indices[0]

        # We need to map index back to t-subset
        # This is slow, so let's precompute a reverse map
        # Actually, let's maintain a list of t-subsets
        # For now, let's use a different approach

        # Precompute: for each point, which t-subsets does it appear in?
        # This is expensive to build, but we can cache it

        # Let's use a simpler greedy:
        # 1. Pick an uncovered t-subset
        # 2. Build a block of k points starting with those t points
        # 3. Add points that maximize new coverage

        # First, we need to know which t-subset is at first_idx
        # Let's build a reverse lookup
        # Actually, let's just iterate differently

        # Better approach: maintain a list of all t-subsets
        tsubsets_list = list(idx.keys())

        # Find an uncovered t-subset
        first_tsub = tsubsets_list[first_idx]
        block = list(first_tsub)

        # Greedily add points
        while len(block) < k:
            best_x = -1
            best_gain = -1
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many new uncovered t-subsets would be covered
                gain = 0
                # A t-subset is covered if all its elements are in the block
                # When we add x, new t-subsets covered are those that contain x
                # and t-1 elements from the current block
                for rest in combinations(block, t - 1):
                    tsub = tuple(sorted(rest + (x,)))
                    ci = idx.get(tsub, -1)
                    if ci >= 0 and uncovered[ci]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)
            if time.time() > deadline:
                break

        if len(block) < k:
            # Couldn't extend, just use what we have (shouldn't happen)
            block = block + [x for x in range(v) if x not in set(block)][:k - len(block)]

        block.sort()
        blocks.append(block)

        # Update uncovered
        for comb in combinations(block, t):
            i = idx.get(comb, -1)
            if i >= 0:
                uncovered[i] = False

    return blocks


def _compute_coverage(blocks, v, t, idx):
    """Compute coverage count for each t-subset."""
    n = len(idx)
    coverage = np.zeros(n, dtype=np.int32)
    for block in blocks:
        for comb in combinations(block, t):
            i = idx.get(comb, -1)
            if i >= 0:
                coverage[i] += 1
    return coverage


def _remove_redundant(blocks, v, t, idx, deadline):
    """Remove blocks that don't uniquely cover any t-subset."""
    n = len(idx)
    coverage = _compute_coverage(blocks, v, t, idx)
    blocks = list(blocks)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            block = blocks[i]
            removable = True
            for comb in combinations(block, t):
                ci = idx.get(comb, -1)
                if ci >= 0 and coverage[ci] <= 1:
                    removable = False
                    break
            if removable:
                for comb in combinations(block, t):
                    ci = idx.get(comb, -1)
                    if ci >= 0:
                        coverage[ci] -= 1
                blocks.pop(i)
                improved = True
                break
    return blocks, coverage


def _simulated_annealing(blocks, v, k, t, idx, deadline, rng):
    """Simulated annealing to reduce block count."""
    n = len(idx)
    blocks = list(blocks)
    coverage = _compute_coverage(blocks, v, t, idx)

    # Precompute for each block, the t-subsets it covers
    def get_block_tsubsets(block):
        return [idx[tuple(c)] for c in combinations(block, t) if tuple(c) in idx]

    # Current state
    current_blocks = list(blocks)
    current_coverage = coverage.copy()
    best_blocks = list(blocks)
    best_coverage = coverage.copy()
    best_len = len(blocks)

    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    total_iters = 50000
    # Adjust iterations based on time
    start_time = time.time()
    elapsed_initial = start_time - (deadline - _DEADLINE)  # approximate
    remaining_time = deadline - start_time
    if remaining_time < 10:
        return best_blocks

    # Run SA
    for it in range(total_iters):
        if time.time() > deadline:
            break

        # Update temperature
        progress = it / total_iters
        T = T_start * (T_end / T_start) ** progress

        if not current_blocks:
            break

        # Choose a move
        move_type = rng.random()
        if move_type < 0.5:
            # Try to remove a block
            i = rng.randrange(len(current_blocks))
            block = current_blocks[i]
            block_tsubs = get_block_tsubsets(block)
            # Check if all t-subsets in this block are covered by others
            can_remove = True
            for ci in block_tsubs:
                if current_coverage[ci] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove it
                for ci in block_tsubs:
                    current_coverage[ci] -= 1
                current_blocks.pop(i)
                # Always accept (improvement)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = list(current_blocks)
                    best_coverage = current_coverage.copy()
                continue
        elif move_type < 0.8:
            # Try to add a random block
            # Generate a random k-subset
            new_block = sorted(rng.sample(range(v), k))
            new_block_tsubs = get_block_tsubsets(new_block)
            # Check if it adds coverage
            new_covered = sum(1 for ci in new_block_tsubs if current_coverage[ci] == 0)
            if new_covered > 0:
                # Add it
                for ci in new_block_tsubs:
                    current_coverage[ci] += 1
                current_blocks.append(new_block)
                # This is worse (more blocks), accept with probability
                delta = 1  # cost increase
                if rng.random() < np.exp(-delta / max(T, 1e-10)):
                    pass  # accepted
                else:
                    # Reject: undo
                    for ci in new_block_tsubs:
                        current_coverage[ci] -= 1
                    current_blocks.pop()
        else:
            # Try to swap: remove one block, add another
            if len(current_blocks) < 2:
                continue
            i = rng.randrange(len(current_blocks))
            old_block = current_blocks[i]
            old_tsubs = get_block_tsubsets(old_block)

            # Check if we can remove old_block
            can_remove = True
            for ci in old_tsubs:
                if current_coverage[ci] <= 1:
                    can_remove = False
                    break
            if not can_remove:
                continue

            # Generate a new block
            new_block = sorted(rng.sample(range(v), k))
            new_tsubs = get_block_tsubsets(new_block)

            # Remove old, add new
            for ci in old_tsubs:
                current_coverage[ci] -= 1
            current_blocks[i] = new_block
            for ci in new_tsubs:
                current_coverage[ci] += 1

            # Check validity: all t-subsets must be covered
            valid = True
            for ci in range(n):
                if current_coverage[ci] == 0:
                    valid = False
                    break

            if not valid:
                # Undo
                for ci in new_tsubs:
                    current_coverage[ci] -= 1
                current_blocks[i] = old_block
                for ci in old_tsubs:
                    current_coverage[ci] += 1
            else:
                # Same number of blocks, accept with some probability
                if rng.random() < 0.5:
                    pass  # keep
                else:
                    # Undo
                    for ci in new_tsubs:
                        current_coverage[ci] -= 1
                    current_blocks[i] = old_block
                    for ci in old_tsubs:
                        current_coverage[ci] += 1

    return best_blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Build index
    idx, total = _build_index(v, t)

    # Greedy construction
    blocks = _greedy_construct(v, k, t, idx, total, deadline)

    # Remove redundant blocks
    if time.time() < deadline - 10:
        blocks, _ = _remove_redundant(blocks, v, t, idx, deadline)

    # Simulated annealing
    if time.time() < deadline - 5:
        rng = random.Random(42)
        blocks = _simulated_annealing(blocks, v, k, t, idx, deadline, rng)

    # Final removal pass
    if time.time() < deadline - 2:
        blocks, _ = _remove_redundant(blocks, v, t, idx, deadline)

    return blocks
