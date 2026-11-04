"""Covering design C(19,8,5) with fast greedy + local search optimization."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # For each block (k-subset), precompute which t-subsets it covers
    # We'll use a bitset approach for fast coverage tracking

    # Index structure: for each point, list of t-subsets containing it
    point_to_tsubs = [[] for _ in range(v)]
    for i, sub in enumerate(all_t_subs):
        for p in sub:
            point_to_tsubs[p].append(i)

    # For each pair of points, list of t-subsets containing both
    # This helps with incremental gain calculation
    # Actually, let's use a different approach:
    # For a partial block B, gain(x) = number of uncovered t-subsets that contain x and are subsets of B∪{x}
    # = number of uncovered t-subsets S where S ⊆ B∪{x} and x ∈ S
    # Since |S| = t and x ∈ S, we need S\{x} ⊆ B and |S\{x}| = t-1
    # So gain(x) = |{S : S ⊆ B∪{x}, x ∈ S, S uncovered}|

    def time_ok():
        return time.time() < deadline

    def greedy_construct():
        """Fast greedy covering with efficient gain calculation."""
        covered = bytearray(n_t)
        uncovered_count = n_t
        blocks = []

        # For efficiency, track which t-subsets are uncovered
        uncovered_set = set(range(n_t))

        while uncovered_set and time_ok():
            # Pick the first uncovered t-subset (or a random one for diversity)
            # Use the lexicographically smallest for determinism in base greedy
            first_idx = min(uncovered_set)
            first_sub = all_t_subs[first_idx]

            # Build block starting with first_sub
            block = set(first_sub)

            while len(block) < k and time_ok():
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Count uncovered t-subsets that would be covered by adding x
                    # These are t-subsets S where S ⊆ block ∪ {x} and x ∈ S
                    # Equivalently, S\{x} ⊆ block and |S\{x}| = t-1
                    gain = 0
                    # Iterate over (t-1)-subsets of block
                    for rest in itertools.combinations(sorted(block), t - 1):
                        sub_with_x = tuple(sorted(rest + (x,)))
                        idx = t_sub_to_idx.get(sub_with_x)
                        if idx is not None and idx in uncovered_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.add(best_x)

            block = sorted(block)
            blocks.append(block)

            # Update coverage
            for sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[sub]
                if idx in uncovered_set:
                    uncovered_set.discard(idx)
                    covered[idx] = 1
                    uncovered_count -= 1

        return blocks, uncovered_set

    # Run initial greedy
    blocks, remaining = greedy_construct()

    if not time_ok():
        return blocks

    # Local optimization: try to remove redundant blocks
    def is_block_redundant(blocks, block_idx, t_sub_to_idx, all_t_subs):
        """Check if block at block_idx is redundant (all its t-subsets are covered by other blocks)."""
        block = blocks[block_idx]
        # For each t-subset in this block, check if it's covered by another block
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            found = False
            for j, other_block in enumerate(blocks):
                if j == block_idx:
                    continue
                if all(p in other_block for p in sub):
                    found = True
                    break
            if not found:
                return False
        return True

    # Remove redundant blocks (pass 1)
    changed = True
    while changed and time_ok():
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time_ok():
                if is_block_redundant(blocks, i, t_sub_to_idx, all_t_subs):
                    blocks.pop(i)
                    changed = True

    # Local search: try swapping points in blocks to improve coverage
    def try_swap(blocks, t_sub_to_idx, all_t_subs):
        """Try to improve the covering by swapping points in blocks."""
        # Build coverage count for each t-subset
        covered_count = np.zeros(n_t, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[sub]
                covered_count[idx] += 1

        improved = True
        while improved and time_ok():
            improved = False
            random.shuffle(blocks)
            for bi, block in enumerate(blocks):
                if not time_ok():
                    break
                block_set = set(block)
                for pos in range(len(block)):
                    if not time_ok():
                        break
                    old_point = block[pos]
                    for new_point in range(v):
                        if new_point in block_set:
                            continue
                        # Calculate net change in uncovered t-subsets
                        # t-subsets lost: those containing old_point but not new_point, in this block
                        # t-subsets gained: those containing new_point but not old_point, in this block
                        # Only matters if covered_count == 1 for lost, or 0 for gained
                        lost = 0
                        gained = 0
                        for sub in itertools.combinations(block, t):
                            if old_point in sub and new_point not in sub:
                                idx = t_sub_to_idx[sub]
                                if covered_count[idx] == 1:
                                    lost += 1
                            elif new_point in sub and old_point not in sub:
                                idx = t_sub_to_idx[sub]
                                if covered_count[idx] == 0:
                                    gained += 1
                        if gained > lost:
                            # Apply swap
                            block[pos] = new_point
                            block_set.remove(old_point)
                            block_set.add(new_point)
                            block.sort()
                            # Update coverage counts
                            for sub in itertools.combinations(block, t):
                                if new_point in sub and old_point not in sub:
                                    idx = t_sub_to_idx[sub]
                                    covered_count[idx] += 1
                            for sub in itertools.combinations(block, t):
                                if old_point in sub and new_point not in sub:
                                    idx = t_sub_to_idx[sub]
                                    covered_count[idx] -= 1
                            improved = True
                            break

        return blocks, covered_count

    blocks, covered_count = try_swap(blocks, t_sub_to_idx, all_t_subs)

    # Final redundancy removal
    changed = True
    while changed and time_ok():
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time_ok():
                if is_block_redundant(blocks, i, t_sub_to_idx, all_t_subs):
                    blocks.pop(i)
                    changed = True

    # Verify coverage is complete
    # Rebuild coverage from final blocks
    covered = bytearray(n_t)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            covered[idx] = 1

    # If not fully covered, we need to add more blocks
    if not all(covered) and time_ok():
        uncovered_set = {i for i in range(n_t) if not covered[i]}
        # Add greedy blocks to cover remaining
        while uncovered_set and time_ok():
            first_idx = min(uncovered_set)
            first_sub = all_t_subs[first_idx]
            block = set(first_sub)
            while len(block) < k and time_ok():
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(sorted(block), t - 1):
                        sub_with_x = tuple(sorted(rest + (x,)))
                        idx = t_sub_to_idx.get(sub_with_x)
                        if idx is not None and idx in uncovered_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.add(best_x)
            block = sorted(block)
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[sub]
                if idx in uncovered_set:
                    uncovered_set.discard(idx)
                    covered[idx] = 1

    return blocks
# EVOLVE-BLOCK-END
