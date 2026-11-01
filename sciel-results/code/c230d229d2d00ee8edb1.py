"""Covering design C(24,7,3) via bitset-accelerated greedy with randomized restarts and redundancy removal."""
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute (t-1)-subsets
    t_minus_1_subs = list(itertools.combinations(range(v), t - 1))
    num_tm1 = len(t_minus_1_subs)

    # For each point x and each (t-1)-subset s (not containing x), the t-subset index of s∪{x}
    x_t_sub_idx = [[-1] * num_tm1 for _ in range(v)]
    for x in range(v):
        for s_idx, s in enumerate(t_minus_1_subs):
            if x in s:
                continue
            new_sub = tuple(sorted(s + (x,)))
            x_t_sub_idx[x][s_idx] = t_sub_to_idx.get(new_sub, -1)

    def gain_for_point(x, block, covered):
        """Number of uncovered t-subsets that adding x to block would cover."""
        block_set = set(block)
        if x in block_set:
            return 0
        count = 0
        for s_idx, s in enumerate(t_minus_1_subs):
            # Check if s is a subset of block
            s_set = set(s)
            if not s_set.issubset(block_set):
                continue
            t_idx = x_t_sub_idx[x][s_idx]
            if t_idx != -1 and not covered[t_idx]:
                count += 1
        return count

    def update_covered(block, covered):
        """Mark all t-subsets of block as covered."""
        for sub in itertools.combinations(block, t):
            covered[t_sub_to_idx[sub]] = True

    def is_complete(covered):
        return all(covered)

    def remove_redundant_blocks(blocks, covered, deadline):
        """Try to remove redundant blocks."""
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    return blocks
                saved = blocks.pop(i)
                # Rebuild covered
                cov_check = [False] * num_t
                for b in blocks:
                    for sub in itertools.combinations(b, t):
                        cov_check[t_sub_to_idx[sub]] = True
                if all(cov_check):
                    # Redundant, keep removed
                    changed = True
                    break
                else:
                    blocks.insert(i, saved)
        return blocks

    best_blocks = None
    best_count = float('inf')
    rng = random.Random(42)

    for run_id in range(500):
        if time.time() > deadline:
            break

        covered = [False] * num_t
        blocks = []

        # Shuffle the order of t-subsets for randomization
        order = list(range(num_t))
        rng.shuffle(order)

        # Process t-subsets in shuffled order
        for idx in order:
            if time.time() > deadline:
                break
            if covered[idx]:
                continue

            # Start a new block with this t-subset
            start_combo = all_t_subs[idx]
            block = list(start_combo)
            block_set = set(block)

            # Mark this t-subset as covered
            covered[idx] = True

            # Add points to reach size k
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    g = gain_for_point(x, block, covered)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
                block_set.add(best_x)
                # Update covered
                update_covered(block, covered)

            block.sort()
            blocks.append(block)

        # Check completeness
        if all(covered):
            # Redundancy removal
            if time.time() < deadline - 10:
                blocks = remove_redundant_blocks(blocks, covered, deadline)

            if len(blocks) < best_count:
                best_blocks = blocks[:]
                best_count = len(blocks)

        if best_count <= 76:
            break

    if best_blocks is None:
        # Fallback: deterministic greedy
        covered = [False] * num_t
        blocks = []
        for idx in range(num_t):
            if covered[idx]:
                continue
            start_combo = all_t_subs[idx]
            block = list(start_combo)
            block_set = set(block)
            covered[idx] = True
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    g = gain_for_point(x, block, covered)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
                block_set.add(best_x)
                update_covered(block, covered)
            block.sort()
            blocks.append(block)
        best_blocks = blocks

    return best_blocks
