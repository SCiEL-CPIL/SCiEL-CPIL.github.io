"""Improved covering design: greedy with cost-based block selection + local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubsets)

    # Map each t-subset to an index
    tsub_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Coverage count: how many blocks cover each t-subset
    cover_count = [0] * n_tsubs

    blocks = []
    block_tsubs = []  # list of t-subset indices covered by each block

    def is_valid():
        return all(c > 0 for c in cover_count)

    def add_block(block):
        block = tuple(sorted(block))
        blocks.append(block)
        tsubs_in_block = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            cover_count[idx] += 1
            tsubs_in_block.append(idx)
        block_tsubs.append(tsubs_in_block)

    def remove_block(bi):
        for idx in block_tsubs[bi]:
            cover_count[idx] -= 1
        blocks[bi] = None
        block_tsubs[bi] = None

    def restore_block(bi, block):
        blocks[bi] = block
        tsubs_in_block = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            cover_count[idx] += 1
            tsubs_in_block.append(idx)
        block_tsubs[bi] = tsubs_in_block

    # --- Phase 1: Greedy construction ---
    # For each t-subset, precompute which points can be in a block covering it
    # Actually, any k-subset containing the t-subset covers it.
    # For cost-based selection: a t-subset is "hard" if few k-blocks contain it.
    # Number of k-blocks containing a specific t-subset = C(v-t, k-t)
    # This is the same for all t-subsets, so we need a dynamic measure.

    # Dynamic cost: for each uncovered t-subset, count how many candidate points
    # would cover it if added to a growing block. Initially, all points work equally.
    # Better: use the number of currently-uncovered t-subsets that share points.

    uncovered = set(range(n_tsubs))  # indices of uncovered t-subsets

    while uncovered:
        if time.time() > deadline:
            break

        # Select the "hardest" uncovered t-subset: the one with the fewest
        # potential completions to a k-block that also covers other uncovered t-subsets.
        # Approximation: pick the uncovered t-subset whose points appear in the
        # fewest other uncovered t-subsets (i.e., most isolated).
        # Simpler heuristic: pick the one with max sum of "rarity" of its elements.
        # Rarity of element x = number of uncovered t-subsets containing x (lower = rarer).
        # We want to cover rare elements first.

        # Compute element rarity among uncovered
        elem_count = [0] * v
        for idx in uncovered:
            for x in all_tsubsets[idx]:
                elem_count[x] += 1

        # Score each uncovered t-subset: sum of 1/element_count (higher = more isolated/rare)
        # To avoid division by zero, use a large number for count=0 (shouldn't happen if uncovered)
        best_idx = -1
        best_score = -1
        # Sample if too many to evaluate all (shouldn't be an issue for v=19)
        for idx in uncovered:
            score = 0.0
            for x in all_tsubsets[idx]:
                score += 1.0 / max(1, elem_count[x])
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx == -1:
            break

        # Start block with the selected t-subset
        block = list(all_tsubsets[best_idx])

        # Greedily add points to complete the block
        while len(block) < k:
            if time.time() > deadline:
                break
            block_set = set(block)
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Gain: number of uncovered t-subsets that would be covered
                # by adding x to the current block
                gain = 0
                for ts_idx in uncovered:
                    ts = all_tsubsets[ts_idx]
                    # Check if ts is covered by (block + [x])
                    # i.e., ts is a t-subset of block+[x]
                    # ts must be a subset of block+[x] and contain x or be in block
                    # More precisely: ts ⊆ (block ∪ {x})
                    if all(elem in block_set or elem == x for elem in ts):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point == -1 or best_gain == 0:
                # No point helps; just pick an arbitrary uncovered element
                # to keep making progress (shouldn't happen in valid covering)
                for x in range(v):
                    if x not in block_set:
                        best_point = x
                        break
            block.append(best_point)

        block.sort()
        add_block(block)

        # Update uncovered
        for idx in block_tsubs[-1]:
            if cover_count[idx] == 1:  # just became covered
                uncovered.discard(idx)

    if not uncovered:
        return blocks

    # If deadline hit during greedy, try to finish quickly
    if time.time() > deadline - 2:
        # Emergency: add blocks for remaining uncovered
        while uncovered:
            idx = min(uncovered)
            block = list(all_tsubsets[idx])
            while len(block) < k:
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
            block.sort()
            add_block(block)
            for idx2 in block_tsubs[-1]:
                if cover_count[idx2] == 1:
                    uncovered.discard(idx2)
        return blocks

    # --- Phase 2: Local search ---
    # Try to remove redundant blocks
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        n_blocks = len(blocks)
        for bi in range(n_blocks - 1, -1, -1):
            if blocks[bi] is None:
                continue
            # Check if removing block bi leaves all t-subsets covered
            removable = True
            for idx in block_tsubs[bi]:
                if cover_count[idx] == 1:
                    removable = False
                    break
            if removable:
                remove_block(bi)
                improved = True

    # Try swap: replace a point in a block to cover more uncovered... 
    # Actually, at this point everything is covered. Try to see if we can
    # restructure to use fewer blocks.
    # Try: for each block, try replacing one point with another and see if
    # we can then remove another block.
    if time.time() < deadline - 10:
        for _ in range(50):  # limited iterations
            if time.time() > deadline - 5:
                break
            found_improvement = False
            for bi in range(len(blocks)):
                if blocks[bi] is None:
                    continue
                if time.time() > deadline - 5:
                    break
                block = list(blocks[bi])
                for pos in range(k):
                    old_point = block[pos]
                    for new_point in range(v):
                        if new_point == old_point:
                            continue
                        if new_point in block:
                            continue
                        block[pos] = new_point
                        block.sort()
                        # Check if this new block is valid (distinct elements)
                        if len(set(block)) != k:
                            block[pos] = old_point
                            continue
                        # Temporarily apply the change
                        # Remove old block's coverage
                        for idx in block_tsubs[bi]:
                            cover_count[idx] -= 1
                        # Add new block's coverage
                        new_tsubs = []
                        for ts in itertools.combinations(block, t):
                            idx = tsub_index[ts]
                            cover_count[idx] += 1
                            new_tsubs.append(idx)

                        # Check if all still covered
                        valid = all(c > 0 for c in cover_count)

                        if valid:
                            # See if we can now remove some other block
                            for bj in range(len(blocks)):
                                if bj == bi or blocks[bj] is None:
                                    continue
                                removable = True
                                for idx in block_tsubs[bj]:
                                    if cover_count[idx] == 1:
                                        removable = False
                                        break
                                if removable:
                                    # Accept the swap and removal
                                    blocks[bi] = block
                                    block_tsubs[bi] = new_tsubs
                                    remove_block(bj)
                                    found_improvement = True
                                    break
                            if found_improvement:
                                break
                        else:
                            # Revert
                            for idx in new_tsubs:
                                cover_count[idx] -= 1
                            for idx in block_tsubs[bi]:
                                cover_count[idx] += 1
                            block[pos] = old_point

                        # Revert for next iteration
                        if not found_improvement:
                            for idx in new_tsubs:
                                cover_count[idx] -= 1
                            for idx in block_tsubs[bi]:
                                cover_count[idx] += 1
                            block[pos] = old_point

                    if found_improvement:
                        break
                if found_improvement:
                    break
            if not found_improvement:
                break

    # Clean up None entries
    final_blocks = [b for b in blocks if b is not None]
    return [list(b) for b in final_blocks]
# EVOLVE-BLOCK-END
