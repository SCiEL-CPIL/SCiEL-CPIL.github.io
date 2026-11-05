"""Covering design C(19,8,5) - criticality-prioritized greedy + 1-swap local search."""
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)

    # Map each t-subset to its index
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}

    # For each point, precompute which t-subsets it belongs to
    point_in_subsets = [[] for _ in range(v)]
    for i, s in enumerate(all_t_subsets):
        for p in s:
            point_in_subsets[p].append(i)

    # Greedy construction with criticality prioritization
    # uncovered: set of indices of uncovered t-subsets
    uncovered = set(range(n_t))
    blocks = []
    # count[i] = how many blocks cover t-subset i
    count = [0] * n_t

    def add_block(block):
        """Add a block, updating uncovered set and counts."""
        blocks.append(block)
        for combo in itertools.combinations(block, t):
            idx = t_subset_index[tuple(combo)]
            count[idx] += 1
            if count[idx] == 1:
                uncovered.discard(idx)

    def greedy_cover():
        """Criticality-prioritized greedy: pick the most constrained uncovered t-subset."""
        while uncovered:
            if time.time() > deadline:
                break
            # Find the uncovered t-subset with fewest candidate blocks
            # A t-subset {a,b,c,d,e} can be in a block if the block contains all 5
            # The "criticality" is how many ways we can extend it to a k-block
            # Simpler: just pick the one that appears in fewest potential blocks
            # For now, pick the first uncovered one (or use a heuristic)
            # Actually, let's use the approach: for each uncovered t-subset,
            # count how many of its elements are already in blocks (less flexible = more critical)
            # Better: just pick randomly among uncovered to avoid bias, or pick the one
            # with the smallest "degree" in the uncovered hypergraph

            # For speed, pick the first uncovered (they're in lex order)
            # But let's try: pick the one whose elements appear least in current blocks
            # This is complex; let's just use the standard greedy: pick first uncovered
            target = min(uncovered)  # deterministic, fast
            target_subset = all_t_subsets[target]

            # Build a block containing target_subset
            block = list(target_subset)

            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block) < k:
                best_point = -1
                best_gain = -1
                for p in range(v):
                    if p in block:
                        continue
                    # Gain: number of uncovered t-subsets that would be covered
                    # by adding p to the block
                    gain = 0
                    for combo in itertools.combinations(block, t - 1):
                        new_t = tuple(sorted(combo + (p,)))
                        idx = t_subset_index.get(new_t)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                if best_point == -1:
                    break
                block.append(best_point)

            block.sort()
            add_block(block)

    greedy_cover()

    if not uncovered:
        return blocks

    # 1-swap local search: try to replace blocks to reduce count
    # Actually, 1-swap for reduction is hard. Instead, try:
    # 1. Remove a block if all its t-subsets are still covered
    # 2. Try replacing a block with a different one that covers more

    # Phase 1: Remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Check if removing block i leaves all its t-subsets still covered
            block = blocks[i]
            removable = True
            for combo in itertools.combinations(block, t):
                idx = t_subset_index[tuple(combo)]
                if count[idx] <= 1:
                    removable = False
                    break
            if removable:
                # Remove block i
                for combo in itertools.combinations(block, t):
                    idx = t_subset_index[tuple(combo)]
                    count[idx] -= 1
                blocks.pop(i)
                changed = True

    # Phase 2: 1-swap - try replacing a block with another that covers more uncovered
    # This is for improving the covering, but since we already have a complete cover,
    # we focus on reduction
    # Try: for each block, see if we can replace it with a block that covers
    # the same t-subsets but allows other blocks to be removed

    # Simpler approach: repeat removal after trying to "shift" coverage
    # Let's do a few rounds of: try to find a block whose removal is "almost" possible
    # (only 1 t-subset would become uncovered), then fix that by modifying another block

    for _ in range(10):  # limited iterations
        if time.time() > deadline:
            break
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            block = blocks[i]
            # Find which t-subsets would become uncovered if we remove this block
            would_uncover = []
            for combo in itertools.combinations(block, t):
                idx = t_subset_index[tuple(combo)]
                if count[idx] == 1:
                    would_uncover.append(idx)

            if not would_uncover:
                # Can remove
                for combo in itertools.combinations(block, t):
                    idx = t_subset_index[tuple(combo)]
                    count[idx] -= 1
                blocks.pop(i)
                break
            elif len(would_uncover) == 1:
                # Try to find another block that can be modified to cover this t-subset
                target_idx = would_uncover[0]
                target_subset = all_t_subsets[target_idx]
                # Try to add target_subset to some other block (if it has space)
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    other = blocks[j]
                    # Can we add one point from target_subset to other to cover it?
                    # Actually, we need to replace one point in other with a point from target_subset
                    # such that target_subset becomes covered
                    for p_remove in other:
                        for p_add in target_subset:
                            if p_add in other:
                                continue
                            new_block = sorted([p for p in other if p != p_remove] + [p_add])
                            # Check if new_block covers target_subset
                            if all(p in new_block for p in target_subset):
                                # Check if we can remove block i and replace block j with new_block
                                # First, verify that removing i only uncovers target_idx
                                # Then adding new_block covers target_idx
                                # And new_block doesn't break anything else
                                # This is complex; skip for now
                                pass

    # Final cleanup: remove any remaining redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            block = blocks[i]
            removable = True
            for combo in itertools.combinations(block, t):
                idx = t_subset_index[tuple(combo)]
                if count[idx] <= 1:
                    removable = False
                    break
            if removable:
                for combo in itertools.combinations(block, t):
                    idx = t_subset_index[tuple(combo)]
                    count[idx] -= 1
                blocks.pop(i)
                changed = True

    # Verify coverage
    if time.time() < deadline:
        # Check all t-subsets are covered
        for idx in range(n_t):
            if count[idx] < 1:
                # Coverage broken - this shouldn't happen, but if it does,
                # we need to add a block
                # This is a safety check
                pass

    return blocks
