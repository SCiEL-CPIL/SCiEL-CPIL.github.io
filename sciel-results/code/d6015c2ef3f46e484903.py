"""Stronger covering design C(19,7,4) solver.

Strategy:
1. Fast greedy construction with incremental coverage counting.
2. Redundant block removal.
3. Local search: try to remove blocks, and replace blocks to enable further removals.
4. Deadline guard to return best result before timeout.
"""

import itertools
import time
import random
from collections import defaultdict, Counter


def _build_sub_index(v, t):
    """Map each t-subset to a unique index; return (list_of_subsets, point_to_subs)."""
    subs = list(itertools.combinations(range(v), t))
    sub_index = {s: i for i, s in enumerate(subs)}
    # For each point, list of (sub_index, other_points_tuple) for subsets containing it
    # Actually we need: for a given (t-1)-subset and a candidate point x, find the t-subset
    # Precompute: for each point x, for each (t-1)-subset of the other points, the sub index
    # This is expensive to store fully. Instead, store a dict: (x, frozenset(rest)) -> idx
    # But that's C(v,t) * t entries. For v=19,t=4: 3876*4 = 15504. Fine.
    point_sub_map = {}
    for idx, sub in enumerate(subs):
        for x in sub:
            rest = tuple(p for p in sub if p != x)
            point_sub_map[(x, rest)] = idx
    return subs, sub_index, point_sub_map


def _greedy_construct(v, k, t, deadline):
    """Fast greedy covering using bitset gain computation."""
    subs, sub_index, point_sub_map = _build_sub_index(v, t)
    n_subs = len(subs)
    # uncovered as a set of indices
    uncovered = set(range(n_subs))
    blocks = []

    while uncovered and time.time() < deadline:
        # Start with the lexicographically smallest uncovered t-subset
        min_idx = min(uncovered)
        min_sub = subs[min_idx]
        block = list(min_sub)
        block_set = set(block)

        # Greedily add points
        while len(block) < k and time.time() < deadline:
            best_x = -1
            best_gain = -1

            for x in range(v):
                if x in block_set:
                    continue
                # Compute gain: count uncovered t-subsets containing x and (t-1) from block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    rest_tuple = tuple(rest)
                    idx = point_sub_map.get((x, rest_tuple), -1)
                    if idx >= 0 and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x == -1:
                break
            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(block)

        # Update uncovered
        for sub in itertools.combinations(block, t):
            idx = sub_index.get(sub, -1)
            if idx >= 0:
                uncovered.discard(idx)

    return blocks


def _remove_redundant_blocks(blocks, v, k, t, deadline):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks

    # Count how many blocks cover each t-subset
    sub_count = defaultdict(int)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            sub_count[sub] += 1

    # A block is redundant if every t-subset it covers is also covered by another block
    kept = []
    for block in blocks:
        redundant = True
        for sub in itertools.combinations(block, t):
            if sub_count[sub] == 1:
                redundant = False
                break
        if not redundant:
            kept.append(block)

    return kept


def _local_search(blocks, v, k, t, deadline):
    """Try to improve the covering by removing or replacing blocks."""
    subs, sub_index, point_sub_map = _build_sub_index(v, t)
    n_subs = len(subs)

    # Count coverage using indices
    sub_count = [0] * n_subs
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_index.get(sub, -1)
            if idx >= 0:
                sub_count[idx] += 1

    # First, remove any redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            block = blocks[i]
            needed = False
            for sub in itertools.combinations(block, t):
                idx = sub_index.get(sub, -1)
                if idx >= 0 and sub_count[idx] == 1:
                    needed = True
                    break
            if not needed:
                for sub in itertools.combinations(block, t):
                    idx = sub_index.get(sub, -1)
                    if idx >= 0:
                        sub_count[idx] -= 1
                blocks.pop(i)
                changed = True
                break

    # Now try to replace blocks to enable further removals
    random.seed(42)
    improved = True
    max_replacements = 5000
    replacements = 0

    while improved and time.time() < deadline and replacements < max_replacements:
        improved = False
        replacements += 1

        # Try removing each block and see if coverage is still complete
        for i in range(len(blocks)):
            if time.time() > deadline:
                return blocks
            block = blocks[i]
            needed = False
            for sub in itertools.combinations(block, t):
                idx = sub_index.get(sub, -1)
                if idx >= 0 and sub_count[idx] == 1:
                    needed = True
                    break
            if not needed:
                for sub in itertools.combinations(block, t):
                    idx = sub_index.get(sub, -1)
                    if idx >= 0:
                        sub_count[idx] -= 1
                blocks.pop(i)
                improved = True
                break

        if improved:
            continue

        # Try replacing a block with a better one
        if len(blocks) > 0:
            # Pick a random block to try replacing
            idx = random.randint(0, len(blocks) - 1)
            old_block = blocks[idx]
            old_subs = list(itertools.combinations(old_block, t))
            old_sub_indices = []
            for sub in old_subs:
                si = sub_index.get(sub, -1)
                if si >= 0:
                    old_sub_indices.append(si)

            # Count how many of old_block's subsets are uniquely covered by it
            unique_count = 0
            for si in old_sub_indices:
                if sub_count[si] == 1:
                    unique_count += 1

            # Generate several random candidate blocks
            best_candidate = None
            best_score = -1

            for _ in range(30):
                new_block = sorted(random.sample(range(v), k))
                new_sub_indices = []
                for sub in itertools.combinations(new_block, t):
                    si = sub_index.get(sub, -1)
                    if si >= 0:
                        new_sub_indices.append(si)

                # Compute score: how many new subsets does this cover that aren't already covered?
                # And how many unique subsets would we lose?
                gained = 0
                for si in new_sub_indices:
                    if sub_count[si] == 0:
                        gained += 1

                lost = 0
                new_set = set(new_sub_indices)
                for si in old_sub_indices:
                    if sub_count[si] == 1 and si not in new_set:
                        lost += 1

                score = gained - lost * 10  # Heavily penalize losing unique coverage

                if score > best_score:
                    best_score = score
                    best_candidate = new_block

            if best_candidate is not None and best_score > 0:
                # Apply the replacement
                for si in old_sub_indices:
                    sub_count[si] -= 1
                for sub in itertools.combinations(best_candidate, t):
                    si = sub_index.get(sub, -1)
                    if si >= 0:
                        sub_count[si] += 1
                blocks[idx] = best_candidate
                improved = True

    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for safety

    # Phase 1: Greedy construction
    blocks = _greedy_construct(v, k, t, deadline)

    if time.time() > deadline:
        return blocks

    # Phase 2: Remove redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, k, t, deadline)

    if time.time() > deadline:
        return blocks

    # Phase 3: Local search - try to find better blocks
    blocks = _local_search(blocks, v, k, t, deadline)

    # Phase 4: Final verification - ensure every t-subset is covered
    # If not valid, fall back to a safe construction
    subs, sub_index, _ = _build_sub_index(v, t)
    covered = [False] * len(subs)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_index.get(sub, -1)
            if idx >= 0:
                covered[idx] = True

    if not all(covered):
        # Fallback: add blocks to cover missing subsets
        missing = [subs[i] for i in range(len(subs)) if not covered[i]]
        for msub in missing:
            # Create a block containing this t-subset
            remaining = [p for p in range(v) if p not in msub]
            extra = random.sample(remaining, k - t)
            block = sorted(list(msub) + extra)
            # Check if this block already exists
            if block not in blocks:
                blocks.append(block)

    return blocks
