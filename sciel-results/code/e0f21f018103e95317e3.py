"""Simulated annealing covering design C(19,7,4) solver.

Strategy:
1. Fast greedy construction with incremental coverage counting.
2. Redundant block removal.
3. Simulated annealing: state is the set of blocks; moves include
   removing a block (if coverage stays complete), swapping a block for
   another that covers its unique subsets, and occasionally adding a block.
   Accept worse solutions probabilistically to escape local minima.
4. Deadline guard to return best result before timeout.
"""

import itertools
import time
import random
import math
from collections import defaultdict


def _build_sub_index(v, t):
    """Map each t-subset to a unique index; return (list_of_subsets, sub_index, point_sub_map)."""
    subs = list(itertools.combinations(range(v), t))
    sub_index = {s: i for i, s in enumerate(subs)}
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
    uncovered = set(range(n_subs))
    blocks = []

    while uncovered and time.time() < deadline:
        min_idx = min(uncovered)
        min_sub = subs[min_idx]
        block = list(min_sub)
        block_set = set(block)

        while len(block) < k and time.time() < deadline:
            best_x = -1
            best_gain = -1

            for x in range(v):
                if x in block_set:
                    continue
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

        for sub in itertools.combinations(block, t):
            idx = sub_index.get(sub, -1)
            if idx >= 0:
                uncovered.discard(idx)

    return blocks


def _remove_redundant_blocks(blocks, v, k, t, deadline):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks

    sub_count = defaultdict(int)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            sub_count[sub] += 1

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


def _simulated_annealing(blocks, v, k, t, deadline):
    """Simulated annealing to minimize block count while maintaining coverage."""
    subs, sub_index, point_sub_map = _build_sub_index(v, t)
    n_subs = len(subs)

    # Build coverage counts
    sub_count = [0] * n_subs
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_index.get(sub, -1)
            if idx >= 0:
                sub_count[idx] += 1

    # Ensure all are covered
    all_covered = all(c > 0 for c in sub_count)
    if not all_covered:
        # Add blocks for missing subsets
        for i in range(n_subs):
            if sub_count[i] == 0:
                msub = subs[i]
                remaining = [p for p in range(v) if p not in msub]
                extra = random.sample(remaining, k - t)
                block = sorted(list(msub) + extra)
                blocks.append(block)
                for sub in itertools.combinations(block, t):
                    idx = sub_index.get(sub, -1)
                    if idx >= 0:
                        sub_count[idx] += 1

    best_blocks = blocks[:]
    best_count = len(blocks)
    best_sub_count = sub_count[:]

    # Annealing parameters
    start_temp = 10.0
    end_temp = 0.01
    cooling_rate = 0.995
    temp = start_temp
    rng = random.Random(42)

    while time.time() < deadline and temp > end_temp:
        # Check if we can remove a block
        # Find blocks with no unique subsets
        removable_indices = []
        for i, block in enumerate(blocks):
            needed = False
            for sub in itertools.combinations(block, t):
                idx = sub_index.get(sub, -1)
                if idx >= 0 and sub_count[idx] == 1:
                    needed = True
                    break
            if not needed:
                removable_indices.append(i)

        # Try moves
        move_type = rng.random()
        
        if move_type < 0.3 and removable_indices:
            # Remove a redundant block
            idx = rng.choice(removable_indices)
            old_block = blocks[idx]
            old_subs = list(itertools.combinations(old_block, t))
            
            for sub in old_subs:
                si = sub_index.get(sub, -1)
                if si >= 0:
                    sub_count[si] -= 1
            
            blocks.pop(idx)
            
            # Check validity
            valid = all(c > 0 for c in sub_count)
            if valid:
                new_count = len(blocks)
                if new_count < best_count:
                    best_count = new_count
                    best_blocks = blocks[:]
                    best_sub_count = sub_count[:]
            else:
                # Revert
                blocks.insert(idx, old_block)
                for sub in old_subs:
                    si = sub_index.get(sub, -1)
                    if si >= 0:
                        sub_count[si] += 1
        elif move_type < 0.7:
            # Swap a block with a better one
            if not blocks:
                temp *= cooling_rate
                continue
            idx = rng.randint(0, len(blocks) - 1)
            old_block = blocks[idx]
            old_subs = list(itertools.combinations(old_block, t))
            old_sub_indices = []
            for sub in old_subs:
                si = sub_index.get(sub, -1)
                if si >= 0:
                    old_sub_indices.append(si)

            # Count unique subsets
            unique_count = sum(1 for si in old_sub_indices if sub_count[si] == 1)

            # Generate candidate
            new_block = sorted(rng.sample(range(v), k))
            new_subs = list(itertools.combinations(new_block, t))
            new_sub_indices = []
            for sub in new_subs:
                si = sub_index.get(sub, -1)
                if si >= 0:
                    new_sub_indices.append(si)

            # Compute energy change
            # Energy = number of blocks + penalty for uncovered
            old_energy = len(blocks)
            
            # Simulate swap
            lost_unique = 0
            for si in old_sub_indices:
                if sub_count[si] == 1 and si not in new_sub_indices:
                    lost_unique += 1
            
            gained = 0
            for si in new_sub_indices:
                if sub_count[si] == 0:
                    gained += 1

            # Apply swap temporarily
            for si in old_sub_indices:
                sub_count[si] -= 1
            for si in new_sub_indices:
                sub_count[si] += 1
            
            new_energy = len(blocks)
            uncovered_after = sum(1 for c in sub_count if c == 0)
            
            if uncovered_after > 0:
                # Revert
                for si in new_sub_indices:
                    sub_count[si] -= 1
                for si in old_sub_indices:
                    sub_count[si] += 1
                temp *= cooling_rate
                continue
            
            # Acceptance probability
            delta_e = new_energy - old_energy
            if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
                blocks[idx] = new_block
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = blocks[:]
                    best_sub_count = sub_count[:]
            else:
                # Revert
                for si in new_sub_indices:
                    sub_count[si] -= 1
                for si in old_sub_indices:
                    sub_count[si] += 1
        else:
            # Add a random block (to explore)
            new_block = sorted(rng.sample(range(v), k))
            blocks.append(new_block)
            for sub in itertools.combinations(new_block, t):
                si = sub_index.get(sub, -1)
                if si >= 0:
                    sub_count[si] += 1

            # Check if we can now remove some blocks
            removed_any = False
            for i in range(len(blocks) - 1, -1, -1):
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
                    removed_any = True
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = blocks[:]
                        best_sub_count = sub_count[:]
            
            if not removed_any:
                # Revert the addition
                blocks.pop()
                for sub in itertools.combinations(new_block, t):
                    si = sub_index.get(sub, -1)
                    if si >= 0:
                        sub_count[si] -= 1

        temp *= cooling_rate

    # Restore best state
    blocks[:] = best_blocks
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

    # Phase 3: Simulated annealing
    blocks = _simulated_annealing(blocks, v, k, t, deadline)

    # Phase 4: Final verification - ensure every t-subset is covered
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
            remaining = [p for p in range(v) if p not in msub]
            extra = random.sample(remaining, k - t)
            block = sorted(list(msub) + extra)
            if block not in blocks:
                blocks.append(block)

    return blocks
