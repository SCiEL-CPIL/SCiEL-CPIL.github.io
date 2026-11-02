import itertools
import time
import random
from collections import defaultdict


def _greedy_cover(v, k, t, rng, deadline, mode="pivot"):
    """
    Constructs a covering using a greedy heuristic with randomization.
    mode: 'pivot' (random uncovered triple as seed) or 'most_uncovered'
    """
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []

    while uncovered:
        if time.time() > deadline:
            break

        if mode == "pivot":
            start_triple = rng.choice(tuple(uncovered))
        else:
            # Pick the uncovered triple that appears in the most potential blocks
            # (i.e., most "central") - here just pick min for determinism bias
            start_triple = min(uncovered)

        block = list(start_triple)
        block_set = set(block)

        # Greedily add points to the block
        while len(block) < k:
            if time.time() > deadline:
                break

            best_gain = -1
            best_points = []

            for x in range(v):
                if x in block_set:
                    continue

                gain = 0
                for a, b in itertools.combinations(block, 2):
                    triple = tuple(sorted((a, b, x)))
                    if triple in uncovered:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain:
                    best_points.append(x)

            if not best_points:
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    x = rng.choice(candidates)
                else:
                    break
            else:
                x = rng.choice(best_points)

            block.append(x)
            block_set.add(x)

        block.sort()
        blocks.append(tuple(block))

        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)

    return blocks


def _reduce_covering(blocks, v, k, t):
    """Remove redundant blocks using cover counts."""
    if not blocks:
        return blocks

    triple_cover_count = defaultdict(int)
    for block in blocks:
        for triple in itertools.combinations(block, t):
            triple_cover_count[tuple(triple)] += 1

    changed = True
    while changed:
        changed = False
        new_blocks = []
        for block in blocks:
            is_essential = False
            for triple in itertools.combinations(block, t):
                if triple_cover_count[tuple(triple)] == 1:
                    is_essential = True
                    break

            if is_essential:
                new_blocks.append(block)
            else:
                changed = True
                for triple in itertools.combinations(block, t):
                    triple_cover_count[tuple(triple)] -= 1

        blocks = new_blocks

    return blocks


def _local_search(blocks, v, k, t, rng, deadline):
    """
    Local search: try to replace a block with a different one that keeps
    coverage but allows removing another block. Also try point swaps.
    Returns potentially improved list of blocks.
    """
    if not blocks:
        return blocks

    n = len(blocks)
    # Precompute triples per block
    block_triples = [list(itertools.combinations(b, t)) for b in blocks]

    # Cover count
    triple_cover_count = defaultdict(int)
    for trs in block_triples:
        for triple in trs:
            triple_cover_count[tuple(triple)] += 1

    # Total number of t-subsets
    total = len(list(itertools.combinations(range(v), t)))

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # Pass 1: try removing redundant blocks (reduction)
        new_blocks = []
        for i, block in enumerate(blocks):
            is_essential = False
            for triple in block_triples[i]:
                if triple_cover_count[tuple(triple)] == 1:
                    is_essential = True
                    break
            if is_essential:
                new_blocks.append(block)
            else:
                for triple in block_triples[i]:
                    triple_cover_count[tuple(triple)] -= 1
                improved = True
        if improved:
            blocks = new_blocks
            block_triples = [list(itertools.combinations(b, t)) for b in blocks]
            continue

        # Pass 2: try to merge two blocks or swap points to reduce count
        # Try: for a random block, try replacing one point with another
        # and see if we can then drop a now-redundant block.
        n = len(blocks)
        if n < 2:
            break

        tried = 0
        max_tries = 200
        while tried < max_tries and time.time() < deadline:
            tried += 1
            i = rng.randrange(n)
            block = list(blocks[i])
            # pick a point to swap out
            j = rng.randrange(k)
            old_point = block[j]
            # pick a new point
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                continue
            new_point = rng.choice(candidates)

            new_block = block[:j] + [new_point] + block[j+1:]
            new_block.sort()
            new_block = tuple(new_block)
            if new_block in blocks:
                continue

            # Tentatively replace
            old_triples = set(itertools.combinations(blocks[i], t))
            new_triples = set(itertools.combinations(new_block, t))

            # Check: would any triple lose its last cover?
            can_replace = True
            for triple in old_triples:
                if triple in new_triples:
                    continue
                if triple_cover_count[tuple(triple)] == 1:
                    can_replace = False
                    break
            if not can_replace:
                continue

            # Apply replacement
            for triple in old_triples:
                triple_cover_count[tuple(triple)] -= 1
            for triple in new_triples:
                triple_cover_count[tuple(triple)] += 1

            blocks[i] = new_block
            block_triples[i] = list(new_triples)

            # Now try to remove a redundant block
            for m in range(n):
                if m == i:
                    continue
                is_essential = False
                for triple in block_triples[m]:
                    if triple_cover_count[tuple(triple)] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    # Remove block m
                    for triple in block_triples[m]:
                        triple_cover_count[tuple(triple)] -= 1
                    blocks.pop(m)
                    block_triples.pop(m)
                    n -= 1
                    improved = True
                    break

    return [list(b) for b in blocks]


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    best_blocks = None
    best_size = float('inf')

    # Strategy 1: randomized greedy with pivot mode
    for seed in [42, 1, 2, 3, 7, 13]:
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        blocks = _greedy_cover(v, k, t, rng, deadline, mode="pivot")
        if time.time() > deadline:
            break
        blocks = _reduce_covering(blocks, v, k, t)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks

    # Strategy 2: local search on the best
    if best_blocks is not None and time.time() < deadline:
        rng = random.Random(99)
        improved = _local_search(best_blocks, v, k, t, rng, deadline)
        if len(improved) < best_size:
            best_size = len(improved)
            best_blocks = improved

    # Strategy 3: more randomized greedy runs with local search
    seed_counter = 100
    while time.time() < deadline:
        seed_counter += 1
        rng = random.Random(seed_counter)
        mode = "pivot" if rng.random() < 0.7 else "most_uncovered"
        blocks = _greedy_cover(v, k, t, rng, deadline, mode=mode)
        if time.time() > deadline:
            break
        blocks = _reduce_covering(blocks, v, k, t)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            if best_size <= 69:
                break
        # Apply local search to this candidate if time allows
        if time.time() < deadline - 5:
            rng2 = random.Random(seed_counter + 1000)
            improved = _local_search(blocks, v, k, t, rng2, deadline - 2)
            if len(improved) < best_size:
                best_size = len(improved)
                best_blocks = improved
                if best_size <= 69:
                    break

    return best_blocks
