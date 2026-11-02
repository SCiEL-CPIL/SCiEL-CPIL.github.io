# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using randomized greedy with
    redundancy removal. Uses a deadline guard to ensure we return within time limits.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin within 120s limit

    # Precompute all t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)

    # Map each triple to its index for fast lookup
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}

    # For each point, precompute which t-subsets it participates in
    # point_triples[p] = list of triple indices that contain point p
    point_triples = [[] for _ in range(v)]
    for i, triple in enumerate(all_triples):
        for p in triple:
            point_triples[p].append(i)

    def greedy_cover(seed_triple_idx, rng, max_blocks=None):
        """
        Greedy covering starting from a seed triple.
        Returns list of blocks (each a sorted list of points).
        """
        # Track coverage: uncovered[i] = True if triple i is not yet covered
        uncovered = bytearray([1] * n_triples)
        uncovered_count = n_triples
        blocks = []

        if max_blocks is None:
            max_blocks = n_triples // (t * (t - 1) // 2) + 100  # generous upper bound

        # Start with the seed triple
        uncovered[seed_triple_idx] = 0
        uncovered_count -= 1

        while uncovered_count > 0 and len(blocks) < max_blocks:
            if time.time() > deadline - 5:
                break  # Stop if we're running out of time

            # Find an uncovered triple to start a new block
            # Pick the first uncovered triple (or a random one for diversity)
            # For speed, we'll scan for the first uncovered
            start_idx = -1
            for i in range(n_triples):
                if uncovered[i]:
                    start_idx = i
                    break
            if start_idx == -1:
                break

            seed_triple = all_triples[start_idx]
            block = list(seed_triple)
            uncovered[start_idx] = 0
            uncovered_count -= 1

            # Extend the block to size k by greedily adding points
            while len(block) < k:
                if time.time() > deadline - 5:
                    break

                best_point = -1
                best_gain = 0

                # For each candidate point not in block, compute gain
                for p in range(v):
                    if p in block:
                        continue

                    # Gain = number of uncovered t-subsets that would be covered
                    # by adding point p to the block
                    # These are t-subsets that contain p and (t-1) points from block
                    gain = 0
                    for combo in itertools.combinations(block, t - 1):
                        triple = tuple(sorted(combo + (p,)))
                        idx = triple_to_idx.get(triple)
                        if idx is not None and uncovered[idx]:
                            gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_point = p

                if best_point == -1 or best_gain == 0:
                    # No beneficial point; pick a random one to complete block
                    candidates = [p for p in range(v) if p not in block]
                    if not candidates:
                        break
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block.sort()

                # Update uncovered
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx.get(triple)
                    if idx is not None and uncovered[idx]:
                        uncovered[idx] = 0
                        uncovered_count -= 1

            if len(block) == k:
                blocks.append(tuple(block))

        return blocks

    def remove_redundant_blocks(blocks):
        """
        Remove blocks that don't cover any unique triple.
        Returns a new list of blocks with redundancies removed.
        """
        if not blocks:
            return []

        n = len(blocks)
        # For each triple, track which blocks cover it
        triple_block_count = defaultdict(int)
        triple_blocks = defaultdict(list)

        for bi, block in enumerate(blocks):
            for triple in itertools.combinations(block, t):
                triple_block_count[triple] += 1
                triple_blocks[triple].append(bi)

        # A block is redundant if every triple it covers is also covered by another block
        # We iteratively remove redundant blocks
        active = [True] * n
        removed = [False] * n

        # Count how many unique triples each block covers (triples covered only by this block)
        unique_triple_count = [0] * n
        for triple, blocks_list in triple_blocks.items():
            if len(blocks_list) == 1:
                unique_triple_count[blocks_list[0]] += 1

        # Iteratively remove blocks with no unique triples
        changed = True
        while changed:
            changed = False
            for bi in range(n):
                if not active[bi]:
                    continue
                if unique_triple_count[bi] == 0:
                    # Remove this block
                    active[bi] = False
                    removed[bi] = True
                    changed = True
                    # Update unique_triple_count for other blocks
                    for triple in itertools.combinations(blocks[bi], t):
                        bl = triple_blocks[triple]
                        # Remove bi from bl
                        if bi in bl:
                            bl.remove(bi)
                        if len(bl) == 1:
                            unique_triple_count[bl[0]] += 1
                        # Decrement for all remaining blocks that had this as unique
                        for other_bi in bl:
                            if other_bi != bi and active[other_bi]:
                                # This triple might no longer be unique for other_bi
                                # But we only increment when len becomes 1
                                pass

        # Build result
        result = [blocks[bi] for bi in range(n) if active[bi]]
        return result

    def verify_covering(blocks):
        """Verify that every t-subset is covered by at least one block."""
        covered = set()
        for block in blocks:
            for triple in itertools.combinations(block, t):
                covered.add(triple)
        return len(covered) == n_triples

    # Multi-start randomized greedy
    best_blocks = None
    best_count = float('inf')

    # Try multiple random seeds
    rng = random.Random(42)

    # First, try a deterministic greedy (seed = first triple)
    blocks = greedy_cover(0, rng)
    if blocks:
        cleaned = remove_redundant_blocks(blocks)
        if verify_covering(cleaned):
            best_blocks = cleaned
            best_count = len(cleaned)
        elif verify_covering(blocks):
            best_blocks = blocks
            best_count = len(blocks)

    # Now try randomized starts
    num_attempts = 20
    for attempt in range(num_attempts):
        if time.time() > deadline - 10:
            break

        seed_idx = rng.randint(0, n_triples - 1)
        blocks = greedy_cover(seed_idx, rng)
        if not blocks:
            continue

        cleaned = remove_redundant_blocks(blocks)
        if verify_covering(cleaned):
            if len(cleaned) < best_count:
                best_blocks = cleaned
                best_count = len(cleaned)
        elif verify_covering(blocks):
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)

    # If we didn't find a valid covering, fall back to a simple greedy that always completes
    if best_blocks is None or not verify_covering(best_blocks):
        # Fallback: simple greedy that always picks the max-gain point
        uncovered = bytearray([1] * n_triples)
        uncovered_count = n_triples
        blocks = []

        while uncovered_count > 0:
            if time.time() > deadline - 2:
                break
            # Find first uncovered triple
            start_idx = -1
            for i in range(n_triples):
                if uncovered[i]:
                    start_idx = i
                    break
            if start_idx == -1:
                break

            seed_triple = all_triples[start_idx]
            block = list(seed_triple)
            uncovered[start_idx] = 0
            uncovered_count -= 1

            while len(block) < k:
                if time.time() > deadline - 2:
                    break
                best_point = -1
                best_gain = 0
                for p in range(v):
                    if p in block:
                        continue
                    gain = 0
                    for combo in itertools.combinations(block, t - 1):
                        triple = tuple(sorted(combo + (p,)))
                        idx = triple_to_idx.get(triple)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                if best_point == -1:
                    candidates = [p for p in range(v) if p not in block]
                    best_point = rng.choice(candidates) if candidates else 0
                block.append(best_point)
                block.sort()
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx.get(triple)
                    if idx is not None and uncovered[idx]:
                        uncovered[idx] = 0
                        uncovered_count -= 1

            if len(block) == k:
                blocks.append(tuple(block))

        if verify_covering(blocks):
            best_blocks = blocks
            best_count = len(blocks)

    # Final cleanup: remove redundant blocks from best solution
    if best_blocks:
        cleaned = remove_redundant_blocks(best_blocks)
        if verify_covering(cleaned):
            best_blocks = cleaned

    # Convert tuples to lists for the return interface
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        # Should not happen, but return empty if something went wrong
        return []
# EVOLVE-BLOCK-END
