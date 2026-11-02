import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    # Allow up to 140 seconds to be safe (limit is 150s)
    timeout = 140.0

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)

    # Precompute all blocks and their covered triples
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # block_triples[i] = list of triple indices covered by block i
    block_triples = []
    # triple_to_blocks[j] = list of block indices covering triple j
    triple_to_blocks = [[] for _ in range(num_triples)]

    for i, block in enumerate(all_blocks):
        covered = [triple_to_idx[tup] for tup in itertools.combinations(block, t)]
        block_triples.append(covered)
        for tr_idx in covered:
            triple_to_blocks[tr_idx].append(i)

    best_blocks = None
    best_count = float('inf')

    def greedy_cover(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)

        uncovered = set(range(num_triples))
        blocks = []

        while uncovered:
            # Pick a random uncovered triple
            target_triple_idx = random.choice(list(uncovered))

            # Get candidate blocks covering this triple
            candidates = triple_to_blocks[target_triple_idx]

            best_gain = -1
            best_block_idx = -1

            # Evaluate gain for each candidate
            for bi in candidates:
                gain = 0
                for tr_idx in block_triples[bi]:
                    if tr_idx in uncovered:
                        gain += 1
                        if gain == 35: 
                            break

                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    if best_gain == 35:
                        break

            if best_block_idx != -1:
                blocks.append(all_blocks[best_block_idx])
                # Update uncovered
                for tr_idx in block_triples[best_block_idx]:
                    uncovered.discard(tr_idx)
            else:
                # Should not happen if v, k, t are valid
                break

        return blocks

    def peel(blocks):
        # blocks: list of tuples
        # Returns new blocks
        if not blocks:
            return blocks

        # Count coverage of each triple
        counts = [0] * num_triples
        for b in blocks:
            for tup in itertools.combinations(b, t):
                counts[triple_to_idx[tup]] += 1

        new_blocks = []
        for b in blocks:
            removable = True
            for tup in itertools.combinations(b, t):
                if counts[triple_to_idx[tup]] == 1:
                    removable = False
                    break
            if not removable:
                new_blocks.append(b)
            else:
                # Decrement counts
                for tup in itertools.combinations(b, t):
                    counts[triple_to_idx[tup]] -= 1

        return new_blocks

    def local_search(blocks, max_iters=100):
        """Try to improve the covering by swapping blocks."""
        if not blocks:
            return blocks

        blocks = list(blocks)
        current_count = len(blocks)

        for _ in range(max_iters):
            improved = False

            # Try to remove a block if possible
            for i in range(len(blocks)):
                # Check if removing block i still covers all triples
                # This is expensive, so only do it if we have time
                if time.time() - start_time > timeout - 10:
                    return blocks

                # Quick check: is block i the only one covering any triple?
                removable = True
                for tup in itertools.combinations(blocks[i], t):
                    tr_idx = triple_to_idx[tup]
                    count = 0
                    for j, b in enumerate(blocks):
                        if j == i:
                            continue
                        if tr_idx in block_triples[j]:
                            count += 1
                            if count > 0:
                                break
                    if count == 0:
                        removable = False
                        break

                if removable:
                    blocks.pop(i)
                    current_count -= 1
                    improved = True
                    break

            if improved:
                continue

            # Try to swap a block with another that covers more uncovered triples
            # This is a heuristic: find a block that's "least useful" and try to replace it
            # For simplicity, we'll just try random swaps
            if len(blocks) > 0:
                # Find the block that covers the fewest unique triples (relative to others)
                # This is expensive, so we'll skip for now
                pass

            if not improved:
                break

        return blocks

    # Main Loop
    attempts = 0
    while time.time() - start_time < timeout:
        seed = attempts
        blocks = greedy_cover(seed)
        blocks = peel(blocks)
        blocks = local_search(blocks, max_iters=50)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

        attempts += 1

        # If we are getting close to timeout, break early to ensure return
        if time.time() - start_time > timeout - 5.0:
            break

    if best_blocks is None:
        # Fallback: single greedy run if no time left for loop?
        # The loop should have run at least once.
        blocks = greedy_cover(0)
        best_blocks = peel(blocks)

    return [list(b) for b in best_blocks]
