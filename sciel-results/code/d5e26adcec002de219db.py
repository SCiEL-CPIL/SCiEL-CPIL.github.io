import numpy as np
import itertools
import time
import random
from collections import Counter, defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110.0

    def nCr(n, r):
        if r > n or r < 0: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res = res * (n - i) // (i + 1)
        return res

    n_ts = nCr(v, t)
    n_ks = nCr(v, k)

    # Build index maps
    all_ts = list(itertools.combinations(range(v), t))
    ts_idx = {comb: i for i, comb in enumerate(all_ts)}
    all_ks = list(itertools.combinations(range(v), k))
    ks_idx = {comb: i for i, comb in enumerate(all_ks)}

    # For each triple, list of block indices that contain it
    triple_to_blocks = defaultdict(list)
    for bi, block in enumerate(all_ks):
        for sub in itertools.combinations(block, t):
            ti = ts_idx[sub]
            triple_to_blocks[ti].append(bi)

    # Precompute for each block, the list of triple indices it covers
    block_triples = np.zeros((n_ks, nCr(k, t)), dtype=np.int32)
    for bi, block in enumerate(all_ks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triples[bi, j] = ts_idx[sub]

    def time_left():
        return time_limit - (time.time() - start_time)

    def greedy_cover(weights, max_blocks=None):
        """Greedy set cover using weights on triples."""
        covered = np.zeros(n_ts, dtype=bool)
        remaining = n_ts
        blocks = []
        cover_count = np.zeros(n_ts, dtype=np.int32)

        while remaining > 0 and (max_blocks is None or len(blocks) < max_blocks):
            if time_left() < 5.0:
                break
            # Compute gain for each block: sum of weights of newly covered triples
            # Use the triple_to_blocks structure for efficiency
            # For each block, gain = sum of weights of its triples that aren't yet covered
            # This is expensive to do for all blocks, so we use a heuristic:
            # Only consider blocks that contain at least one uncovered triple
            # and score by weighted gain

            # Find uncovered triples
            uncovered = np.where(~covered)[0]
            if len(uncovered) == 0:
                break

            # For each uncovered triple, candidate blocks are triple_to_blocks[ti]
            # Score each candidate block by its weighted gain
            block_scores = {}
            for ti in uncovered:
                for bi in triple_to_blocks[ti]:
                    if bi not in block_scores:
                        # Compute weighted gain of this block
                        subs = block_triples[bi]
                        mask = ~covered[subs]
                        block_scores[bi] = np.sum(weights[subs[mask]])

            if not block_scores:
                break

            # Pick the block with highest score
            best_bi = max(block_scores, key=block_scores.get)
            blocks.append(best_bi)

            subs = block_triples[best_bi]
            newly = ~covered[subs]
            remaining -= np.sum(newly)
            covered[subs] = True
            cover_count[subs] += 1

        return blocks, cover_count

    def column_generation(weights_init, max_iter=50):
        """Column generation: iteratively add blocks based on LP dual weights."""
        # Start with a greedy solution
        blocks, cover_count = greedy_cover(weights_init)
        if not blocks:
            return blocks, cover_count

        # LP dual: weight each triple by 1/cover_count (higher weight for under-covered)
        for it in range(max_iter):
            if time_left() < 10.0:
                break
            # Update dual weights: triples with low coverage get higher weight
            # Use smooth surrogate: weight = 1 / (cover_count + 0.5)
            weights = 1.0 / (cover_count.astype(float) + 0.5)
            # Boost uncovered triples heavily
            weights[cover_count == 0] = 100.0

            # Pricing: for each uncovered triple, find best block
            uncovered = np.where(cover_count == 0)[0]
            if len(uncovered) == 0:
                break

            # Find new blocks that cover uncovered triples with high gain
            new_blocks = []
            for ti in uncovered:
                if time_left() < 10.0:
                    break
                candidates = triple_to_blocks[ti]
                # Score each candidate by weighted gain
                best_bi = None
                best_score = -1
                for bi in candidates:
                    if bi in blocks:
                        continue
                    subs = block_triples[bi]
                    mask = cover_count[subs] == 0
                    score = np.sum(weights[subs[mask]])
                    if score > best_score:
                        best_score = score
                        best_bi = bi
                if best_bi is not None and best_score > 0:
                    new_blocks.append(best_bi)

            if not new_blocks:
                break

            # Add the best new blocks
            for bi in new_blocks[:5]:
                if bi not in blocks:
                    blocks.append(bi)
                    subs = block_triples[bi]
                    cover_count[subs] += 1

        return blocks, cover_count

    def delete_blocks(blocks, cover_count):
        """Remove redundant blocks."""
        improved = True
        while improved:
            if time_left() < 8.0:
                break
            improved = False
            new_blocks = []
            for bi in blocks:
                subs = block_triples[bi]
                # Check if all triples in this block are covered by other blocks
                other = cover_count[subs] - 1
                if np.all(other > 0):
                    improved = True
                    cover_count[subs] -= 1
                else:
                    new_blocks.append(bi)
            blocks = new_blocks
        return blocks, cover_count

    def swap_improve(blocks, cover_count):
        """Try swapping blocks to reduce count while maintaining coverage."""
        # For each pair of blocks, try replacing one with a different block
        # that covers the same uncovered triples
        if time_left() < 10.0:
            return blocks, cover_count

        # Find blocks that are "critical" (contain triples covered only by them)
        critical = []
        for bi in blocks:
            subs = block_triples[bi]
            if np.any(cover_count[subs] == 1):
                critical.append(bi)

        # Try to replace non-critical blocks with better ones
        # This is a simplified version; full swap is expensive
        return blocks, cover_count

    # Main: run column generation with different initial weightings
    best_blocks = None
    best_count = float('inf')

    # Initial weights: uniform
    weights_init = np.ones(n_ts, dtype=float)

    # Run column generation
    blocks, cover_count = column_generation(weights_init, max_iter=30)
    blocks, cover_count = delete_blocks(blocks, cover_count)
    if len(blocks) < best_count:
        best_blocks = list(blocks)
        best_count = len(blocks)

    # Try with different random initial weightings
    for restart in range(5):
        if time_left() < 15.0:
            break
        rng = np.random.RandomState(restart)
        weights_init = rng.rand(n_ts) + 0.1
        blocks, cover_count = column_generation(weights_init, max_iter=20)
        blocks, cover_count = delete_blocks(blocks, cover_count)
        if len(blocks) < best_count:
            best_blocks = list(blocks)
            best_count = len(blocks)

    # Final polish: aggressive deletion
    if best_blocks is not None:
        cover_count = np.zeros(n_ts, dtype=np.int32)
        for bi in best_blocks:
            cover_count[block_triples[bi]] += 1
        best_blocks, cover_count = delete_blocks(best_blocks, cover_count)

        # Verify coverage
        if np.all(cover_count > 0):
            best_count = len(best_blocks)
        else:
            # Repair: add blocks for uncovered triples
            uncovered = np.where(cover_count == 0)[0]
            for ti in uncovered:
                if time_left() < 5.0:
                    break
                for bi in triple_to_blocks[ti]:
                    if bi not in best_blocks:
                        best_blocks.append(bi)
                        cover_count[block_triples[bi]] += 1
                        break

    if best_blocks is None:
        # Fallback: pure greedy
        blocks, _ = greedy_cover(np.ones(n_ts))
        best_blocks = blocks

    return [list(all_ks[i]) for i in best_blocks]
