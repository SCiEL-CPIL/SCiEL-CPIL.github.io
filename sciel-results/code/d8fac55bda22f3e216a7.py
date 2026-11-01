"""Packing-first covering design C(19,7,4) with local search polish."""
import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 138.0

    all_tsets = list(itertools.combinations(range(v), t))
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    n_tsets = len(all_tsets)

    # Precompute: for each block (k-subset), the list of tset indices it covers
    # There are C(19,7) = 50388 possible blocks. Precompute all.
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    n_blocks = len(all_blocks)

    # block_cover[i] = set of tset indices covered by block i
    block_cover = []
    for b in all_blocks:
        covers = [tset_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_cover.append(covers)

    # Convert to numpy arrays for speed
    block_cover_arr = np.array(block_cover, dtype=np.int32)

    def greedy_build(use_packing=False, randomize=False, seed=None):
        rng = random.Random(seed) if seed is not None else None
        covered = np.zeros(n_tsets, dtype=bool)
        blocks = []
        used_blocks = set()

        while not covered.all():
            if time.time() > deadline:
                break
            # Find uncovered tsets
            uncovered = np.where(~covered)[0]
            if len(uncovered) == 0:
                break

            # Score each candidate block: new coverage
            # For speed, only consider blocks that cover at least one uncovered tset
            # Use the first uncovered tset to filter
            if randomize and rng:
                seed_tset_idx = rng.choice(uncovered.tolist())
            else:
                seed_tset_idx = uncovered[0]

            # Candidate blocks: those covering seed_tset_idx
            # Precompute: for each tset, which blocks cover it
            # This is expensive to precompute for all, so do it on the fly with filter

            best_block_idx = -1
            best_score = -1
            best_new_count = -1

            # Iterate over all blocks, compute new coverage
            for bi in range(n_blocks):
                if bi in used_blocks:
                    continue
                covers = block_cover_arr[bi]
                new_count = np.sum(~covered[covers])
                if new_count == 0:
                    continue
                if use_packing:
                    # Prefer blocks with less overlap
                    overlap = np.sum(covered[covers])
                    score = new_count - 0.05 * overlap
                else:
                    score = new_count
                if score > best_score:
                    best_score = score
                    best_new_count = new_count
                    best_block_idx = bi

            if best_block_idx == -1:
                # Should not happen if coverage is incomplete
                break

            blocks.append(all_blocks[best_block_idx])
            used_blocks.add(best_block_idx)
            covered[block_cover_arr[best_block_idx]] = True

        return blocks

    def local_search(blocks, max_rounds=30):
        if not blocks:
            return blocks
        blocks = list(blocks)

        def is_valid(blocks):
            covered = np.zeros(n_tsets, dtype=bool)
            for b in blocks:
                bi = block_to_idx[tuple(b)]
                covered[block_cover_arr[bi]] = True
            return covered.all()

        def remove_redundant(blocks):
            # For each block, check if it's essential
            essential = [True] * len(blocks)
            for i, b in enumerate(blocks):
                bi = block_to_idx[tuple(b)]
                covers = block_cover_arr[bi]
                # Check if any tset in covers is not covered by other blocks
                for idx in covers:
                    found_other = False
                    for j, b2 in enumerate(blocks):
                        if j == i:
                            continue
                        bj = block_to_idx[tuple(b2)]
                        if idx in block_cover_arr[bj]:
                            found_other = True
                            break
                    if not found_other:
                        break
                else:
                    essential[i] = False
            return [b for b, e in zip(blocks, essential) if e]

        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            new_blocks = remove_redundant(blocks)
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
            else:
                break
        return blocks

    # Run multiple strategies
    best_blocks = None
    best_count = float('inf')

    # Standard greedy
    b1 = greedy_build(use_packing=False)
    b1 = local_search(b1, max_rounds=20)
    if len(b1) < best_count:
        best_count = len(b1)
        best_blocks = b1

    # Packing-first
    if time.time() < deadline - 10:
        b2 = greedy_build(use_packing=True)
        b2 = local_search(b2, max_rounds=20)
        if len(b2) < best_count:
            best_count = len(b2)
            best_blocks = b2

    # Randomized
    for seed in [42, 123, 7]:
        if time.time() > deadline - 15:
            break
        b3 = greedy_build(use_packing=True, randomize=True, seed=seed)
        b3 = local_search(b3, max_rounds=10)
        if len(b3) < best_count:
            best_count = len(b3)
            best_blocks = b3

    if best_blocks is None:
        best_blocks = greedy_build(use_packing=False)
        best_blocks = local_search(best_blocks, max_rounds=10)

    # Final polish
    if time.time() < deadline - 5:
        best_blocks = local_search(best_blocks, max_rounds=20)

    # Verify and fix
    covered = np.zeros(n_tsets, dtype=bool)
    for b in best_blocks:
        bi = block_to_idx[tuple(b)]
        covered[block_cover_arr[bi]] = True
    if not covered.all():
        for i in range(n_tsets):
            if not covered[i]:
                ts = all_tsets[i]
                # Find a block covering this tset
                for bi in range(n_blocks):
                    if i in block_cover_arr[bi]:
                        best_blocks.append(all_blocks[bi])
                        covered[block_cover_arr[bi]] = True
                        break

    return [list(b) for b in best_blocks]
