"""Bitmask-based greedy set cover for covering design C(v,k,t)."""
import time
import itertools
import random

_DEADLINE_MARGIN = 5.0  # seconds to keep before the 150s wall-clock kill


def _tuple_to_mask(tup, v):
    """Convert a tuple of ints in 0..v-1 to an integer bitmask."""
    m = 0
    for x in tup:
        m |= (1 << x)
    return m


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN

    # Precompute all t-subsets as bitmasks and index them
    all_tsubsets = list(itertools.combinations(range(v), t))
    # Map each t-subset mask to its index for fast lookup
    mask_to_idx = {}
    for i, ts in enumerate(all_tsubsets):
        mask = 0
        for x in ts:
            mask |= (1 << x)
        mask_to_idx[mask] = i

    n_tsub = len(all_tsubsets)

    # For each k-subset (block), precompute which t-subsets it covers
    # We'll build a list of (block_mask, covered_tsub_indices) lazily or precompute
    # C(19,7) = 50388 blocks, each covers C(7,4)=35 t-subsets -> ~1.7M entries
    # Precompute all blocks and their coverage as bitsets over t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # coverage[block_idx] = set of t-subset indices covered
    # Using Python sets for flexibility; with n_tsub up to C(19,4)=3876,
    # a Python int as bitset works well
    coverage = [0] * n_blocks
    for bi, blk in enumerate(all_blocks):
        blk_mask = 0
        for x in blk:
            blk_mask |= (1 << x)
        # Find all t-subsets of this block
        for ts in itertools.combinations(blk, t):
            ts_mask = 0
            for x in ts:
                ts_mask |= (1 << x)
            ti = mask_to_idx[ts_mask]
            coverage[bi] |= (1 << ti)

    # Greedy set cover: repeatedly pick the block covering the most uncovered t-subsets
    uncovered = (1 << n_tsub) - 1  # bitset of all t-subsets
    blocks_chosen = []
    used_blocks = [False] * n_blocks

    # For speed, maintain a count of how many uncovered t-subsets each block covers
    # We recompute gains each iteration (n_blocks * n_tsub/word_size operations)
    # With n_blocks=50388, n_tsub=3876, and ~150 iterations, this is ~50388*150 = 7.5M
    # bit operations on ~60-word ints, which should be fast in Python with big ints

    while uncovered:
        # Check deadline
        if time.time() > deadline:
            break

        best_gain = -1
        best_bi = -1
        for bi in range(n_blocks):
            if used_blocks[bi]:
                continue
            gain = bin(coverage[bi] & uncovered).count('1')
            if gain > best_gain:
                best_gain = gain
                best_bi = bi
                if best_gain == 0:
                    break

        if best_bi == -1 or best_gain <= 0:
            # No block covers any uncovered t-subset (shouldn't happen if valid)
            break

        used_blocks[best_bi] = True
        blocks_chosen.append(list(all_blocks[best_bi]))
        uncovered &= ~coverage[best_bi]

    # If we broke due to deadline and still have uncovered, we need to ensure validity
    # The greedy set cover above should complete well within time for C(19,7,4)
    # Let's verify: ~150 blocks * 50388 blocks * 60 words ≈ 450M word-ops... 
    # Actually Python big int & and count is in C, so it's fast.
    # But let's add a fallback: if still uncovered, do a final pass

    if uncovered:
        # Fallback: try to cover remaining with a simple greedy on points
        # This should rarely trigger
        remaining = []
        for ti in range(n_tsub):
            if uncovered & (1 << ti):
                remaining.append(all_tsubsets[ti])
        
        # Simple greedy for remaining
        for ts in remaining:
            if time.time() > deadline:
                break
            # Find a block containing this t-subset that covers most remaining
            ts_mask = 0
            for x in ts:
                ts_mask |= (1 << x)
            best_bi = -1
            best_gain = -1
            for bi in range(n_blocks):
                if used_blocks[bi]:
                    continue
                if coverage[bi] & (1 << mask_to_idx[ts_mask]):
                    gain = bin(coverage[bi] & uncovered).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_bi = bi
            if best_bi != -1:
                used_blocks[best_bi] = True
                blocks_chosen.append(list(all_blocks[best_bi]))
                uncovered &= ~coverage[best_bi]

    # Ensure blocks are distinct and sorted
    seen = set()
    final_blocks = []
    for b in blocks_chosen:
        b_sorted = sorted(b)
        key = tuple(b_sorted)
        if key not in seen:
            seen.add(key)
            final_blocks.append(b_sorted)

    return final_blocks
