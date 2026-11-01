"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: Bitset-vectorized greedy set cover with multi-start + local search.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 115.0
_START = None


def _elapsed():
    return time.time() - _START


def _precompute_bitsets(v, k, t):
    """Precompute all k-subsets and their t-subset coverage as bitsets."""
    all_k = list(itertools.combinations(range(v), k))
    n_k = len(all_k)
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}

    # block_bits[b] = integer where bit i is set if t-subset i is covered by block b
    block_bits = [0] * n_k
    for b, ks in enumerate(all_k):
        bits = 0
        for ts in itertools.combinations(ks, t):
            bits |= (1 << t_index[ts])
        block_bits[b] = bits

    # For local search: t_to_blocks[s] = list of block indices covering t-subset s
    t_to_blocks = [[] for _ in range(n_t)]
    for b in range(n_k):
        bits = block_bits[b]
        for s in range(n_t):
            if bits & (1 << s):
                t_to_blocks[s].append(b)

    return all_k, all_t, block_bits, t_to_blocks, n_t, n_k


def _greedy_bitset(v, k, t, rng, block_bits, n_k, n_t):
    """Bitset-vectorized greedy: select block with max popcount(block & uncovered)."""
    full_mask = (1 << n_t) - 1
    uncovered = full_mask
    selected = []

    while uncovered and _elapsed() < _DEADLINE - 20:
        best_b = -1
        best_gain = 0
        for b in range(n_k):
            gain = (block_bits[b] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_b = b
        if best_b < 0 or best_gain == 0:
            break
        selected.append(best_b)
        uncovered &= ~block_bits[best_b]

    return selected, uncovered


def _local_search_bitset(selected, block_bits, n_k, n_t, rng, time_budget):
    """Local search: removal + swap passes using bitset coverage."""
    selected = set(selected)

    # Compute coverage as bitset union
    def compute_union(blk_set):
        u = 0
        for b in blk_set:
            u |= block_bits[b]
        return u

    full_mask = (1 << n_t) - 1
    union = compute_union(selected)

    improved = True
    while improved and _elapsed() < time_budget:
        improved = False

        # --- Removal pass ---
        blocks_list = list(selected)
        for i in range(len(blocks_list)):
            if _elapsed() >= time_budget:
                break
            b = blocks_list[i]
            if b not in selected:
                continue
            # Can remove b if union without b still covers everything
            # i.e., (union_without_b) == full_mask
            # union_without_b = union & ~block_bits[b] only if b's bits are all covered by others
            # Actually: remove b, recompute union of remaining
            # Faster: check if block_bits[b] is subset of (union ^ block_bits[b]) ... no
            # Correct check: is there any bit in block_bits[b] that is ONLY in b?
            # i.e., (block_bits[b] & ~union_without_b) == 0
            # But computing union_without_b is expensive. Alternative:
            # Remove b, check if new union == full_mask
            selected.discard(b)
            new_union = compute_union(selected)
            if new_union == full_mask:
                # Keep removed
                union = new_union
                improved = True
                blocks_list = list(selected)
            else:
                # Revert
                selected.add(b)

        # --- Swap pass ---
        if _elapsed() >= time_budget:
            break
        for _ in range(80):
            if _elapsed() >= time_budget:
                break
            b = rng.choice(list(selected))
            new_b = rng.randrange(n_k)
            if new_b in selected:
                continue

            # Tentatively swap
            selected.discard(b)
            selected.add(new_b)

            # Try to remove a redundant block
            removed = False
            for cb in list(selected):
                selected.discard(cb)
                new_union = compute_union(selected)
                if new_union == full_mask:
                    union = new_union
                    removed = True
                    break
                else:
                    selected.add(cb)

            if not removed:
                # Revert swap
                selected.discard(new_b)
                selected.add(b)
            else:
                improved = True

    return selected


def _repair(selected, block_bits, t_to_blocks, n_t, full_mask):
    """Add blocks to cover any missing t-subsets."""
    union = 0
    for b in selected:
        union |= block_bits[b]
    missing = full_mask & ~union

    while missing and _elapsed() < _DEADLINE - 2:
        # Find block covering most missing bits
        best_b = -1
        best_gain = 0
        # Sample from candidates
        candidates = set()
        # Pick a few missing bits and get their covering blocks
        m = missing
        while m and len(candidates) < 50:
            # Pick lowest set bit
            lb = m & (-m)
            s = lb.bit_length() - 1
            for b in t_to_blocks[s]:
                candidates.add(b)
            m &= m - 1
        for b in candidates:
            gain = (block_bits[b] & missing).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_b = b
        if best_b < 0 or best_gain == 0:
            break
        selected.add(best_b)
        union |= block_bits[best_b]
        missing = full_mask & ~union

    return selected


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(42)

    all_k, all_t, block_bits, t_to_blocks, n_t, n_k = _precompute_bitsets(v, k, t)
    full_mask = (1 << n_t) - 1

    # Multi-start: run greedy with different random orderings
    best_selected = None
    best_size = float('inf')

    n_starts = 5  # Number of multi-start attempts
    greedy_time_budget = 30.0  # Total time for all greedy starts

    for start_i in range(n_starts):
        if _elapsed() > _DEADLINE - 40:
            break

        # Shuffle block order for diversity
        block_order = list(range(n_k))
        if start_i > 0:
            rng.shuffle(block_order)

        # Run greedy with shuffled order
        uncovered = full_mask
        selected = []
        start_time = time.time()
        max_greedy_time = greedy_time_budget / max(1, n_starts - start_i)

        while uncovered and (time.time() - start_time) < max_greedy_time:
            best_b = -1
            best_gain = 0
            for b in block_order:
                gain = (block_bits[b] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
            if best_b < 0 or best_gain == 0:
                break
            selected.append(best_b)
            uncovered &= ~block_bits[best_b]

        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)

    if best_selected is None:
        # Fallback: single greedy
        selected, _ = _greedy_bitset(v, k, t, rng, block_bits, n_k, n_t)
        best_selected = set(selected)
        best_size = len(selected)

    # Local search on best solution
    time_budget = _DEADLINE - 8
    best_selected = _local_search_bitset(best_selected, block_bits, n_k, n_t, rng, time_budget)

    # Repair if needed
    best_selected = _repair(best_selected, block_bits, t_to_blocks, n_t, full_mask)

    # Return as list of lists
    result = [list(all_k[b]) for b in best_selected]
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
