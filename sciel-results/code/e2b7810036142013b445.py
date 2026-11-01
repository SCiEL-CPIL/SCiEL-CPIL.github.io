"""Bitset-accelerated greedy covering with randomized restarts for C(v,k,t)."""

import time
import random
import itertools

# Precompute mapping from 3-combination to index for v=24
# Number of 3-combinations of 24 = 2024
_C24_3 = 2024


def _combo_to_idx(a, b, c):
    """Map a sorted 3-tuple to an index 0..2023 using combinatorial number system."""
    # For v=24, t=3: index = C(c,3) - C(b,2) + C(a,1) ... 
    # Actually let's use a simpler approach: precompute a dictionary
    pass


# Precompute lookup table for v=24, t=3
_COMBO_INDEX = {}
_idx = 0
for a in range(24):
    for b in range(a + 1, 24):
        for c in range(b + 1, 24):
            _COMBO_INDEX[(a, b, c)] = _idx
            _idx += 1
assert _idx == _C24_3


def _block_covers_indices(block):
    """Return the set (as a frozenset or list) of 3-subset indices covered by a block."""
    idxs = []
    for combo in itertools.combinations(block, 3):
        idxs.append(_COMBO_INDEX[combo])
    return idxs


def _uncovered_bitset(uncovered_set):
    """Convert a set of indices to a bitset (integer)."""
    bits = 0
    for i in uncovered_set:
        bits |= (1 << i)
    return bits


def _bitset_to_set(bits):
    """Convert a bitset (integer) back to a set of indices."""
    s = set()
    i = 0
    while bits:
        if bits & 1:
            s.add(i)
        bits >>= 1
        i += 1
    return s


def _popcount(x):
    return bin(x).count('1')


def construct(v, k, t):
    start = time.time()
    deadline = start + 110  # Safety margin before 120s limit

    # For the specific case v=24, k=7, t=3
    # We use bitset-based greedy with randomized restarts

    total_triples = _C24_3 if (v == 24 and t == 3) else None

    # If not the specific case, fall back to simple greedy
    if total_triples is None:
        return _generic_greedy(v, k, t, start, deadline)

    # Precompute for each point x, for each subset of the block so far,
    # which triple indices would be covered.
    # Actually, let's precompute: for each point x and each (t-1)-subset S,
    # the list of triple indices containing x and S.
    # For t=3: for each x and each pair (a,b) with a<b, the triples {a,b,x} sorted.

    # Precompute: point_pair_to_triple_indices[x][a][b] -> list of triple indices
    # Actually simpler: for each x in 0..23, and each pair (a,b) in the block,
    # compute which triple indices are covered.

    # Let's precompute: for each x, a dict mapping (a,b) sorted pair to list of triple indices
    # where the triple is {a,b,x} (sorted).
    # For t=3, adding point x to a block B covers triples {x, p, q} for p<q in B.
    # The triple indices are _COMBO_INDEX[sorted((x,p,q))].

    # Precompute for each x, and for each unordered pair (p,q) with p!=q, p!=x, q!=x:
    #   triple_idx[x][p][q] = _COMBO_INDEX[tuple(sorted((x,p,q)))]
    # This is 24 * 23 * 22 / 2 = 6072 entries... actually 24*23*22/2 = 6072? No.
    # For each x, pairs (p,q) where p,q in 0..23, p!=x, q!=x, p<q: C(23,2)=253 per x, total 24*253=6072.

    triple_idx = [[[None] * 24 for _ in range(24)] for _ in range(24)]
    for x in range(24):
        for p in range(24):
            if p == x:
                continue
            for q in range(p + 1, 24):
                if q == x:
                    continue
                tri = tuple(sorted((x, p, q)))
                triple_idx[x][p][q] = _COMBO_INDEX[tri]

    # Now, for a block B (sorted list of points), adding point x covers:
    # for each pair (p,q) in B (p<q): triple_idx[x][p][q]
    # The gain is the number of these that are in the uncovered set.

    # Greedy with randomization:
    # 1. Start with empty uncovered (all covered initially? No, start with all uncovered)
    # 2. Actually: start with all 2024 triples uncovered.
    # 3. Repeatedly pick a block that covers the most uncovered triples.

    # For speed, we'll use:
    # - uncovered as a Python integer bitset (2024 bits, fits in ~26 machine words)
    # - For each candidate block, compute its coverage bitset and count overlap with uncovered.

    # Precompute all C(24,7) = 346104 blocks? That's too many to precompute bitsets for all.
    # Instead, use greedy: pick one block at a time.

    # Strategy: multiple randomized greedy restarts.
    # In each restart:
    #   - Start with all triples uncovered
    #   - Greedily add blocks: pick the block (of 7 points from 24) that maximizes newly covered triples
    #   - To speed up, use a heuristic: start from an uncovered triple, then greedily extend to 7 points

    best_blocks = None
    best_count = float('inf')

    rng = random.Random(42)

    max_restarts = 50  # Will be limited by time

    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break

        uncovered = (1 << total_triples) - 1  # All 2024 triples uncovered
        blocks = []

        while uncovered and time.time() < deadline - 3:
            # Pick a random uncovered triple to seed the block
            # Find a random set bit in uncovered
            # For 2024 bits, we can do this efficiently
            # Shift right by random amount and check
            # Better: find the lowest set bit, then use it
            # Or: sample a random position and check if it's set

            # Get a random uncovered triple
            # Method: create a list of set positions? Too slow for 2024.
            # Use: random integer in [0, 2023], check if bit is set; retry if not.
            # Expected ~1 try on average when few are uncovered, more when many.
            # Actually when many are uncovered (early), most positions are set.
            # When few are uncovered (late), we need more tries.
            # Let's use: find the lowest set bit position.
            # For a 2024-bit integer, we can do:
            #   pos = (uncovered & -uncovered).bit_length() - 1
            # This gives the lowest set bit.

            # To add randomness, we can shift:
            #   shifted = uncovered >> (restart * 137 % total_triples)
            #   But that changes which triples are "uncovered" in the shifted view.
            # Simpler: just use the lowest set bit for determinism within a restart,
            # but randomize the initial seed.

            # Actually, let's randomize by permuting the point labels at the start of each restart.
            # This makes the greedy choices different.

            # For now, let's use the lowest uncovered triple as the seed.
            low_bit = uncovered & -uncovered
            seed_triple_idx = low_bit.bit_length() - 1

            # Convert seed_triple_idx back to a triple
            # We need the inverse mapping. Let's precompute it.
            # Actually, we can just pick the first uncovered triple by scanning.
            # But for speed, let's precompute an inverse table.

            # Precompute inverse: idx_to_combo
            idx_to_combo = []
            for combo, idx in _COMBO_INDEX.items():
                idx_to_combo.append(None)
            for combo, idx in _COMBO_INDEX.items():
                idx_to_combo[idx] = combo
            idx_to_combo = tuple(idx_to_combo)  # Make it a tuple for faster access

            seed_triple = idx_to_combo[seed_triple_idx]
            block = list(seed_triple)

            # Greedily extend the block to k=7 points
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Compute gain: number of uncovered triples covered by adding x
                    gain = 0
                    for i in range(len(block)):
                        for j in range(i + 1, len(block)):
                            p = block[i]
                            q = block[j]
                            ti = triple_idx[x][p][q]
                            if ti is not None and (uncovered >> ti) & 1:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    # Early termination if we can't beat best
                block.append(best_x)
                block.sort()

            # Compute the coverage of this block
            block_bits = 0
            for combo in itertools.combinations(block, 3):
                ti = _COMBO_INDEX[combo]
                block_bits |= (1 << ti)

            # Update uncovered
            newly_covered = uncovered & block_bits
            if newly_covered:
                uncovered &= ~block_bits
                blocks.append(block)
            else:
                # This block covers nothing new; skip it (shouldn't happen if we seeded from uncovered)
                pass

            if time.time() > deadline - 3:
                break

        if not uncovered and len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
            if best_count <= 76:
                break  # Can't do better than known best
        elif not uncovered:
            # Complete but not better; still use it if we have nothing
            if best_blocks is None or len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)

    if best_blocks is not None:
        return best_blocks

    # Fallback: if we never completed a covering, return a greedy covering (possibly incomplete)
    # But the task requires validity. So we must ensure we return a valid covering.
    # If time runs out, we should still return the best complete covering found,
    # or complete the current one if possible.
    # Let's do one more greedy pass without time limit check (but with a hard stop).
    uncovered = (1 << total_triples) - 1
    blocks = []
    while uncovered and time.time() < deadline - 1:
        low_bit = uncovered & -uncovered
        seed_triple_idx = low_bit.bit_length() - 1
        seed_triple = idx_to_combo[seed_triple_idx]
        block = list(seed_triple)
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for i in range(len(block)):
                    for j in range(i + 1, len(block)):
                        p = block[i]
                        q = block[j]
                        ti = triple_idx[x][p][q]
                        if ti is not None and (uncovered >> ti) & 1:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
            block.sort()
        block_bits = 0
        for combo in itertools.combinations(block, 3):
            ti = _COMBO_INDEX[combo]
            block_bits |= (1 << ti)
        uncovered &= ~block_bits
        blocks.append(block)

    return blocks


def _generic_greedy(v, k, t, start, deadline):
    """Fallback for non-24-point cases."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered and time.time() < deadline - 5:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
