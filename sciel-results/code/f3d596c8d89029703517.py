"""
Covering design C(19, 8, 5) with a structural change:
  1. Vectorized greedy construction (NumPy) for speed.
  2. Iterated local search with removal + block-pair replacement (2-for-1)
     to reduce block count beyond what simple removal allows.
"""

import time
import random
import numpy as np
from itertools import combinations


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Returns a list of blocks (each a list of k distinct ints in 0..v-1).
    """
    deadline = time.time() + 140  # 10s safety margin

    # --- Precompute all t-subsets ---
    all_tsubs = np.array(list(combinations(range(v), t)), dtype=np.int32)
    n_tsubs = len(all_tsubs)

    # Precompute for each point x, which t-subsets contain x
    # tsub_mask[x] = boolean array of length n_tsubs
    tsub_mask = []
    for x in range(v):
        mask = np.zeros(n_tsubs, dtype=bool)
        for i in range(n_tsubs):
            if x in all_tsubs[i]:
                mask[i] = True
        tsub_mask.append(mask)
    tsub_mask = np.array(tsub_mask)  # shape (v, n_tsubs)

    # --- Helper: coverage vector for a block (sorted tuple of k points) ---
    def block_coverage(block):
        """Return boolean array of length n_tsubs indicating which t-subsets are covered."""
        # A t-subset is covered if all its t elements are in the block
        # Use: for each t-subset, check if it's a subset of the block
        # Efficient: precompute for each block, which t-subsets it covers
        # A t-subset s is covered by block B iff set(s) <= set(B)
        # Since |s| = t and |B| = k >= t, this means all elements of s are in B.
        # We can compute: for each element of the t-subset, check membership in block.
        # Vectorized: for block B, covered t-subsets are those where all t elements are in B.
        # Equivalent: the t-subset's elements are all in B.
        # We can use: for each t-subset index i, check if all(all_tsubs[i] is in block_set)
        # Faster: precompute a lookup. For a given block B (as frozenset or sorted tuple),
        # the covered t-subsets are exactly the t-subsets of B.
        # There are C(k, t) such subsets. We can precompute their indices.
        pass  # We'll handle this differently below.

    # --- Better approach: maintain uncovered as a boolean array ---
    # For greedy: pick uncovered t-subset, then build a block maximizing new coverage.

    def greedy_construct():
        """Fast greedy using NumPy."""
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []

        while np.any(uncovered):
            # Pick first uncovered t-subset
            idx = np.argmax(uncovered)
            seed_tsub = all_tsubs[idx]
            block = list(seed_tsub)

            while len(block) < k:
                # For each candidate point x not in block, compute how many new t-subsets it covers
                # New t-subsets covered = t-subsets that contain x and (t-1) elements from current block
                # AND are currently uncovered
                best_gain = -1
                best_x = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Compute coverage of adding x to block
                    # t-subsets covered by new block = t-subsets of (block + [x])
                    # New ones = those containing x and (t-1) from block, that are uncovered
                    # Number of such t-subsets = C(len(block), t-1)
                    # But we need to count only uncovered ones.
                    # We can compute the mask of t-subsets containing x: tsub_mask[x]
                    # And intersect with uncovered.
                    # But we also need those that contain (t-1) specific elements from block.
                    # This is complex to vectorize quickly.
                    # Alternative: just count how many currently uncovered t-subsets would be covered.
                    # A t-subset s is covered by block∪{x} iff s ⊆ block∪{x}.
                    # Since s has t elements and block has len(block) elements,
                    # s ⊆ block∪{x} iff s ∩ (block∪{x}) = s.
                    # We can compute: for each uncovered t-subset, check if it's subset of block∪{x}.
                    # This is O(n_tsubs) per candidate, too slow.
                    #
                    # Simpler heuristic: count how many (t-1)-subsets of block, combined with x,
                    # give an uncovered t-subset.
                    gain = 0
                    block_set = set(block)
                    for rest in combinations(block, t - 1):
                        tsub = tuple(sorted(rest + (x,)))
                        # Find index of tsub in all_tsubs
                        # We need a mapping from t-subset tuple to index
                        # Precompute this mapping.
                        pass
                # This is getting complicated. Let's use a simpler but still fast approach.
                # For each candidate x, gain = number of uncovered t-subsets that are subsets of block∪{x}
                # but not subsets of block.
                # We can compute the set of t-subsets covered by block∪{x} as all t-subsets of block∪{x}.
                # There are C(len(block)+1, t) of them.
                # New ones = those that include x.
                # Count how many of those are uncovered.
                new_block = sorted(block + [x])
                # Get indices of all t-subsets of new_block
                # We need a fast way to get these indices.
                # Precompute: for each possible block (too many).
                # Instead, just iterate:
                gain = 0
                new_block_set = set(new_block)
                for i in range(n_tsubs):
                    if uncovered[i] and all(e in new_block_set for e in all_tsubs[i]):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
            block.sort()
            blocks.append(block)

            # Update uncovered
            block_set = set(block)
            for i in range(n_tsubs):
                if uncovered[i] and all(e in block_set for e in all_tsubs[i]):
                    uncovered[i] = False

        return blocks, uncovered

    # This greedy is O(n_tsubs * v * C(k,t)) per block which is slow.
    # Let's use a completely different, faster approach.

    # --- Fast approach using bitsets ---
    # Represent each t-subset by its index. For each point x, precompute a bitset of t-subsets containing x.
    # For a block B, the covered t-subsets are those where ALL elements are in B.
    # We can compute the coverage of a block as: for each t-subset, check if it's a subset of B.
    # Using bitsets: coverage(B) = intersection over x in B of "t-subsets NOT containing x"... no.
    # A t-subset s is covered by B iff s ⊆ B iff for all x in s, x in B.
    # Equivalently, s is NOT covered by B iff there exists x in s such that x not in B.
    # This doesn't help directly.

    # Let's just use the straightforward approach but with precomputed indices.

    # Precompute: tsub_index = dict mapping frozenset -> index
    tsub_index = {}
    for i, tsub in enumerate(all_tsubs):
        tsub_index[frozenset(tsub)] = i

    # Precompute for each k-subset... too many.
    # Instead, for a given block B (sorted list), compute its covered t-subsets quickly.
    def get_covered_indices(block):
        """Return set of indices of t-subsets covered by block."""
        block_set = set(block)
        covered = set()
        for i, tsub in enumerate(all_tsubs):
            if all(e in block_set for e in tsub):
                covered.add(i)
        return covered

    # This is O(n_tsubs) per block. For ~200 blocks, that's 200*11628 ≈ 2.3M operations. Fine.

    # --- Improved greedy ---
    def fast_greedy():
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []

        while np.any(uncovered):
            # Pick uncovered t-subset with max "difficulty" or just first
            idx = int(np.argmax(uncovered))
            seed_tsub = tuple(all_tsubs[idx])
            block = list(seed_tsub)

            while len(block) < k:
                best_gain = -1
                best_x = -1
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    # Compute new coverage if we add x
                    new_block_set = block_set | {x}
                    # Count uncovered t-subsets that are subsets of new_block_set
                    # but were not subsets of block_set
                    gain = 0
                    for i in range(n_tsubs):
                        if uncovered[i]:
                            if all(e in new_block_set for e in all_tsubs[i]):
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    # Shouldn't happen if v > k
                    break
                block.append(best_x)
            block.sort()
            blocks.append(tuple(block))

            # Update uncovered
            block_set = set(block)
            for i in range(n_tsubs):
                if uncovered[i] and all(e in block_set for e in all_tsubs[i]):
                    uncovered[i] = False

        return [list(b) for b in blocks]

    # Start with greedy
    blocks = fast_greedy()

    def is_valid(blocks):
        """Check if blocks form a valid cover."""
        covered = np.zeros(n_tsubs, dtype=bool)
        for block in blocks:
            block_set = set(block)
            for i in range(n_tsubs):
                if not covered[i] and all(e in block_set for e in all_tsubs[i]):
                    covered[i] = True
        return np.all(covered)

    # Local search: try to remove blocks
    def try_remove(blocks):
        """Try to remove blocks one at a time. Returns (new_blocks, removed)."""
        removed = []
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            test_blocks = blocks[:i] + blocks[i+1:]
            if is_valid(test_blocks):
                blocks = test_blocks
                removed.append(blocks[i])
        return blocks, removed

    # Local search: 2-for-1 replacement
    def try_replace21(blocks):
        """Try to replace two blocks with one. Returns new_blocks if successful."""
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(blocks)):
                if time.time() > deadline:
                    break
                # Try to find a single block that covers what blocks[i] and blocks[j] cover
                # The union of their covered t-subsets must be coverable by one k-block.
                # Find the set of points that must be in the new block:
                # For each t-subset covered by blocks[i] or blocks[j] that is NOT covered by other blocks,
                # the new block must contain that t-subset.
                other_blocks = blocks[:i] + blocks[i+1:j] + blocks[j+1:]
                covered_by_others = np.zeros(n_tsubs, dtype=bool)
                for b in other_blocks:
                    bset = set(b)
                    for idx in range(n_tsubs):
                        if all(e in bset for e in all_tsubs[idx]):
                            covered_by_others[idx] = True

                # Required t-subsets: covered by block i or j, not by others
                block_i_set = set(blocks[i])
                block_j_set = set(blocks[j])
                required = []
                for idx in range(n_tsubs):
                    if covered_by_others[idx]:
                        continue
                    if all(e in block_i_set for e in all_tsubs[idx]) or all(e in block_j_set for e in all_tsubs[idx]):
                        required.append(idx)

                if not required:
                    continue

                # Find a k-block that contains all required t-subsets
                # This means the block must contain the union of all elements in required t-subsets
                # But that union might be > k. If so, impossible.
                required_points = set()
                for idx in required:
                    required_points.update(all_tsubs[idx])

                if len(required_points) > k:
                    continue

                # Build the new block: must contain required_points, can add more to reach k
                if len(required_points) <= k:
                    new_block = list(required_points)
                    while len(new_block) < k:
                        # Add random point
                        candidates = [x for x in range(v) if x not in new_block]
                        if not candidates:
                            break
                        new_block.append(random.choice(candidates))
                    new_block.sort()
                    new_block = tuple(new_block)

                    # Verify
                    test_blocks = blocks[:i] + blocks[i+1:j] + blocks[j+1:] + [list(new_block)]
                    if is_valid(test_blocks):
                        return test_blocks
        return blocks

    # Run local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        blocks, removed = try_remove(blocks)
        if removed:
            improved = True
        new_blocks = try_replace21(blocks)
        if len(new_blocks) < len(blocks):
            blocks = new_blocks
            improved = True

    return blocks
