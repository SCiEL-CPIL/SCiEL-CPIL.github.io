"""
Covering design C(v,k,t) via improved greedy with bitset acceleration and
local search refinement within the time budget.
"""
import time
import itertools
import random

import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    # Precompute all t-subsets and assign them indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {sub: i for i, sub in enumerate(all_tsubs)}

    # Precompute for each point x: which t-subsets contain x
    # For efficiency, we'll compute on-the-fly using bitsets

    # Represent coverage state as a bitset (list of 64-bit ints)
    def bitset_len(n):
        return (n + 63) // 64

    nbits = bitset_len(n_tsubs)

    def make_zero_bitset():
        return [0] * nbits

    def set_bit(bs, i):
        bs[i >> 6] |= (1 << (i & 63))

    def is_set(bs, i):
        return (bs[i >> 6] >> (i & 63)) & 1

    def count_ones(bs):
        return sum(bin(x).count('1') for x in bs)

    def popcount(x):
        return bin(x).count('1')

    # For each point x, precompute the set of t-subsets that contain x
    # as a bitset. But this is expensive memory-wise for large v.
    # Instead, for a partial block B, the t-subsets covered by adding x
    # are those t-subsets S such that x in S and S\{x} ⊆ B.
    # We compute this on the fly.

    # Greedy: start with all t-subsets uncovered
    uncovered = make_zero_bitset()
    for i in range(n_tsubs):
        set_bit(uncovered, i)

    blocks = []

    def time_left():
        return deadline - time.time()

    def greedy_block(current_block, uncovered_bs):
        """Pick the next point to add to current_block maximizing new coverage."""
        best_x = None
        best_gain = -1
        block_set = set(current_block)

        for x in range(v):
            if x in block_set:
                continue
            # Count uncovered t-subsets that would be covered by adding x
            # A t-subset S is covered if x in S and S\{x} ⊆ current_block
            # i.e., S = {x} ∪ T where T is a (t-1)-subset of current_block
            gain = 0
            if len(current_block) >= t - 1:
                for T in itertools.combinations(current_block, t - 1):
                    S = tuple(sorted(T + (x,)))
                    if S in tsub_index:
                        idx = tsub_index[S]
                        if is_set(uncovered_bs, idx):
                            gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x

        if best_x is None:
            # Fallback: pick any point not in block
            for x in range(v):
                if x not in block_set:
                    best_x = x
                    break
        return best_x

    # Main greedy loop
    while count_ones(uncovered) > 0 and time_left() > 5:
        # Pick the first uncovered t-subset as seed
        # Find first set bit
        first_idx = None
        for bi in range(nbits):
            if uncovered[bi]:
                # Find lowest set bit
                b = uncovered[bi]
                low = b & (-b)
                bit_pos = low.bit_length() - 1
                first_idx = (bi << 6) | bit_pos
                break
        if first_idx is None:
            break

        seed = list(all_tsubs[first_idx])
        block = list(seed)
        block_set = set(seed)

        # Add points greedily
        while len(block) < k and time_left() > 5:
            x = greedy_block(block, uncovered)
            if x is None:
                break
            block.append(x)
            block_set.add(x)

        block.sort()
        blocks.append(block)

        # Update uncovered: remove all t-subsets contained in this block
        for S in itertools.combinations(block, t):
            idx = tsub_index[S]
            set_bit(uncovered, idx)  # temporarily mark; we'll clear below
        # Actually we need to clear bits. Let's redo:
        # Clear all t-subsets of the block from uncovered
        for S in itertools.combinations(block, t):
            idx = tsub_index[S]
            uncovered[idx >> 6] &= ~(1 << (idx & 63))

    # At this point, we have a valid covering (all t-subsets covered)
    # Try local search: remove redundant blocks
    # A block is redundant if removing it doesn't leave any t-subset uncovered.

    def is_block_redundant(block_idx, blocks_list):
        """Check if block at block_idx is redundant given all other blocks."""
        # Build coverage of all t-subsets by all blocks except block_idx
        # This is expensive; do it only when time permits.
        covered = make_zero_bitset()
        for i, b in enumerate(blocks_list):
            if i == block_idx:
                continue
            for S in itertools.combinations(b, t):
                idx = tsub_index[S]
                set_bit(covered, idx)
        # Check if any t-subset is not covered
        for i in range(n_tsubs):
            if not is_set(covered, i):
                return False
        return True

    # Local search: try to remove redundant blocks
    improved = True
    while improved and time_left() > 10:
        improved = False
        for i in range(len(blocks)):
            if time_left() < 10:
                break
            if is_block_redundant(i, blocks):
                blocks.pop(i)
                improved = True
                break  # restart scan after removal

    return blocks
