"""Covering design C(19,7,4) — greedy construction with time budget and local repair."""
import itertools
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(v, k, t):
    """
    Greedy covering with:
      1) Fast vectorized-ish gain computation using sets
      2) Time budget guard
      3) Local search: try to merge/prune redundant blocks
    """
    deadline = time.time() + 115.0  # safety margin before 150s kill

    # Precompute all t-subsets as frozensets for fast membership
    all_tsubs = list(itertools.combinations(range(v), t))
    uncovered = set(map(frozenset, all_tsubs))

    blocks = []
    # For tracking: set of all t-subsets covered so far
    covered = set()

    while uncovered:
        if time.time() > deadline:
            break

        # Pick the t-subset with fewest candidates? No — greedy: pick first uncovered.
        # To improve: pick the uncovered t-subset that appears in the fewest potential blocks.
        # For speed, just pick the first one.
        first_uncovered = min(uncovered, key=lambda x: (list(x)))
        block = list(first_uncovered)
        block_set = set(block)

        # Fill the block to size k by picking the point that covers the most uncovered t-subsets
        while len(block) < k:
            if time.time() > deadline:
                break

            best_gain = -1
            best_x = -1
            # Precompute: for each candidate x not in block, count how many uncovered t-subsets
            # it would complete
            # A t-subset S is completed by x if S \ {x} is a (t-1)-subset of the current block
            # Actually: S is covered by block if S ⊆ block. Adding x covers S iff
            # (S \ {x}) ⊆ block and x ∉ S, i.e., S contains x and S\{x} ⊆ block.
            # So count = number of uncovered S such that x ∈ S and S\{x} ⊆ block.
            # S\{x} is a (t-1)-set, and it must be a subset of the current block (size len(block)).
            # So we need (t-1)-subsets of block, and for each, check if the t-subset formed by adding x is uncovered.

            # More efficient: for each (t-1)-subset of block, and for each x not in block,
            # form the t-subset and check if uncovered.
            block_list = list(block)
            t_minus_1 = t - 1

            # Build a dict: for each candidate x, count uncovered t-subsets
            gains = {x: 0 for x in range(v) if x not in block_set}
            for sub in itertools.combinations(block_list, t_minus_1):
                sub_set = set(sub)
                for x in gains:
                    if x not in sub_set:
                        new_tsub = frozenset(sub_set | {x})
                        if new_tsub in uncovered:
                            gains[x] += 1

            if best_gain < 0:
                # No improvement possible; just pick any point
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    best_x = candidates[0]
                else:
                    break
            else:
                best_x = max(gains, key=lambda x: gains[x])
                best_gain = gains[best_x]

            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        block_tuple = tuple(block)
        # Only add if not already present
        block_frozenset = frozenset(block)
        if block_frozenset not in {frozenset(b) for b in blocks}:
            blocks.append(block)
            # Update covered/uncovered
            for ts in itertools.combinations(block, t):
                tsf = frozenset(ts)
                covered.add(tsf)
                uncovered.discard(tsf)

    # If we hit the deadline, try to repair: for each remaining uncovered t-subset,
    # add a minimal block covering it
    if uncovered:
        while uncovered:
            if time.time() > deadline:
                break
            first_uncovered = min(uncovered, key=lambda x: (list(x)))
            block = list(first_uncovered)
            block_set = set(block)
            # Fill to size k with points that maximize new coverage
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_x = max(candidates, key=lambda x: sum(
                    1 for rest in itertools.combinations(block, t - 1)
                    if frozenset(set(rest) | {x}) in uncovered
                ))
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                tsf = frozenset(ts)
                covered.add(tsf)
                uncovered.discard(tsf)

    return blocks

# EVOLVE-BLOCK-END
