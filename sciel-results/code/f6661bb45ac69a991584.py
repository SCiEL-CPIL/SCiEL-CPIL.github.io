"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using greedy + redundancy removal."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)

    # Map each t-subset to an index for fast lookup
    tsub_index = {sub: i for i, sub in enumerate(all_tsubsets)}

    # Greedy covering
    uncovered = set(range(n_tsubsets))  # set of indices of uncovered t-subsets
    blocks = []

    while uncovered and time.time() < deadline - 5:
        # Pick the first uncovered t-subset (arbitrary, but deterministic)
        first_uncovered = min(uncovered)
        tsub = all_tsubsets[first_uncovered]
        block = list(tsub)

        while len(block) < k:
            block_set = set(block)
            candidates = [x for x in range(v) if x not in block_set]

            if not candidates:
                break

            # For each candidate, compute how many new t-subsets it covers
            # A t-subset {a1,...,a_{t-1}, x} is covered if {a1,...,a_{t-1}} ⊆ block
            # and the t-subset index is in uncovered
            best_candidate = None
            best_gain = -1

            # Precompute all (t-1)-subsets of current block
            block_tminus1 = list(itertools.combinations(block, t - 1))

            for x in candidates:
                gain = 0
                for rest in block_tminus1:
                    tsub_candidate = tuple(sorted(rest + (x,)))
                    idx = tsub_index.get(tsub_candidate)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = x

            if best_candidate is None:
                break
            block.append(best_candidate)

        block.sort()
        blocks.append(block)

        # Remove covered t-subsets
        for tsub in itertools.combinations(block, t):
            idx = tsub_index[tsub]
            if idx in uncovered:
                uncovered.discard(idx)

    # Redundancy removal: remove blocks whose t-subsets are all covered by other blocks
    if time.time() < deadline - 2:
        # Build coverage count for each t-subset across all blocks
        # For each block, compute the set of t-subset indices it covers
        block_tsubs = []
        for b in blocks:
            tsubs = set()
            for tsub in itertools.combinations(b, t):
                tsubs.add(tsub_index[tsub])
            block_tsubs.append(tsubs)

        # For each t-subset, count how many blocks cover it
        tsub_count = defaultdict(int)
        for tsubs in block_tsubs:
            for idx in tsubs:
                tsub_count[idx] += 1

        # Remove redundant blocks
        kept = [True] * len(blocks)
        for i in range(len(blocks)):
            if not kept[i]:
                continue
            # Check if all t-subsets of block i are covered by other kept blocks
            redundant = True
            for idx in block_tsubs[i]:
                # Count how many kept blocks (other than i) cover idx
                count = 0
                for j in range(len(blocks)):
                    if j == i or not kept[j]:
                        continue
                    if idx in block_tsubs[j]:
                        count += 1
                        if count >= 1:
                            break
                if count == 0:
                    redundant = False
                    break
            if redundant:
                kept[i] = False

        blocks = [blocks[i] for i in range(len(blocks)) if kept[i]]

    return blocks
# EVOLVE-BLOCK-END
