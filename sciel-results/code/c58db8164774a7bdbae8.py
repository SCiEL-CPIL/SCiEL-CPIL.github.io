"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys
from collections import defaultdict
import numpy as np


def _precompute(v, k, t):
    """Precompute all t-subsets, index map, and per-block t-subset index lists."""
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    sub_to_idx = {sub: i for i, sub in enumerate(all_tsubsets)}
    # For each possible k-block (as a frozenset), the list of t-subset indices it covers
    # We build lazily per block used, but precompute a dict keyed by tuple of sorted points
    return all_tsubsets, n_tsubsets, sub_to_idx


def _block_tsub_idx(block, sub_to_idx):
    """Return list of t-subset indices covered by a given block (sorted tuple of points)."""
    out = []
    for sub in itertools.combinations(block, t_global):
        idx = sub_to_idx.get(sub)
        if idx is not None:
            out.append(idx)
    return out


t_global = 0  # set in construct


def _greedy_cover(v, k, t, deadline, rng, all_tsubsets, n_tsubsets, sub_to_idx):
    """
    Greedy covering: build k-blocks one at a time, each started from a random
    uncovered t-subset, then greedily add points maximizing newly covered t-subsets.
    Uses vectorized gain computation via numpy for speed.
    """
    global t_global
    t_global = t

    uncovered = set(range(n_tsubsets))
    blocks = []

    # Precompute: for each point x, bitmask of t-subset indices containing x
    # Represent as numpy bool array of shape (v, n_tsubsets)
    point_mask = np.zeros((v, n_tsubsets), dtype=bool)
    for idx, sub in enumerate(all_tsubsets):
        for x in sub:
            point_mask[x, idx] = True

    while uncovered:
        if time.time() > deadline:
            break

        # Pick a random uncovered t-subset to anchor the block
        start_idx = rng.choice(list(uncovered))
        start_sub = all_tsubsets[start_idx]
        block = list(start_sub)
        block_set = set(block)

        # Current coverage of t-subsets by partial block:
        # A t-subset S is "potentially completable" by block if S contains all of block's
        # elements that are in S, and the remaining t-|block∩S| elements can be added.
        # For gain when adding point x: S is newly covered iff S ⊆ block∪{x} and S ⊄ block.
        # i.e., S contains x, S ⊆ block∪{x}, and S has at least one element not in block.
        # Since |S|=t and |block|>=t-1 at this stage (block starts with t elements),
        # when |block| >= t: S ⊆ block∪{x} and x∈S means S = (S\{x}) ∪ {x} with S\{x} ⊆ block.
        # So S\{x} is a (t-1)-subset of block.
        # When |block| == t-1 (not possible here since we start with t): skip.

        # For |block| >= t: gain(x) = |{ idx : idx in uncovered, x in all_tsubsets[idx], 
        #   all_tsubsets[idx] \ {x} ⊆ block }|
        # = |{ idx in uncovered : x in sub, sub \ {x} ⊆ block }|

        # Precompute for current block: for each (t-1)-subset R of block, the t-subsets
        # that are R ∪ {x} for some x. We need: for each x not in block, count uncovered
        # t-subsets S with S\{x} ⊆ block, i.e., S\{x} is a (t-1)-subset of block.

        # Efficient approach: 
        # For each x not in block, gain(x) = number of uncovered t-subsets S such that
        # x ∈ S and S \ {x} ⊆ block.
        # S \ {x} ⊆ block means all elements of S except x are in block.
        # So S = {x} ∪ R where R ⊆ block, |R| = t-1.
        # gain(x) = sum over (t-1)-subsets R of block of [ {x}∪R is a t-subset and its idx is in uncovered ]
        # = sum over R in C(block, t-1) of [ idx({x}∪R) in uncovered ]

        # Precompute: for the current block, for each (t-1)-subset R of block, 
        # which x values make {x}∪R a valid t-subset (i.e., x not in R, x < v).
        # Actually {x}∪R is always a t-subset as long as x ∉ R.
        # So for each R (t-1-subset of block), for each x ∉ R (and x not in block for gain purposes),
        # check if idx(sorted(R∪{x})) is in uncovered.

        # This is expensive if done naively. Use numpy:
        # For each x not in block:
        #   gain(x) = count of uncovered t-subsets S with x∈S and S\{x}⊆block
        #   = count of uncovered S with x∈S and (S ⊆ block∪{x}) and (|S∩(complement of block)| = 1, that element is x)
        #   = count of uncovered S with x∈S and S∩(complement of block) = {x}
        #   = count of uncovered S with x∈S and S⊆block∪{x}

        # Using point_mask: S has x in it iff point_mask[x, idx].
        # S ⊆ block∪{x}: all elements of S are in block∪{x}.
        # For a given x, the t-subsets S with S⊆block∪{x} and x∈S:
        #   These are t-subsets of block∪{x} that contain x.
        #   block∪{x} has size |block|+1. t-subsets containing x: C(|block|, t-1) of them.
        #   But we need to count only those that are in uncovered.

        # Vectorized: for each x not in block, compute mask of t-subsets S where:
        #   point_mask[x, idx] AND all elements of S are in block∪{x}
        # The second condition: S ⊆ block∪{x} iff for all y not in block∪{x}, y ∉ S.
        # Equivalent: (union over y not in block∪{x} of NOT point_mask[y, idx]) AND point_mask[x, idx]
        # But that's expensive. Alternative: S ⊆ block∪{x} iff S is a t-subset of block∪{x}.
        # The set of t-subsets of block∪{x} is small: C(|block|+1, t). We can enumerate them.

        # Since |block| ≤ k = 8 and t = 5, C(9,5) = 126 at most. Very manageable.
        # So for each x not in block, enumerate all t-subsets of block∪{x} that contain x,
        # and count how many are in uncovered.

        while len(block) < k:
            if time.time() > deadline:
                break

            best_gain = -1
            best_points = []
            for x in range(v):
                if x in block_set:
                    continue

                # Enumerate t-subsets of block∪{x} that contain x
                # block∪{x} = block + [x], size = len(block)+1
                # t-subsets containing x: choose t-1 from block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    idx = sub_to_idx.get(sub)
                    if idx is not None and idx in uncovered:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain and gain > 0:
                    best_points.append(x)

            if not best_points:
                break

            chosen = rng.choice(best_points)
            block.append(chosen)
            block_set.add(chosen)

        block.sort()
        blocks.append(tuple(block))

        # Remove covered t-subsets from uncovered
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx.get(sub)
            if idx is not None:
                uncovered.discard(idx)

    return blocks, uncovered


def _remove_redundant_blocks(blocks, v, t, deadline, sub_to_idx):
    """Remove redundant blocks iteratively using coverage counts."""
    if not blocks:
        return blocks

    # coverage[i] = count of blocks covering t-subset i
    # Build coverage counts
    n_tsubsets = len(sub_to_idx)
    coverage = np.zeros(n_tsubsets, dtype=np.int32)

    # For each block, list of t-subset indices it covers
    block_tsubs = []
    for block in blocks:
        tsubs = []
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx.get(sub)
            if idx is not None:
                tsubs.append(idx)
        block_tsubs.append(tsubs)
        coverage[tsubs] += 1

    # Iteratively remove redundant blocks
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        new_blocks = []
        new_block_tsubs = []
        for bidx, block in enumerate(blocks):
            # Block is redundant if all its t-subsets have coverage > 1
            redundant = True
            for idx in block_tsubs[bidx]:
                if coverage[idx] <= 1:
                    redundant = False
                    break
            if not redundant:
                new_blocks.append(block)
                new_block_tsubs.append(block_tsubs[bidx])
            else:
                changed = True
                # Decrement coverage for this block's t-subsets
                for idx in block_tsubs[bidx]:
                    coverage[idx] -= 1

        blocks = new_blocks
        block_tsubs = new_block_tsubs

    return blocks


def _local_search(blocks, v, k, t, deadline, rng, sub_to_idx):
    """
    Local search polish: try to drop blocks (redundancy) and swap blocks.
    Returns improved list of blocks.
    """
    if not blocks:
        return blocks

    n_tsubsets = len(sub_to_idx)
    coverage = np.zeros(n_tsubsets, dtype=np.int32)

    # Build block -> t-subset index lists
    block_tsubs = []
    for block in blocks:
        tsubs = []
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx.get(sub)
            if idx is not None:
                tsubs.append(idx)
        block_tsubs.append(tsubs)
        coverage[tsubs] += 1

    # Pass 1: iterative drop of redundant blocks
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        new_blocks = []
        new_block_tsubs = []
        for bidx in range(len(blocks)):
            redundant = True
            for idx in block_tsubs[bidx]:
                if coverage[idx] <= 1:
                    redundant = False
                    break
            if not redundant:
                new_blocks.append(blocks[bidx])
                new_block_tsubs.append(block_tsubs[bidx])
            else:
                improved = True
                for idx in block_tsubs[bidx]:
                    coverage[idx] -= 1
        blocks = new_blocks
        block_tsubs = new_block_tsubs

    # Pass 2: try to replace blocks with better ones (swap)
    # For each block, try replacing it with a random k-block that covers the same or more t-subsets
    if time.time() < deadline - 5:
        for _ in range(200):  # limited swap attempts
            if time.time() > deadline:
                break
            if not blocks:
                break
            bidx = rng.randrange(len(blocks))
            old_block = blocks[bidx]
            old_tsubs = block_tsubs[bidx]
            
            # Try a few random replacements
            replaced = False
            for attempt in range(20):
                # Generate a random k-block
                new_block = tuple(sorted(rng.sample(range(v), k)))
                new_tsubs = []
                for sub in itertools.combinations(new_block, t):
                    idx = sub_to_idx.get(sub)
                    if idx is not None:
                        new_tsubs.append(idx)
                
                # Check: can we replace old_block with new_block without losing coverage?
                # A t-subset idx is safe to lose from old_block if coverage[idx] > 1 OR it's covered by new_block
                safe = True
                for idx in old_tsubs:
                    if coverage[idx] <= 1 and idx not in new_tsubs:
                        safe = False
                        break
                if safe:
                    # Perform swap
                    for idx in old_tsubs:
                        coverage[idx] -= 1
                    for idx in new_tsubs:
                        coverage[idx] += 1
                    blocks[bidx] = new_block
                    block_tsubs[bidx] = new_tsubs
                    replaced = True
                    break
            # If not replaced, just skip

    # Final pass: remove any newly redundant blocks
    if time.time() < deadline:
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            new_blocks = []
            new_block_tsubs = []
            for bidx in range(len(blocks)):
                redundant = True
                for idx in block_tsubs[bidx]:
                    if coverage[idx] <= 1:
                        redundant = False
                        break
                if not redundant:
                    new_blocks.append(blocks[bidx])
                    new_block_tsubs.append(block_tsubs[bidx])
                else:
                    improved = True
                    for idx in block_tsubs[bidx]:
                        coverage[idx] -= 1
            blocks = new_blocks
            block_tsubs = new_block_tsubs

    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    # Precompute
    all_tsubsets, n_tsubsets, sub_to_idx = _precompute(v, k, t)

    best_blocks = None
    best_count = float('inf')

    # Multiple random restarts
    seeds = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        rng = random.Random(seed)
        blocks, uncovered = _greedy_cover(v, k, t, deadline, rng, all_tsubsets, n_tsubsets, sub_to_idx)
        if len(uncovered) == 0:
            # Apply local search polish
            polished = _local_search(blocks, v, k, t, deadline, rng, sub_to_idx)
            if len(polished) < best_count:
                best_count = len(polished)
                best_blocks = polished

    if best_blocks is None:
        # Fallback: no valid covering found
        return []

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
