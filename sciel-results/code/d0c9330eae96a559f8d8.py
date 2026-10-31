"""Greedy covering + local-search refinement for C(19,7,4), with deadline guard."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _bit_index(v, t, comb):
    """Map a t-combination to a linear index (lexicographic rank)."""
    # Precompute C(n, k) table up to v
    C = [[0] * (t + 1) for _ in range(v + 1)]
    for n in range(v + 1):
        C[n][0] = 1
        for k in range(1, t + 1):
            if k > n:
                C[n][k] = 0
            else:
                C[n][k] = C[n - 1][k - 1] + C[n - 1][k]
    def rank(comb):
        r = 0
        for i, c in enumerate(comb):
            # number of t-combs starting with elements < c at position i
            for start in range(c):
                if start + (t - i) - 1 < v:
                    r += C[v - 1 - start][t - i]
        return r
    return rank


def construct(v, k, t):
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s

    # Precompute all t-subsets and their bit positions
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    comb_to_idx = {c: i for i, c in enumerate(all_tsubs)}

    # Precompute for each point x, the bitmask of t-subsets containing x
    point_mask = np.zeros(n_tsubs, dtype=np.uint8)  # per point, we store index list
    point_indices = []
    for x in range(v):
        idxs = [comb_to_idx[c] for c in all_tsubs if x in c]
        point_indices.append(np.array(idxs, dtype=np.int64))

    # uncovered as a set of indices (start with all)
    uncovered_set = set(range(n_tsubs))
    blocks = []

    # Greedy construction
    def make_block():
        # start from the smallest uncovered t-subset (lexicographic min)
        min_comb = min(all_tsubs[i] for i in uncovered_set)
        block = list(min_comb)
        while len(block) < k:
            # find point maximizing new coverage
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # gain = number of uncovered t-subsets that contain x and (block minus one) ... 
                # Actually: t-subsets of (block + x) that are uncovered
                # A t-subset of block+x containing x = x + (t-1)-subset of block
                # A t-subset of block+x not containing x = t-subset of block
                gain = 0
                # t-subsets not containing x: t-subsets of block
                for c in itertools.combinations(block, t):
                    if comb_to_idx[c] in uncovered_set:
                        gain += 1
                # t-subsets containing x: x + (t-1)-subset of block
                for c in itertools.combinations(block, t - 1):
                    cc = tuple(sorted(c + (x,)))
                    if comb_to_idx[cc] in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        block.sort()
        return block

    while uncovered_set:
        if time.time() > deadline - 30:
            break
        block = make_block()
        blocks.append(block)
        # remove covered t-subsets
        for c in itertools.combinations(block, t):
            idx = comb_to_idx[c]
            if idx in uncovered_set:
                uncovered_set.discard(idx)

    if not blocks:
        return blocks

    # Local search: redundancy removal + 1-for-1 swaps
    # Build a map: for each t-subset, which blocks cover it (by index)
    # And per-block set of t-subsets
    def block_tsubs(b):
        return [comb_to_idx[c] for c in itertools.combinations(b, t)]

    block_subsets = [set(block_tsubs(b)) for b in blocks]
    # coverage count
    cover_count = {}
    for s in block_subsets:
        for idx in s:
            cover_count[idx] = cover_count.get(idx, 0) + 1

    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # redundancy removal
        new_blocks = []
        new_block_subsets = []
        # recompute cover_count from current
        cover_count = {}
        for s in block_subsets:
            for idx in s:
                cover_count[idx] = cover_count.get(idx, 0) + 1
        for bi, s in enumerate(block_subsets):
            # check if any idx in s has cover_count == 1
            unique = False
            for idx in s:
                if cover_count.get(idx, 0) == 1:
                    unique = True
                    break
            if not unique:
                # remove this block
                for idx in s:
                    cover_count[idx] -= 1
                improved = True
            else:
                new_blocks.append(blocks[bi])
                new_block_subsets.append(s)
        blocks = new_blocks
        block_subsets = new_block_subsets

        if time.time() > deadline - 10:
            break

        # 1-for-1 swaps: try replacing block i with a new block
        # A good candidate new block covers a unique subset of block i.
        # We'll try: for each block i, find a unique t-subset u of it, build a
        # candidate block around u that also covers other unique subsets of i,
        # and check it doesn't leave any new uncovered subset.
        cover_count = {}
        for s in block_subsets:
            for idx in s:
                cover_count[idx] = cover_count.get(idx, 0) + 1

        for i in range(len(blocks)):
            if time.time() > deadline - 10:
                break
            si = block_subsets[i]
            # unique subsets of block i
            unique_of_i = [idx for idx in si if cover_count.get(idx, 0) == 1]
            if not unique_of_i:
                continue
            # Build candidate block: start with the unique t-subset (as indices -> points)
            # pick the first unique t-subset
            start_comb = all_tsubs[unique_of_i[0]]
            cand = list(start_comb)
            # add points to maximize coverage of unique_of_i and not create new gaps
            while len(cand) < k:
                best_x = -1
                best_score = -1
                for x in range(v):
                    if x in cand:
                        continue
                    # coverage of unique_of_i by cand+x
                    cov_unique = 0
                    for uidx in unique_of_i:
                        uc = all_tsubs[uidx]
                        if x in uc and all(p in cand for p in uc if p != x):
                            cov_unique += 1
                    # also general gain (uncovered after removal) - approximate
                    # penalty: does cand+x fail to cover some t-subset that block i covered
                    # and that no other block covers? That would create a gap.
                    # We'll check fully at the end. For now score by unique coverage.
                    if cov_unique > best_score:
                        best_score = cov_unique
                        best_x = x
                if best_x == -1:
                    break
                cand.append(best_x)
            cand.sort()
            cand_set = set(block_tsubs(cand))
            # Check: every t-subset of block i that is unique must be in cand_set
            ok = True
            for uidx in unique_of_i:
                if uidx not in cand_set:
                    ok = False
                    break
            if not ok:
                continue
            # Check: cand_set must not introduce any t-subset that is currently
            # covered only by block i (i.e., removing i would uncover it) and not in cand.
            # i.e., for idx in si not in cand_set, cover_count[idx] must be > 1
            # and for idx in cand_set not in si, fine (new coverage ok)
            for idx in si:
                if idx not in cand_set and cover_count.get(idx, 0) <= 1:
                    ok = False
                    break
            if not ok:
                continue
            # Accept swap
            blocks[i] = cand
            block_subsets[i] = cand_set
            improved = True
            # recompute cover_count
            cover_count = {}
            for s in block_subsets:
                for idx in s:
                    cover_count[idx] = cover_count.get(idx, 0) + 1

    return blocks
# EVOLVE-BLOCK-END
