"""Covering design C(v,k,t) via candidate-pool construction + local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def _triple_indices(triples, v):
    """Map each t-subset (as a sorted tuple) to a compact index 0..T-1."""
    idx = {}
    for i, tr in enumerate(triples):
        idx[tr] = i
    return idx


def construct(v, k, t):
    deadline = time.time() + 115.0
    rng = random.Random(20240517)

    # Precompute all t-subsets and their compact indices
    triples = list(itertools.combinations(range(v), t))
    T = len(triples)
    tidx = _triple_indices(triples, v)

    def block_to_mask(block):
        m = np.zeros(T, dtype=np.int8)
        for tr in itertools.combinations(block, t):
            m[tidx[tr]] = 1
        return m

    # ---- Phase 1: build a valid covering with randomized greedy (fast) ----
    uncovered = np.ones(T, dtype=bool)
    blocks = []
    while uncovered.any():
        first = int(np.argmax(uncovered))
        tr0 = triples[first]
        block = list(tr0)
        while len(block) < k:
            best_x, best_gain = -1, -1
            for x in range(v):
                if x in block:
                    continue
                g = 0
                for r in itertools.combinations(block, t - 1):
                    if tidx[tuple(sorted(r + (x,)))] < T and uncovered[tidx[tuple(sorted(r + (x,)))]]:
                        g += 1
                if g > best_gain:
                    best_gain, best_x = g, x
            block.append(best_x)
        block.sort()
        blocks.append(block)
        uncovered[~uncovered & block_to_mask(block)] = False
        uncovered &= ~block_to_mask(block)

    best_blocks = list(blocks)
    best_count = len(best_blocks)
    best_cov = np.zeros(T, dtype=np.int32)
    for b in best_blocks:
        best_cov += block_to_mask(b)

    # ---- Phase 2: local search (remove / swap) with NumPy coverage ----
    def cov_from_blocks(blks):
        c = np.zeros(T, dtype=np.int32)
        for b in blks:
            c += block_to_mask(b)
        return c

    cur_blocks = list(best_blocks)
    cur_cov = cov_from_blocks(cur_blocks)
    cur_count = len(cur_blocks)

    # Precompute masks for current blocks to speed up swaps
    cur_masks = [block_to_mask(b) for b in cur_blocks]

    def is_valid(cov):
        return bool((cov >= 1).all())

    # Candidate pool for swaps: random 7-subsets
    def random_block():
        return sorted(rng.sample(range(v), k))

    # Local search loop
    while time.time() < deadline - 1.0:
        improved = False

        # --- Removal pass: try to drop any redundant block ---
        for i in range(len(cur_blocks)):
            if time.time() > deadline - 1.0:
                break
            cur_cov[i] = 0  # placeholder; recompute cheaply
            # Actually: check if removing block i keeps all covered
            tmp = cur_cov - cur_masks[i]
            if (tmp >= 1).all():
                # Remove block i
                del cur_blocks[i]
                del cur_masks[i]
                cur_cov = tmp
                cur_count -= 1
                improved = True
                if cur_count < best_count:
                    best_count = cur_count
                    best_blocks = list(cur_blocks)
                    best_cov = cur_cov.copy()
                # Restart removal scan from 0 since indices shifted
                i = -1
        if cur_count < best_count:
            best_count = cur_count
            best_blocks = list(cur_blocks)
            best_cov = cur_cov.copy()

        # --- Swap pass: try replacing a block with a random one ---
        for _ in range(200):
            if time.time() > deadline - 1.0:
                break
            if not cur_blocks:
                break
            i = rng.randrange(len(cur_blocks))
            nb = random_block()
            if nb == cur_blocks[i]:
                continue
            nm = block_to_mask(nb)
            tmp = cur_cov - cur_masks[i] + nm
            if (tmp >= 1).all():
                # Accept the swap
                cur_blocks[i] = nb
                cur_masks[i] = nm
                cur_cov = tmp
                improved = True
                # After a swap, some blocks may now be redundant
                # Try a quick removal of a random few
                for j in range(len(cur_blocks)):
                    tmp2 = cur_cov - cur_masks[j]
                    if (tmp2 >= 1).all():
                        del cur_blocks[j]
                        del cur_masks[j]
                        cur_cov = tmp2
                        cur_count -= 1
                        if cur_count < best_count:
                            best_count = cur_count
                            best_blocks = list(cur_blocks)
                            best_cov = cur_cov.copy()
                        break

    # ---- Phase 3: final cleanup removals on best ----
    best_cov = cov_from_blocks(best_blocks)
    best_masks = [block_to_mask(b) for b in best_blocks]
    changed = True
    while changed and time.time() < deadline - 0.5:
        changed = False
        for i in range(len(best_blocks)):
            tmp = best_cov - best_masks[i]
            if (tmp >= 1).all():
                del best_blocks[i]
                del best_masks[i]
                best_cov = tmp
                changed = True
                break

    return best_blocks
# EVOLVE-BLOCK-END
