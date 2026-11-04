"""Rewritten covering design C(v,k,t) with bitmask greedy + local search."""
import itertools
import time
import random
import numpy as np


def _combo_to_mask(c, v):
    m = 0
    for x in c:
        m |= (1 << x)
    return m


def _mask_to_combo(m, v):
    out = []
    i = 0
    while m:
        if m & 1:
            out.append(i)
        m >>= 1
        i += 1
    return out


def _gen_combos(n, r):
    """Generate all r-combinations of range(n) as bitmasks (generator)."""
    if r == 0:
        yield 0
        return
    if r > n:
        return
    # recursive via itertools for clarity
    for c in itertools.combinations(range(n), r):
        m = 0
        for x in c:
            m |= (1 << x)
        yield m


def _gen_tsubsets_of_block(block_mask, k, t, v):
    """Yield all t-subset bitmasks of the k-subset block_mask."""
    # extract the k points
    pts = []
    m = block_mask
    i = 0
    while m:
        if m & 1:
            pts.append(i)
        m >>= 1
        i += 1
    # generate t-combinations of these k points
    for c in itertools.combinations(pts, t):
        mm = 0
        for x in c:
            mm |= (1 << x)
        yield mm


def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s kill

    # Precompute all t-subset bitmasks and an index map
    all_tsubsets = list(_gen_combos(v, t))
    n_tsub = len(all_tsubsets)
    tset_index = {m: i for i, m in enumerate(all_tsubsets)}

    # Precompute, for each k-block, the list of t-subset indices it covers
    # This is the heavy part but done once.
    block_list = list(itertools.combinations(range(v), k))
    n_blocks = len(block_list)

    # For each block, list of t-subset indices
    block_tsets = []
    for b in block_list:
        bmask = 0
        for x in b:
            bmask |= (1 << x)
        ts = []
        for tm in _gen_tsubsets_of_block(bmask, k, t, v):
            ts.append(tset_index[tm])
        block_tsets.append(ts)

    # Greedy set cover: maintain uncovered count
    # covered_count[i] = how many selected blocks cover t-subset i
    # We want to add blocks that maximize newly covered t-subsets.
    # Use a priority approach: repeatedly pick block covering most uncovered.
    # Maintain uncovered set as a boolean array.
    uncovered = np.ones(n_tsub, dtype=bool)
    remaining = n_tsub
    selected = []  # list of block indices

    # Precompute block_tsets as arrays of indices
    block_tsets_arr = [np.array(ts, dtype=np.int32) for ts in block_tsets]

    # To speed up, we can precompute for each block its tset array.
    # Greedy: each step compute gain = number of uncovered tsets in block.
    # This is O(n_blocks * k_choose_t) per step which is heavy.
    # Instead, maintain for each tset the list of blocks covering it (inverse),
    # and for each block a count of how many of its tsets are still uncovered.
    # When we select a block, we decrement the uncovered flag and update
    # the "gain" of all blocks that share those tsets.

    # Build inverse: for each tset, list of blocks covering it.
    tset_to_blocks = [[] for _ in range(n_tsub)]
    for bi, ts in enumerate(block_tsets):
        for ti in ts:
            tset_to_blocks[ti].append(bi)

    # gain[bi] = number of currently-uncovered tsets in block bi
    gain = np.zeros(n_blocks, dtype=np.int32)
    for bi, ts in enumerate(block_tsets):
        gain[bi] = len(ts)  # initially all uncovered

    # We'll use a simple loop: pick argmax gain, select it, update.
    # To update efficiently: when we select block bi, for each tset ti in it
    # that is uncovered, we set uncovered[ti]=False and for every block bj
    # in tset_to_blocks[ti], decrement gain[bj].
    # This is O(sum over selected blocks of (k_choose_t * avg_blocks_per_tset)).
    # avg_blocks_per_tset = n_blocks * C(k,t) / C(v,t). For v=19,k=8,t=5:
    # n_blocks = C(19,8)=75582, C(8,5)=56, C(19,5)=11628 => 75582*56/11628 ≈ 364.
    # So per tset we update ~364 blocks. Per block we touch 56 tsets => ~20k updates.
    # For ~300 selected blocks => ~6M updates. Feasible.

    # Use a max-heap? Gain changes dynamically. Simpler: scan for max each step.
    # n_blocks=75582, scanning 300 times => 22M ops, OK.

    for _ in range(n_tsub):  # at most n_tsub steps
        if time.time() > deadline:
            break
        if remaining == 0:
            break
        # pick block with max gain
        bi = int(np.argmax(gain))
        if gain[bi] <= 0:
            break
        selected.append(bi)
        # update
        ts = block_tsets_arr[bi]
        for ti in ts:
            if uncovered[ti]:
                uncovered[ti] = False
                remaining -= 1
                # update gains of all blocks covering ti
                for bj in tset_to_blocks[ti]:
                    gain[bj] -= 1
        gain[bi] = 0  # mark as used (we won't reselect)

    if remaining > 0:
        # Fallback: if we ran out of time, do a naive completion
        # Find uncovered tsets and add blocks covering them.
        unc_list = np.where(uncovered)[0]
        for ti in unc_list:
            if time.time() > deadline:
                break
            # pick any block covering ti not already selected
            for bj in tset_to_blocks[ti]:
                if gain[bj] < 0 or (bi_selected := bj) in selected:
                    pass
            # simpler: just pick first block covering ti
            bj = tset_to_blocks[ti][0]
            if bj not in selected:
                selected.append(bj)
                for ti2 in block_tsets_arr[bj]:
                    uncovered[ti2] = False
        # recompute remaining
        remaining = int(np.sum(uncovered))

    # Convert selected block indices to actual blocks (sorted lists)
    blocks = []
    for bi in selected:
        blocks.append(list(block_list[bi]))

    # Local search: try to reduce block count by removing redundant blocks
    # A block is redundant if every tset it covers is covered by at least one other selected block.
    # We'll iterate: for each block, check if removing it leaves all its tsets covered.
    # Use coverage counts.
    if time.time() < deadline - 20.0:
        # build coverage count over selected blocks
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in selected:
            for ti in block_tsets_arr[bi]:
                cov[ti] += 1
        # try removing each block
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for bi in selected:
                # check if removing bi is safe
                ts = block_tsets_arr[bi]
                safe = True
                for ti in ts:
                    if cov[ti] <= 1:
                        safe = False
                        break
                if safe:
                    # remove bi
                    for ti in ts:
                        cov[ti] -= 1
                    changed = True
                else:
                    new_selected.append(bi)
            selected = new_selected

        blocks = []
        for bi in selected:
            blocks.append(list(block_list[bi]))

    return blocks
