"""Packing-First greedy covering for C(19,7,4).
Vectorized greedy: fast gain computation via numpy, deadline guard."""
import time
import itertools
import numpy as np

# Precompute all t-subsets as integer bitmasks (t=4 -> 12870 subsets)
def _build_masks(v, t):
    masks = []
    for combo in itertools.combinations(range(v), t):
        m = 0
        for x in combo:
            m |= (1 << x)
        masks.append(m)
    return masks

# Precompute, for every k-subset, the list of t-subset indices it covers
def _build_cover_index(v, k, t, masks):
    mask_to_idx = {m: i for i, m in enumerate(masks)}
    cover = []
    for block in itertools.combinations(range(v), k):
        bm = 0
        for x in block:
            bm |= (1 << x)
        # find which t-subsets are contained in block
        idxs = []
        for sub in itertools.combinations(block, t):
            sm = 0
            for x in sub:
                sm |= (1 << x)
            idxs.append(mask_to_idx[sm])
        cover.append(idxs)
    return cover


def construct(v, k, t):
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    masks = _build_masks(v, t)
    n_sub = len(masks)

    # uncovered: boolean array, True = not yet covered
    uncovered = np.ones(n_sub, dtype=bool)
    # counts: how many times each t-subset is covered (for tie-breaking / efficiency)
    # We'll track number of blocks and try to minimize.

    # For gain computation: for a candidate point x added to current partial block B,
    # gain = number of newly-covered t-subsets that contain x and (t-1) points from B.
    # Precompute: for each point x, for each (t-1)-subset S, the list of t-subset indices
    # that equal S union {x}.

    # Build: point_submap[x] = dict mapping frozenset(S) -> list of t-indices
    # Actually simpler: for each x and each (t-1)-tuple S of other points, the t-set
    # S+{x} has a known index. We precompute a 2D structure.

    # Let's precompute: for each point x, an array over all (t-1)-subsets of points != x.
    # That's complex. Instead, precompute a mapping:
    #   for each x, for each (t-1)-subset S (as bitmask), the t-subset index of S U {x}
    # Represent S as bitmask. There are 2^(v-1) possible but we only need (t-1)-subsets.

    # Simpler approach: precompute cover_count for a full block quickly.
    # For greedy, at each step we have partial block B (size b < k), and candidate x not in B.
    # gain(x) = |{ T in uncovered : x in T, and T \ {x} subset of B }|
    #          = |{ S : S is (t-1)-subset of B, and S U {x} in uncovered }|

    # Precompute for each x: a dictionary from (t-1)-subset bitmask to t-subset index.
    # (t-1)-subsets of points other than x.
    point_x_to_submap = {}
    for x in range(v):
        submap = {}
        others = [y for y in range(v) if y != x]
        for S in itertools.combinations(others, t - 1):
            smask = 0
            for y in S:
                smask |= (1 << y)
            # t-set = S U {x}
            tmask = smask | (1 << x)
            # find index
            submap[smask] = tmask  # store tmask; we'll map to index via mask_to_idx
        point_x_to_submap[x] = submap

    mask_to_idx = {m: i for i, m in enumerate(masks)}

    # Convert submap to use index directly
    point_x_to_subidx = {}
    for x in range(v):
        smap = point_x_to_submap[x]
        point_x_to_subidx[x] = {smask: mask_to_idx[tmask] for smask, tmask in smap.items()}

    blocks = []

    while uncovered.any():
        if time.time() > deadline:
            break

        # Find first uncovered t-subset to anchor the new block
        first_idx = int(np.argmax(uncovered))
        anchor_mask = masks[first_idx]
        anchor_set = [x for x in range(v) if (anchor_mask >> x) & 1]

        # Build block greedily: start with anchor points? No, start with one point, add best.
        # Actually for packing-first / efficiency, start with the anchor's first point and
        # grow to maximize new coverage.
        block = [anchor_set[0]]
        block_set = {anchor_set[0]}

        while len(block) < k:
            if time.time() > deadline:
                break

            # Compute gain for each candidate x not in block
            best_x = None
            best_gain = -1
            # (t-1)-subsets of current block
            block_list = block
            for x in range(v):
                if x in block_set:
                    continue
                subidx_map = point_x_to_subidx[x]
                gain = 0
                for S in itertools.combinations(block_list, t - 1):
                    smask = 0
                    for y in S:
                        smask |= (1 << y)
                    if smask in subidx_map:
                        idx = subidx_map[smask]
                        if uncovered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x is None or best_gain <= 0:
                # no candidate adds new coverage; pick an arbitrary unused point to fill
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break
                if best_x is None:
                    break

            block.append(best_x)
            block_set.add(best_x)

        if len(block) < k:
            # fill with arbitrary unused points
            for x in range(v):
                if x not in block_set and len(block) < k:
                    block.append(x)
                    block_set.add(x)

        block.sort()

        # Mark all t-subsets within this block as covered
        for sub in itertools.combinations(block, t):
            sm = 0
            for x in sub:
                sm |= (1 << x)
            uncovered[mask_to_idx[sm]] = False

        blocks.append(block)

    # If we ran out of time and still have uncovered, we cannot return an invalid result.
    # The contract requires validity. So we must ensure full coverage.
    # Fallback: continue with a slower but guaranteed-complete greedy if needed.
    if uncovered.any():
        # Slow fallback to guarantee completeness (only used if deadline hit early)
        uncovered_set = set(int(i) for i in np.where(uncovered)[0])
        while uncovered_set:
            idx = min(uncovered_set)
            anchor_mask = masks[idx]
            anchor_set = [x for x in range(v) if (anchor_mask >> x) & 1]
            block = [anchor_set[0]]
            block_set = {anchor_set[0]}
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    subidx_map = point_x_to_subidx[x]
                    gain = 0
                    for S in itertools.combinations(block, t - 1):
                        smask = 0
                        for y in S:
                            smask |= (1 << y)
                        if smask in subidx_map:
                            i2 = subidx_map[smask]
                            if i2 in uncovered_set:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None or best_gain <= 0:
                    for x in range(v):
                        if x not in block_set:
                            best_x = x
                            break
                    if best_x is None:
                        break
                block.append(best_x)
                block_set.add(best_x)
            for x in range(v):
                if x not in block_set and len(block) < k:
                    block.append(x)
                    block_set.add(x)
            block.sort()
            for sub in itertools.combinations(block, t):
                sm = 0
                for x in sub:
                    sm |= (1 << x)
                uncovered_set.discard(mask_to_idx[sm])
            blocks.append(block)

    return blocks
