# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin below 120s

    points = list(range(v))
    # Precompute all t-subsets as bitmasks
    tsubs = [set(c) for c in itertools.combinations(points, t)]
    # Map t-subset (frozenset) -> index
    tsub_to_idx = {frozenset(c): i for i, c in enumerate(tsubs)}
    n_tsubs = len(tsubs)

    # uncovered: boolean array
    uncovered = np.ones(n_tsubs, dtype=bool)

    # Precompute bitmasks for each point
    point_bit = {p: 1 << p for p in points}

    # Precompute all (t-1)-subsets of each point-set on the fly via helper
    # For gain computation we need (t-1)-subsets of the current block.

    blocks = []
    v_list = points
    v_arr = np.array(v_list)
    # Precompute bit masks array
    bit_arr = np.array([1 << p for p in v_list], dtype=np.int64)

    # Degree: number of uncovered t-subsets containing each point
    # Compute initial degrees
    # For each t-subset, it contains t points; we can accumulate.
    # Build list of point indices for each t-subset
    tsub_points = [list(c) for c in itertools.combinations(v_list, t)]
    # degree[p] = count of uncovered tsubs containing p
    degree = np.zeros(v, dtype=np.int64)
    # Build point -> list of tsub indices containing it (for fast update)
    from collections import defaultdict
    point_tsubs = defaultdict(list)
    for i, c in enumerate(tsub_points):
        for p in c:
            point_tsubs[p].append(i)
            degree[p] += 1

    def compute_gain_vec(block_set, cand_mask):
        """Compute number of uncovered t-subsets covered by adding cand to block_set.
        block_set: list of points currently in block (size < k)
        cand_mask: bitmask of candidate point
        Returns gain (int).
        """
        b = list(block_set)
        if t - 1 == 0:
            # t=1: adding cand covers the t-subset {cand} if uncovered
            idx = tsub_to_idx.get(frozenset([cand]))
            if idx is not None and uncovered[idx]:
                return 1
            return 0
        # (t-1)-subsets of b
        gain = 0
        for sub in itertools.combinations(b, t - 1):
            sub_mask = 0
            for p in sub:
                sub_mask |= point_bit[p]
            full_mask = sub_mask | cand_mask
            # find index
            # We need to map a bitmask to a t-subset index.
            # Precompute a dict from bitmask to index.
            idx = bitmask_to_idx.get(full_mask)
            if idx is not None and uncovered[idx]:
                gain += 1
        return gain

    # Precompute bitmask -> index for all t-subsets
    bitmask_to_idx = {}
    for i, c in enumerate(tsub_points):
        m = 0
        for p in c:
            m |= point_bit[p]
        bitmask_to_idx[m] = i

    # For vectorized gain: precompute list of (t-1)-subset masks for a given block is dynamic.
    # We'll use a vectorized approach: for candidate x, gain = number of uncovered
    # t-subsets of form (t-1-subset of block) union {x}.

    # To speed up, precompute for each block-size the (t-1)-subsets. But block changes.
    # Alternative: maintain the set of (t-1)-subset masks of the current block as we build it.

    def greedy_block():
        """Build one block greedily, return list of points."""
        # Choose seed: pick uncovered t-subset with max sum of point degrees
        # We need an uncovered t-subset. Find one with max degree-sum.
        best_tsub = None
        best_score = -1
        # Sample or scan? n_tsubs is small (3876), scan all uncovered.
        for i in range(n_tsubs):
            if not uncovered[i]:
                continue
            c = tsub_points[i]
            s = 0
            for p in c:
                s += degree[p]
            if s > best_score:
                best_score = s
                best_tsub = c
        if best_tsub is None:
            return None
        block = list(best_tsub)
        block_set = set(block)

        # We need (t-1)-subsets of the current block for gain computation.
        # As we add points, recompute (t-1)-subsets.
        # For efficiency, precompute masks of (t-1)-subsets of current block.
        def submasks_of(block_list, r):
            masks = []
            for sub in itertools.combinations(block_list, r):
                m = 0
                for p in sub:
                    m |= point_bit[p]
                masks.append(m)
            return masks

        while len(block) < k:
            if time.time() > deadline:
                break
            # Compute gain for each candidate not in block
            best_cand = None
            best_gain = -1
            cur_submasks = submasks_of(block, t - 1)
            # For each candidate, gain = count of uncovered tsubs = submask | cand_bit
            # Vectorize: build array of full masks, look up indices, check uncovered.
            cand_mask_arr = bit_arr.copy()
            # We'll iterate candidates but vectorize the lookup per candidate is still O(v * C(|B|,t-1))
            # Instead: for each candidate, full_masks = [sm | cand_bit for sm in cur_submasks]
            # Use numpy: for a fixed candidate, full = cur_submask_arr | cand_bit
            cur_submask_arr = np.array(cur_submasks, dtype=np.int64)
            # Build a dict-based lookup vectorized via np.vectorize? Slower.
            # Precompute an array mapping bitmask -> index using a dict; for vectorization
            # we can't easily index a dict. Use a large array if v small.
            # v=19, bitmask up to 2^19 = 524288. We can make an array of that size.
            # idx_arr[bitmask] = index or -1
            # Build once.
            pass

            # Simpler: iterate candidates, compute gain via loop over cur_submasks (small).
            for x in v_list:
                if x in block_set:
                    continue
                g = 0
                xb = point_bit[x]
                for sm in cur_submasks:
                    fm = sm | xb
                    idx = bitmask_to_idx.get(fm)
                    if idx is not None and uncovered[idx]:
                        g += 1
                if g > best_gain:
                    best_gain = g
                    best_cand = x
            if best_cand is None or best_gain <= 0:
                # No gain; pick arbitrary unused point to fill block
                for x in v_list:
                    if x not in block_set:
                        best_cand = x
                        break
                if best_cand is None:
                    break
            block.append(best_cand)
            block_set.add(best_cand)

        # Now cover all t-subsets of this block
        for c in itertools.combinations(block, t):
            idx = tsub_to_idx[frozenset(c)]
            if uncovered[idx]:
                uncovered[idx] = False
                # Update degrees: this t-subset's points each lose 1
                for p in c:
                    degree[p] -= 1
        return sorted(block)

    while True:
        if time.time() > deadline:
            break
        if not uncovered.any():
            break
        b = greedy_block()
        if b is None:
            break
        blocks.append(b)

    # Verify completeness; if incomplete (shouldn't happen if we finished), still return.
    # The greedy loop above updates uncovered, so if uncovered.any() is True we didn't finish.
    # Return whatever we have.
    return blocks
# EVOLVE-BLOCK-END
