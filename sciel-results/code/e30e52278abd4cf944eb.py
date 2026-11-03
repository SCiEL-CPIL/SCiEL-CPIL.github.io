"""
Covering design C(19,7,4) solver.

Strategy:
  1. Fast greedy construction using precomputed incidence data and a max-heap
     to always pick the block that covers the most uncovered t-subsets.
  2. Local search (block replacement / swap) to reduce block count.
  3. Deadline guard to return best result before the 150 s wall-clock limit.

The greedy step is vectorised with NumPy for speed.  Local search tries to
replace a block with a different block that covers at least as many currently
uncovered t-subsets while removing a block that covers few unique subsets.
"""
import itertools
import time
import numpy as np
import heapq

# Precomputed data for v=19, k=7, t=4
_V = 19
_K = 7
_T = 4
_ALL_TSUBS = list(itertools.combinations(range(_V), _T))
_NTSUBS = len(_ALL_TSUBS)  # C(19,4) = 3876
_TSUB_INDEX = {ts: i for i, ts in enumerate(_ALL_TSUBS)}

# Precompute for every 7-subset: the list of t-subset indices it covers
_ALL_KSUBS = list(itertools.combinations(range(_V), _K))
_NKSUBS = len(_ALL_KSUBS)  # C(19,7) = 50388
_KSUB_TO_TSUBS = []
for ks in _ALL_KSUBS:
    idxs = []
    for ts in itertools.combinations(ks, _T):
        idxs.append(_TSUB_INDEX[ts])
    _KSUB_TO_TSUBS.append(idxs)

# Map from each t-subset to the list of k-subset indices that contain it
_TSUB_TO_KSUBS = [[] for _ in range(_NTSUBS)]
for ki, ks in enumerate(_ALL_KSUBS):
    for ts_idx in _KSUB_TO_TSUBS[ki]:
        _TSUB_TO_KSUBS[ts_idx].append(ki)


def _greedy_construct():
    """Fast greedy: repeatedly add the k-subset that covers the most uncovered t-subsets."""
    uncovered = np.ones(_NTSUBS, dtype=np.bool_)
    blocks = []
    # For each k-subset, precompute its t-subset indices as a numpy array
    ksub_ts = [np.array(lst, dtype=np.int32) for lst in _KSUB_TO_TSUBS]

    while uncovered.any():
        # Find the k-subset covering the most uncovered t-subsets
        best_count = -1
        best_ki = -1
        for ki in range(_NKSUBS):
            c = np.count_nonzero(uncovered[ksub_ts[ki]])
            if c > best_count:
                best_count = c
                best_ki = ki
            if c == 0:
                continue
        if best_ki < 0:
            break
        block = list(_ALL_KSUBS[best_ki])
        blocks.append(block)
        uncovered[ksub_ts[best_ki]] = False
    return blocks


def _local_search(blocks, time_limit):
    """
    Local search: try to remove a block and re-cover its unique t-subsets with
    other existing blocks or a new block. Also try swapping one block for another.
    """
    blocks = list(blocks)
    deadline = time.time() + time_limit
    n = len(blocks)
    if n == 0:
        return blocks

    # Build coverage count: for each t-subset, how many blocks cover it
    cover_count = np.zeros(_NTSUBS, dtype=np.int32)
    block_ts = []
    for b in blocks:
        ts_idxs = []
        for ts in itertools.combinations(b, _T):
            ts_idxs.append(_TSUB_INDEX[ts])
        ts_arr = np.array(ts_idxs, dtype=np.int32)
        block_ts.append(ts_arr)
        cover_count[ts_arr] += 1

    # Unique t-subsets per block (those covered only by this block)
    # We'll recompute as needed.

    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(blocks)
        for i in range(n):
            if time.time() >= deadline:
                break
            # Temporarily remove block i
            cover_count[block_ts[i]] -= 1
            # Find t-subsets now uncovered (count == 0)
            uncovered_mask = cover_count == 0
            n_uncovered = np.count_nonzero(uncovered_mask)
            if n_uncovered == 0:
                # Block i is redundant!
                cover_count[block_ts[i]] += 1  # restore (not needed, we remove it)
                blocks.pop(i)
                block_ts.pop(i)
                improved = True
                break
            # Try to find a replacement block that covers all currently uncovered t-subsets
            # and is not already in the list
            # For speed, only check k-subsets that contain at least one of the uncovered t-subsets
            best_ki = -1
            best_extra = -1
            uncovered_list = np.where(uncovered_mask)[0]
            # Build set of uncovered for fast lookup
            unc_set = set(uncovered_list.tolist())
            for ki in range(_NKSUBS):
                ts_arr = _KSUB_TO_TSUBS[ki]
                # Count how many of the uncovered t-subsets this k-subset covers
                c = 0
                for idx in ts_arr:
                    if idx in unc_set:
                        c += 1
                if c == len(unc_set):
                    # This k-subset covers all uncovered t-subsets
                    # Check it's not already a block
                    already = False
                    for j, b in enumerate(blocks):
                        if j == i:
                            continue
                        if set(b) == set(_ALL_KSUBS[ki]):
                            already = True
                            break
                    if not already:
                        # Accept this replacement
                        cover_count[block_ts[i]] += 1  # restore old block count
                        blocks[i] = list(_ALL_KSUBS[ki])
                        new_ts = np.array(ts_arr, dtype=np.int32)
                        block_ts[i] = new_ts
                        cover_count[new_ts] += 1
                        improved = True
                        break
            if improved:
                break
            # Restore
            cover_count[block_ts[i]] += 1

    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + 135  # 15 s safety margin before 150 s kill

    # For the specific task v=19, k=7, t=4, use precomputed data
    if v == _V and k == _K and t == _T:
        # Phase 1: Greedy
        blocks = _greedy_construct()
        if time.time() > deadline:
            return blocks

        # Phase 2: Local search
        remaining = deadline - time.time()
        if remaining > 5:
            blocks = _local_search(blocks, remaining - 5)

        return blocks
    else:
        # Fallback: simple greedy for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                best_x = max((x for x in range(v) if x not in block), key=gain)
                block.append(best_x)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
            if time.time() > deadline:
                break
        return blocks
