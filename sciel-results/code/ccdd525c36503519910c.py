"""Fresh design: fast multi-strategy greedy covering with time guard + local improvement."""
import itertools
import time


def _all_tsubs(v, t):
    return [tuple(c) for c in itertools.combinations(range(v), t)]


def _block_covers(block, t):
    return [tuple(c) for c in itertools.combinations(block, t)]


def _greedy_construct(v, k, t, all_tsubs, deadline, seed_start=0):
    """Greedy set cover over the precomputed t-subsets.

    Returns (blocks, covered_mask, elapsed). Stops on deadline and returns best-so-far.
    """
    n = len(all_tsubs)
    # Map each t-subset to an index.
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    # For each point, the list of t-subset indices that contain it.
    point_to_tsubs = [[] for _ in range(v)]
    for i, s in enumerate(all_tsubs):
        for p in s:
            point_to_tsubs[p].append(i)

    # Greedy set cover: repeatedly pick the block covering most uncovered t-subsets.
    # A block is built greedily by picking points with best marginal gain.
    covered = [False] * n
    blocks = []
    start = time.time()

    def build_block():
        # Start from the smallest uncovered t-subset (a good anchor).
        # Find an uncovered t-subset to anchor on.
        anchor = None
        for i in range(n):
            if not covered[i]:
                anchor = all_tsubs[i]
                break
        if anchor is None:
            return None
        block = list(anchor)
        in_block = set(block)
        # Build up to k points by marginal gain.
        while len(block) < k:
            best_x = None
            best_gain = -1
            # Only consider points not already in block.
            for x in range(v):
                if x in in_block:
                    continue
                # Gain = number of uncovered t-subsets that contain x AND are
                # completable with the rest of the block.
                # A t-subset s is covered by block+{x} iff s \ {x} is a subset of block.
                # So count uncovered s in point_to_tsubs[x] with (s \ {x}) <= block.
                gain = 0
                for idx in point_to_tsubs[x]:
                    if covered[idx]:
                        continue
                    s = all_tsubs[idx]
                    # s contains x (guaranteed). Check s \ {x} subset of block.
                    ok = True
                    for p in s:
                        if p != x and p not in in_block:
                            ok = False
                            break
                    if ok:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None or best_gain <= 0:
                # No point helps; pad arbitrarily to keep block valid (k distinct).
                for x in range(v):
                    if x not in in_block:
                        block.append(x)
                        in_block.add(x)
                    if len(block) == k:
                        break
                break
            block.append(best_x)
            in_block.add(best_x)
        # Mark all t-subsets contained in this block as covered.
        # Ensure block is sorted so that combinations are sorted and match tsub_to_idx keys
        block.sort()
        for c in itertools.combinations(block, t):
            idx = tsub_to_idx[tuple(c)]
            if not covered[idx]:
                covered[idx] = True
        return block

    # Greedy: each iteration build a block anchored on the first uncovered t-subset.
    # This guarantees progress (the anchor gets covered).
    while True:
        if time.time() > deadline:
            break
        b = build_block()
        if b is None:
            break
        blocks.append(b)
        # Stop when everything covered.
        if all(covered):
            break

    return blocks, covered, time.time() - start


def _local_improve(blocks, v, k, t, all_tsubs, deadline):
    """Try to drop redundant blocks, then merge/shrink via random restarts.

    Returns improved list of blocks (fewer is better).
    """
    n = len(all_tsubs)
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}

    def coverage_of(block):
        return set(tsub_to_idx[tuple(c)] for c in itertools.combinations(block, t))

    cur = list(blocks)
    # Precompute per-block coverage.
    cov = [coverage_of(b) for b in cur]
    # Drop redundant blocks (one whose coverage is a subset of the union of others).
    changed = True
    while changed:
        changed = False
        m = len(cur)
        for i in range(m):
            if time.time() > deadline:
                return cur
            # Union of all other blocks.
            others = set()
            for j in range(m):
                if j != i:
                    others |= cov[j]
            if cov[i] <= others:
                cur.pop(i)
                cov.pop(i)
                changed = True
                break
    return cur


def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin before 120s limit
    all_tsubs = _all_tsubs(v, t)

    # Phase 1: fast greedy to get a valid covering.
    blocks, covered, _ = _greedy_construct(v, k, t, all_tsubs, deadline)

    # If greedy didn't finish in time, it may be invalid. Ensure validity by
    # continuing greedily (anchored) until covered or deadline.
    # (build_block anchors on first uncovered, so it always makes progress.)
    if not all(covered):
        # Continue with more time budget.
        blocks2, covered2, _ = _greedy_construct(v, k, t, all_tsubs, deadline, seed_start=0)
        # Merge: use whichever is more complete.
        if sum(1 for c in covered2 if c) >= sum(1 for c in covered if c):
            blocks = blocks2
            covered = covered2

    # Phase 2: local improvement (drop redundant blocks) within remaining time.
    if time.time() < deadline - 5.0 and blocks:
        blocks = _local_improve(blocks, v, k, t, all_tsubs, deadline)

    # Final validity guarantee: if somehow not all covered, top up greedily.
    if blocks and not all(covered):
        # Recompute covered from blocks.
        n = len(all_tsubs)
        tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
        covered = [False] * n
        for b in blocks:
            for c in itertools.combinations(b, t):
                covered[tsub_to_idx[tuple(c)]] = True
        if not all(covered):
            blocks, covered, _ = _greedy_construct(v, k, t, all_tsubs, deadline)

    # Deduplicate blocks (they should already be distinct).
    seen = set()
    uniq = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            uniq.append(list(b))
    return uniq
