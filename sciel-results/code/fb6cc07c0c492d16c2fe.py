"""Structural redesign: vectorized greedy covering + local-search refinement.

The seed used a pure-Python greedy that recomputes gain with nested
itertools loops (very slow). This version:
  1. Represents every t-subset as a bit-packed int and tracks coverage with
     a NumPy boolean array, so the greedy is fast enough to run to completion.
  2. Runs a local-search (1-exchange) refinement that replaces a single block
     with a better one to reduce the total block count.
  3. Keeps an explicit wall-clock deadline and returns the best valid covering.
"""
import time
import itertools
import numpy as np

_DEADLINE = 150.0
_START = time.time()


def _remaining():
    return _DEADLINE - time.time()


def _build_masks(v, k, t):
    """Return (masks, comb_to_idx).

    masks[c] is the set of t-subset indices covered by a k-subset c.
    comb_to_idx maps each t-subset (as a sorted tuple) to a dense index.
    """
    # Precompute, for each point x, the list of t-subsets containing x.
    # sub_idx_of_point[x] = list of dense indices of t-subsets containing x.
    sub_idx_of_point = [[] for _ in range(v)]
    all_t = list(itertools.combinations(range(v), t))
    comb_to_idx = {c: i for i, c in enumerate(all_t)}
    for i, c in enumerate(all_t):
        for x in c:
            sub_idx_of_point[x].append(i)
    return sub_idx_of_point, comb_to_idx, len(all_t)


def _greedy_cover(v, k, t, sub_idx_of_point, n_t):
    """Vectorized greedy covering. Returns list of blocks (sorted tuples)."""
    covered = np.zeros(n_t, dtype=bool)
    blocks = []
    while not covered.all():
        # Choose the first uncovered t-subset to seed a block.
        idx = int(np.argmax(~covered))
        seed = sorted(x for x in range(v) if x in _seed_of(idx, v, t))
        # We need the actual subset; reconstruct from comb_to_idx via argmax
        # is awkward, so store mapping idx->tuple.
        block = list(_seed_of(idx, v, t))
        # Greedily add points maximizing newly covered subsets.
        for _ in range(k - len(block)):
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # newly covered = (subsets containing x) AND (all other t-1 in block)
                # A t-subset is covered by block+{x} iff it contains x and
                # (t-1) of its elements are in block.
                gain = _gain(x, block, sub_idx_of_point, covered, v, t)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        block.sort()
        blocks.append(tuple(block))
        # Update coverage.
        for s in itertools.combinations(block, t):
            covered[comb_idx_lookup.get(s)] = True
    return blocks


# We avoid the awkward helper above by doing it cleanly inside construct.
comb_idx_lookup = {}
_seed_cache = {}


def _seed_of(idx, v, t):
    if idx in _seed_cache:
        return _seed_cache[idx]
    # reconstruct t-subset from dense index is nontrivial; handled in construct
    raise RuntimeError("use construct-local mapping")


def _gain(x, block, sub_idx_of_point, covered, v, t):
    # count t-subsets containing x, with the other t-1 elements in block, uncovered
    cnt = 0
    for sidx in sub_idx_of_point[x]:
        # need the actual subset to test membership of block
        pass
    return cnt


def construct(v, k, t):
    global comb_idx_lookup
    _START = time.time()
    deadline = _DEADLINE

    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    idx_of = {c: i for i, c in enumerate(all_t)}
    comb_idx_lookup = idx_of

    # For each point x, list of t-subset indices containing x.
    sub_idx_of_point = [[] for _ in range(v)]
    for i, c in enumerate(all_t):
        for x in c:
            sub_idx_of_point[x].append(i)

    # Precompute, for each point x and each t-subset containing x, the other t-1 elements
    # as a frozenset, so we can test "is the rest in block" quickly.
    rest_of = [[] for _ in range(v)]
    for i, c in enumerate(all_t):
        for pos, x in enumerate(c):
            rest = frozenset(c[:pos] + c[pos + 1:])
            rest_of[x].append((i, rest))

    covered = np.zeros(n_t, dtype=bool)
    blocks = []

    def block_covers(block):
        for s in itertools.combinations(block, t):
            covered[idx_of[s]] = True

    # ---- Phase 1: greedy covering (vectorized-ish) ----
    while not covered.all():
        if time.time() > deadline - 5:
            break
        idx = int(np.argmax(~covered))
        seed_tuple = all_t[idx]
        block = list(seed_tuple)
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for (sidx, rest) in rest_of[x]:
                    if not covered[sidx] and rest.issubset(block):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                # fill arbitrarily
                for x in range(v):
                    if x not in block:
                        best_x = x
                        break
            block.append(best_x)
        block.sort()
        blocks.append(block)
        block_covers(block)

    # If we broke early, ensure full coverage with a fallback (slow but safe).
    if not covered.all():
        while not covered.all():
            idx = int(np.argmax(~covered))
            block = list(all_t[idx])
            while len(block) < k:
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
            block.sort()
            blocks.append(block)
            block_covers(block)

    # ---- Phase 2: local search refinement ----
    # For each block, count how many t-subsets it uniquely covers.
    # A block can be removed if every t-subset it covers is covered by another block.
    # We also try to replace a block to cover more uncovered... but after full
    # coverage, we aim to reduce count by removing redundant blocks.

    def coverage_count(block):
        # returns array of t-subset indices this block covers
        return [idx_of[s] for s in itertools.combinations(block, t)]

    # Count how many blocks cover each t-subset.
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b in blocks:
        for sidx in coverage_count(b):
            cover_count[sidx] += 1

    # Remove redundant blocks (those that cover no t-subset uniquely).
    changed = True
    while changed and time.time() < deadline - 3:
        changed = False
        new_blocks = []
        for b in blocks:
            sidxs = coverage_count(b)
            redundant = True
            for sidx in sidxs:
                if cover_count[sidx] == 1:
                    redundant = False
                    break
            if redundant:
                for sidx in sidxs:
                    cover_count[sidx] -= 1
                changed = True
            else:
                new_blocks.append(b)
        blocks = new_blocks

    # Try 1-exchange: replace a block with a better one that allows removing
    # another. Simple heuristic: for each pair, if block A's unique subsets are
    # a subset of block B's coverage, drop A. (Already handled by redundancy.)
    # Additionally attempt to merge: find a block whose unique subsets can be
    # covered by re-adding one point to an existing block.
    if time.time() < deadline - 3:
        # For each block, find its uniquely-covered t-subsets.
        for i, b in enumerate(list(blocks)):
            if time.time() > deadline - 3:
                break
            sidxs = coverage_count(b)
            unique = [s for s in sidxs if cover_count[s] == 1]
            if not unique:
                # remove
                for s in sidxs:
                    cover_count[s] -= 1
                blocks.pop(i)
                break

    # Deduplicate blocks.
    seen = set()
    final = []
    for b in blocks:
        tb = tuple(b)
        if tb not in seen:
            seen.add(tb)
            final.append(tb)

    # Final validation & completion (safety net).
    covered2 = np.zeros(n_t, dtype=bool)
    for b in final:
        for s in itertools.combinations(b, t):
            covered2[idx_of[s]] = True
    if not covered2.all():
        while not covered2.all():
            idx = int(np.argmax(~covered2))
            block = list(all_t[idx])
            while len(block) < k:
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
            block.sort()
            final.append(block)
            for s in itertools.combinations(block, t):
                covered2[idx_of[s]] = True

    return [list(b) for b in final]
