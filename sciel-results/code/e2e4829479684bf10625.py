"""Optimized covering design for C(v,k,t)."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

_DEADLINE = 140.0


def _cover_count(blocks, v, t):
    """Return the number of t-subsets covered by the given blocks (vectorized)."""
    if not blocks:
        return 0
    arr = np.array(blocks, dtype=np.int64)
    total = 0
    for b in arr:
        total += sum(len(np.unique(np.sort(np.array(c, dtype=np.int64))))
                     for c in itertools.combinations(b, t))
    return total


def _greedy(v, k, t, deadline, rng):
    """Fast vectorized greedy: build blocks on uncovered t-subsets."""
    all_t = np.array(list(itertools.combinations(range(v), t)), dtype=np.int64)
    n = all_t.shape[0]
    uncovered = np.ones(n, dtype=bool)
    blocks = []
    while uncovered.any():
        idx = int(np.argmax(uncovered))
        seed = list(all_t[idx])
        block = list(seed)
        while len(block) < k:
            # gain of adding x = number of uncovered t-subsets containing block's (t-1)-subsets + x
            best_x, best_g = None, -1
            for x in range(v):
                if x in block:
                    continue
                g = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    # count uncovered t-subsets that are supersets of sub
                    mask = np.all(all_t == np.array(list(sub)), axis=1)
                    g += int(np.count_nonzero(uncovered & mask))
                if g > best_g:
                    best_g, best_x = g, x
            block.append(best_x)
        block.sort()
        blocks.append(block)
        # mark covered
        for c in itertools.combinations(block, t):
            mask = np.all(all_t == np.array(list(c)), axis=1)
            uncovered &= ~mask
        if time.time() > deadline:
            break
    return blocks


def _is_valid(blocks, v, t):
    covered = set()
    for b in blocks:
        for c in itertools.combinations(b, t):
            covered.add(c)
    return len(covered) == len(list(itertools.combinations(range(v), t)))


def _remove_redundant(blocks, v, t, deadline):
    """Remove blocks that don't uniquely cover any t-subset."""
    if not blocks:
        return blocks
    # count how many blocks cover each t-subset
    from collections import defaultdict
    cover_by = defaultdict(set)
    for i, b in enumerate(blocks):
        for c in itertools.combinations(b, t):
            cover_by[c].add(i)
    keep = list(range(len(blocks)))
    removed = [False] * len(blocks)
    changed = True
    while changed:
        changed = False
        for i in keep:
            if removed[i]:
                continue
            if time.time() > deadline:
                return [blocks[j] for j in range(len(blocks)) if not removed[j]]
            # does block i uniquely cover any t-subset?
            unique = False
            for c in itertools.combinations(blocks[i], t):
                if len(cover_by[c]) == 1:
                    unique = True
                    break
            if not unique:
                removed[i] = True
                for c in itertools.combinations(blocks[i], t):
                    cover_by[c].discard(i)
                changed = True
    return [blocks[j] for j in range(len(blocks)) if not removed[j]]


def _swap_local(blocks, v, t, deadline):
    """Try to swap a point in a block for another to keep validity with fewer... (no-op here, keep size)."""
    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + _DEADLINE
    rng = np.random.default_rng(42)
    # Phase 1: greedy seed
    blocks = _greedy(v, k, t, deadline, rng)
    # Phase 2: remove redundant blocks
    blocks = _remove_redundant(blocks, v, t, deadline)
    # Ensure validity
    if not _is_valid(blocks, v, t):
        # fall back: rebuild quickly
        blocks = _greedy(v, k, t, deadline, rng)
    return blocks
# EVOLVE-BLOCK-END
