"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: vectorized NumPy greedy + aggressive local search with multiple restarts.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 145.0  # seconds, safety margin under 150s
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute all t-subsets and for every k-subset its coverage as a NumPy uint8 array."""
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    k_subsets = list(itertools.combinations(range(v), k))
    n_ks = len(k_subsets)
    # coverage: NumPy uint8 array of shape (n_ks, n_t)
    coverage = np.zeros((n_ks, n_t), dtype=np.uint8)
    for i, ks in enumerate(k_subsets):
        for ts in itertools.combinations(ks, t):
            coverage[i, t_index[ts]] = 1
    return all_t, t_index, k_subsets, coverage


def _greedy_construct(coverage, rng, top_k=1):
    """Vectorized greedy set cover with optional top-k randomization."""
    n_ks, n_t = coverage.shape
    uncovered = np.ones(n_t, dtype=np.uint8)
    blocks = []
    while uncovered.any() and _elapsed() < _DEADLINE - 5:
        gains = coverage.dot(uncovered)
        if gains.max() == 0:
            break
        if top_k > 1:
            sorted_idx = np.argsort(gains)[::-1]
            best_i = sorted_idx[rng.randrange(min(top_k, n_ks))]
        else:
            best_i = int(gains.argmax())
        blocks.append(best_i)
        uncovered[coverage[best_i] > 0] = 0
    return blocks


def _is_fully_covered(blocks, coverage):
    """Check if blocks cover all t-subsets."""
    if not blocks:
        return False
    union = np.zeros(coverage.shape[1], dtype=np.uint8)
    for b in blocks:
        union |= coverage[b]
    return union.all()


def _local_search(blocks, coverage, rng, time_budget):
    """Local search: vectorized removal + swap passes."""
    blocks = list(blocks)
    n_ks, n_t = coverage.shape

    def compute_union(exclude_idx=None):
        if not blocks:
            return np.zeros(n_t, dtype=np.uint8)
        union = np.zeros(n_t, dtype=np.uint8)
        for idx, b in enumerate(blocks):
            if exclude_idx is not None and idx == exclude_idx:
                continue
            union |= coverage[b]
        return union

    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        # --- Removal pass ---
        while _elapsed() < time_budget:
            removed_any = False
            i = 0
            while i < len(blocks) and _elapsed() < time_budget:
                union = compute_union(exclude_idx=i)
                if union.all():
                    blocks.pop(i)
                    removed_any = True
                    improved = True
                    continue
                i += 1
            if not removed_any:
                break

        # --- Swap pass ---
        n_swaps = 40
        for _ in range(n_swaps):
            if _elapsed() >= time_budget:
                break
            i = rng.randrange(len(blocks))
            j = rng.randrange(n_ks)
            if j in blocks:
                continue
            old = blocks[i]
            blocks[i] = j
            removed = False
            for idx in range(len(blocks)):
                union = compute_union(exclude_idx=idx)
                if union.all():
                    blocks.pop(idx)
                    removed = True
                    improved = True
                    break
            if not removed:
                blocks[i] = old
            if removed:
                # Additional removal pass
                while _elapsed() < time_budget:
                    removed_any = False
                    ii = 0
                    while ii < len(blocks) and _elapsed() < time_budget:
                        union = compute_union(exclude_idx=ii)
                        if union.all():
                            blocks.pop(ii)
                            removed_any = True
                            improved = True
                            continue
                        ii += 1
                    if not removed_any:
                        break

        # --- Repair if needed ---
        if not _is_fully_covered(blocks, coverage):
            uncovered = np.ones(n_t, dtype=np.uint8)
            for b in blocks:
                uncovered[coverage[b] > 0] = 0
            gains = coverage.dot(uncovered)
            if gains.max() > 0:
                blocks.append(int(gains.argmax()))

    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    all_t, t_index, k_subsets, coverage = _precompute(v, k, t)
    n_t = len(all_t)

    best_blocks = None
    best_len = float('inf')

    # Multiple restarts with varying top_k
    restarts = [
        (1, 1),   # pure greedy
        (3, 2),   # top-3 randomization
        (5, 3),   # top-5 randomization
        (7, 5),   # top-7 randomization
        (10, 7),  # top-10 randomization
    ]

    # Reserve time for local search on best solution
    ls_time = 30.0
    time_per_restart = max(5.0, (_DEADLINE - ls_time - 5) / len(restarts))

    for top_k, seed_offset in restarts:
        if _elapsed() > _DEADLINE - ls_time - 5:
            break
        rng_r = random.Random(12345 + seed_offset)
        t_start = _elapsed()
        blocks = _greedy_construct(coverage, rng_r, top_k=top_k)
        # Local search with limited time per restart
        blocks = _local_search(blocks, coverage, rng_r, t_start + time_per_restart)
        if _is_fully_covered(blocks, coverage) and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)

    # Final local search on best solution with remaining time
    if best_blocks is not None:
        remaining = _DEADLINE - 3 - _elapsed()
        if remaining > 2:
            best_blocks = _local_search(best_blocks, coverage, rng, _elapsed() + remaining)

    # Safety: ensure valid
    if best_blocks is None or not _is_fully_covered(best_blocks, coverage):
        # Fallback: pure greedy
        blocks = _greedy_construct(coverage, rng, top_k=1)
        if not _is_fully_covered(blocks, coverage):
            uncovered = np.ones(n_t, dtype=np.uint8)
            for b in blocks:
                uncovered[coverage[b] > 0] = 0
            while uncovered.any() and _elapsed() < _DEADLINE - 1:
                gains = coverage.dot(uncovered)
                if gains.max() == 0:
                    break
                best_i = int(gains.argmax())
                blocks.append(best_i)
                uncovered[coverage[best_i] > 0] = 0
        best_blocks = blocks

    # Return as list of k-subsets (lists of ints)
    result = [list(k_subsets[b]) for b in best_blocks]
    # Deduplicate
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
