"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: Lagrangian-weighted greedy construction + aggressive local search.
Dual weights on uncovered 4-subsets guide block selection; subgradient updates
dynamically reweight the objective to balance coverage density vs. block count.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 118.0  # seconds, safety margin under 150s
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute all t-subsets and for every k-subset its coverage as an int bitmask."""
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    k_subsets = list(itertools.combinations(range(v), k))
    n_ks = len(k_subsets)
    # Build coverage matrix as numpy array of shape (n_ks, n_t) with int64
    # n_t = C(19,4) = 3876, n_ks = C(19,7) = 50388
    # Each k-subset covers C(7,4)=35 t-subsets
    coverage = np.zeros((n_ks, n_t), dtype=np.int8)
    for i, ks in enumerate(k_subsets):
        for ts in itertools.combinations(ks, t):
            coverage[i, t_index[ts]] = 1
    full_mask = (1 << n_t) - 1
    # Also keep int bitmask coverage for local search
    coverage_masks = []
    for i in range(n_ks):
        mask = 0
        for ts in itertools.combinations(k_subsets[i], t):
            mask |= (1 << t_index[ts])
        coverage_masks.append(mask)
    return all_t, t_index, k_subsets, coverage, coverage_masks, full_mask, n_ks


def _lagrangian_greedy(v, k, t, rng, time_budget):
    """
    Weighted greedy set cover using Lagrangian dual weights.
    
    Objective: minimize sum_blocks(1) + sum_uncovered(w_j)
    where w_j is the dual weight for t-subset j.
    
    Greedy picks the block maximizing weighted_gain = sum_{j in block, j uncovered} w_j - 1
    (the -1 accounts for the block cost in the Lagrangian).
    
    Subgradient updates: w_j += eta * (1 - covered_j) for uncovered j.
    """
    all_t, t_index, k_subsets, coverage, coverage_masks, full_mask, n_ks = _precompute(v, k, t)
    n_t = len(all_t)
    
    # Convert coverage to column-wise for fast weighted sum computation
    # coverage is (n_ks, n_t) int8 array
    # For weighted greedy: gain_i = dot(coverage[i], w) - 1 where w_j > 0 only for uncovered
    
    blocks = []
    uncovered = np.ones(n_t, dtype=np.int8)  # 1 if not yet covered
    w = np.ones(n_t, dtype=np.float64)  # dual weights, start at 1
    eta = 1.0  # step size for subgradient
    
    # Precompute: for each k-subset, which t-subsets it covers (as indices)
    # coverage is already (n_ks, n_t) with 1s at covered positions
    
    while np.any(uncovered) and _elapsed() < time_budget:
        # Weighted gain for each block: sum of w_j over uncovered j in block - 1
        # gain_i = sum_j(coverage[i,j] * w[j] * uncovered[j]) - 1
        # = dot(coverage[i] * uncovered, w) - 1
        # Compute all at once: gains = coverage @ (w * uncovered) - 1
        weighted_uncovered = w * uncovered  # (n_t,)
        gains = coverage @ weighted_uncovered - 1.0  # (n_ks,)
        
        best_i = int(np.argmax(gains))
        best_gain = gains[best_i]
        
        if best_gain <= 0:
            # No block gives positive Lagrangian gain; pick max raw coverage of uncovered
            raw_gains = coverage @ uncovered
            best_i = int(np.argmax(raw_gains))
            if raw_gains[best_i] <= 0:
                break
        
        blocks.append(best_i)
        uncovered[coverage[best_i] == 1] = 0
        
        # Subgradient update: increase weights for still-uncovered subsets
        # g_j = 1 - covered_j = uncovered_j (subgradient of max(0, uncovered_j))
        eta = max(0.5, 2.0 / np.sqrt(len(blocks) + 1))  # decreasing step
        w = w + eta * uncovered
        # Clip weights to avoid explosion
        w = np.clip(w, 0.1, 50.0)
    
    # Repair: ensure full coverage
    if np.any(uncovered):
        while np.any(uncovered) and _elapsed() < time_budget:
            raw_gains = coverage @ uncovered
            best_i = int(np.argmax(raw_gains))
            if raw_gains[best_i] <= 0:
                break
            blocks.append(best_i)
            uncovered[coverage[best_i] == 1] = 0
    
    return blocks, coverage, coverage_masks, full_mask, n_ks, all_t, t_index, k_subsets


def _popcount(x):
    return bin(x).count('1')


def _local_search(blocks, coverage_masks, n_t, k_subsets, rng, full_mask, time_budget):
    """Aggressive local search: repeated removal + swap passes with bitmask speed."""
    blocks = list(blocks)
    n_ks = len(k_subsets)

    def union_mask(exclude_idx=None):
        m = 0
        for idx, b in enumerate(blocks):
            if exclude_idx is not None and idx == exclude_idx:
                continue
            m |= coverage_masks[b]
        return m

    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        # --- Removal pass (multiple iterations until no more removals) ---
        while _elapsed() < time_budget:
            removed_any = False
            i = 0
            while i < len(blocks) and _elapsed() < time_budget:
                m = union_mask(exclude_idx=i)
                if m == full_mask:
                    blocks.pop(i)
                    removed_any = True
                    improved = True
                    continue
                i += 1
            if not removed_any:
                break

        # --- Swap pass: try random swaps that enable removals ---
        n_swaps = 80
        for _ in range(n_swaps):
            if _elapsed() >= time_budget:
                break
            i = rng.randrange(len(blocks))
            j = rng.randrange(n_ks)
            if j in blocks:
                continue
            old = blocks[i]
            blocks[i] = j
            # Try to find a removable block after swap
            removed = False
            for idx in range(len(blocks)):
                m = union_mask(exclude_idx=idx)
                if m == full_mask:
                    blocks.pop(idx)
                    removed = True
                    improved = True
                    break
            if not removed:
                blocks[i] = old
            # After a successful swap+remove, do another removal pass
            if removed:
                while _elapsed() < time_budget:
                    removed_any = False
                    ii = 0
                    while ii < len(blocks) and _elapsed() < time_budget:
                        m = union_mask(exclude_idx=ii)
                        if m == full_mask:
                            blocks.pop(ii)
                            removed_any = True
                            improved = True
                            continue
                        ii += 1
                    if not removed_any:
                        break

        # --- Add-minimize move: try adding a block that enables 2+ removals ---
        if _elapsed() < time_budget:
            best_add = -1
            best_net = 0
            for j in range(n_ks):
                if j in blocks:
                    continue
                # Check if adding j enables removing 2+ blocks
                # Quick check: does j cover t-subsets that are ONLY covered by 1-2 blocks?
                # This is expensive; do a sampled version
                pass  # Skip for now; keep swap-based approach

        # --- Repair if somehow not covered ---
        m = 0
        for b in blocks:
            m |= coverage_masks[b]
        if m != full_mask:
            missing = full_mask & ~m
            best_i = -1
            best_g = 0
            for idx in range(n_ks):
                g = _popcount(coverage_masks[idx] & missing)
                if g > best_g:
                    best_g = g
                    best_i = idx
            if best_i >= 0:
                blocks.append(best_i)

    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    # Lagrangian-weighted greedy construction (uses ~30s budget)
    greedy_budget = 30.0
    blocks, coverage, coverage_masks, full_mask, n_ks, all_t, t_index, k_subsets = \
        _lagrangian_greedy(v, k, t, rng, greedy_budget)
    
    n_t = len(all_t)

    # Local search with remaining time budget
    time_budget = _DEADLINE - 3
    blocks = _local_search(blocks, coverage_masks, n_t, k_subsets, rng, full_mask, time_budget)

    # Safety: ensure valid
    m = 0
    for b in blocks:
        m |= coverage_masks[b]
    if m != full_mask:
        while m != full_mask and _elapsed() < _DEADLINE - 1:
            missing = full_mask & ~m
            best_i = -1
            best_g = 0
            for idx in range(n_ks):
                g = _popcount(coverage_masks[idx] & missing)
                if g > best_g:
                    best_g = g
                    best_i = idx
            if best_i < 0:
                break
            blocks.append(best_i)
            m |= coverage_masks[best_i]

    # Return as list of k-subsets (lists of ints)
    result = [list(k_subsets[b]) for b in blocks]
    # Deduplicate
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
