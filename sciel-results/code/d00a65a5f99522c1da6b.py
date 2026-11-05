import itertools
import math
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s kill

    # --- Precompute all t-subsets and map to indices ---
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubsets)}
    n_tsubs = len(all_tsubsets)

    # --- Precompute all k-subsets (blocks) and their triple indices ---
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    t_per_block = math.comb(k, t)
    block_indices = np.empty((n_blocks, t_per_block), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        block_indices[i] = [tsub_to_idx[sub] for sub in itertools.combinations(block, t)]

    # Inverse frequency weighting (rarity of each triple across all blocks)
    freq = np.bincount(block_indices.flatten(), minlength=n_tsubs)
    inv_freq = 1.0 / (freq + 1e-9)

    # Map a block (tuple) to its global index
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}

    def _remove_redundant(blocks, deadline):
        """Remove blocks that cover no unique triple. Returns list of tuples."""
        if not blocks:
            return blocks
        # coverage count per triple
        cover = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            idx = block_to_idx[b]
            cover[block_indices[idx]] += 1
        kept = []
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for b in blocks:
                idx = block_to_idx[b]
                redundant = True
                for j in block_indices[idx]:
                    if cover[j] <= 1:
                        redundant = False
                        break
                if redundant:
                    cover[block_indices[idx]] -= 1
                    changed = True
        # rebuild kept from blocks minus redundant (recompute to be safe)
        # simpler: recompute coverage from blocks and drop redundant
        cover2 = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            cover2[block_to_idx[b]] += 1
        kept = []
        for b in blocks:
            idx = block_to_idx[b]
            if np.any(cover2[block_indices[idx]] >= 1):
                kept.append(b)
        return kept

    def _local_search(blocks, deadline):
        """Swap-improve: replace a block with an alternative that preserves coverage
        and enables further redundancy removal."""
        if not blocks:
            return blocks
        cover = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            cover[block_to_idx[b]] += 1

        def try_remove():
            nonlocal cover
            changed = True
            while changed:
                if time.time() > deadline:
                    return
                changed = False
                for i in range(len(blocks) - 1, -1, -1):
                    idx = block_to_idx[blocks[i]]
                    redundant = True
                    for j in block_indices[idx]:
                        if cover[j] <= 1:
                            redundant = False
                            break
                    if redundant:
                        cover[block_indices[idx]] -= 1
                        blocks.pop(i)
                        changed = True

        try_remove()

        rng = random.Random(12345)
        for _it in range(4):
            if time.time() > deadline:
                break
            progress = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                orig = blocks[i]
                oidx = block_to_idx[orig]
                # critical triples only this block covers
                cover[block_indices[oidx]] -= 1
                critical = [j for j in block_indices[oidx] if cover[j] == 0]
                # try a few random replacements
                best_cand = None
                for _a in range(4):
                    cand = tuple(sorted(rng.sample(range(v), k)))
                    cidx = block_to_idx[cand]
                    if critical:
                        ct = set(block_indices[cidx].tolist())
                        if all(c in ct for c in critical):
                            best_cand = (cand, cidx)
                            break
                    else:
                        best_cand = (cand, cidx)
                        break
                if best_cand is not None:
                    cand, cidx = best_cand
                    if cand != orig:
                        cover[block_indices[cidx]] += 1
                        blocks[i] = cand
                        progress = True
                        try_remove()
                        if len(blocks) < 80:
                            break
                    else:
                        cover[block_indices[oidx]] += 1
                else:
                    cover[block_indices[oidx]] += 1
            if not progress:
                break
        try_remove()
        return blocks

    def _greedy_full(uncovered, mode, seed_val, deadline):
        """Anchor-greedy over the full block matrix with UCB-style exploration.
        mode: 'greedy', 'weighted', 'ucb'
        """
        rng = random.Random(seed_val)
        selected = []
        # per-block selection counts for UCB exploration
        sel_count = np.zeros(n_blocks, dtype=np.int32)
        total_picks = 0
        while np.any(uncovered):
            if time.time() > deadline:
                return None
            if mode == 'weighted':
                w = np.where(uncovered, inv_freq, 0.0)
                gains = w[block_indices].sum(axis=1)
                best_idx = int(np.argmax(gains))
            else:
                gains = uncovered[block_indices].sum(axis=1)
                if mode == 'ucb':
                    # UCB: exploitation (gain) + exploration (uncertainty over rarely picked)
                    total_picks += 1
                    # exploration term: favor blocks picked rarely
                    explore = 1.0 / (sel_count + 1.0)
                    # normalize gains to a comparable scale (max gain ~ C(k,t)=35)
                    score = gains + 0.15 * gains.max() * explore
                    best_idx = int(np.argmax(score))
                else:
                    best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                # should not happen; fall back to any uncovered triple
                break
            selected.append(best_idx)
            sel_count[best_idx] += 1
            uncovered[block_indices[best_idx]] = False

        if np.any(uncovered):
            return None
        return [all_blocks[i] for i in selected]

    best_blocks = None
    best_len = float('inf')

    trial = 0
    while time.time() < deadline - 25:  # reserve 25s for final local search
        trial += 1
        mode = trial % 3  # cycle greedy, weighted, ucb
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = _greedy_full(uncovered, mode, trial, deadline)
        if blocks is None:
            continue
        blocks = _remove_redundant(blocks, deadline)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        if best_len <= 76:
            break

    # Final local search on the best
    if best_blocks is not None and time.time() < deadline - 3:
        improved = _local_search(list(best_blocks), deadline)
        if len(improved) < best_len:
            best_blocks = improved
            best_len = len(improved)

    if best_blocks is None:
        # Fallback deterministic greedy
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = _greedy_full(uncovered, 'greedy', 0, deadline)
        if blocks is not None:
            best_blocks = _remove_redundant(blocks, deadline)
            best_blocks = _local_search(list(best_blocks), deadline)
        else:
            best_blocks = []

    return [list(b) for b in best_blocks]
