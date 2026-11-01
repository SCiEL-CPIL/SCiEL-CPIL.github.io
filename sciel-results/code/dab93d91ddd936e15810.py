import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start = time.time()
    deadline = start + 118.0  # safety margin before 120s kill

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Sparse coverage: for each block, list of t-subset indices it covers.
    # Build via vectorized mapping (no Python loop over tsubs).
    sub_to_idx_arr = np.empty((n_blocks, int(np.prod([k - i for i in range(t)]))), dtype=np.int64)
    # Precompute the index of each t-subset of each k-block via a single pass.
    # Use itertools per block but vectorize the lookup with a dict -> array.
    # n_blocks=19503, C(7,4)=35 -> 682605 lookups, fast.
    for i, B in enumerate(all_kblocks):
        idxs = [tsub_to_idx[ts] for ts in itertools.combinations(B, t)]
        sub_to_idx_arr[i] = idxs
    # Per-block t-subset indices as a list of arrays for fast slicing.
    block_tsubs = [sub_to_idx_arr[i] for i in range(n_blocks)]

    # Dense coverage matrix for fast redundant-removal / swap checks.
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    np.add.at(Cover, (np.repeat(np.arange(n_blocks), 35), sub_to_idx_arr.ravel()), 1)

    def is_valid(selected):
        if len(selected) == 0:
            return n_tsubs == 0
        cov = np.zeros(n_tsubs, dtype=np.uint8)
        for idx in selected:
            cov |= Cover[idx]
        return bool(np.all(cov > 0))

    def gain_matrix(selected, uncovered):
        # gain[i] = number of uncovered tsubs in block i
        gains = np.zeros(n_blocks, dtype=np.int64)
        for idx in selected:
            pass
        # Vectorized: gains = Cover @ uncovered (Cover uint8, uncovered uint8 -> int)
        return Cover @ uncovered

    def greedy(seed, budget):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=np.uint8)
        selected = []
        # Precompute a random tie-break order to diversify starts.
        order = list(range(n_blocks))
        rng.shuffle(order)
        # Random initial perturbation: pick a few random blocks first (small).
        n_init = rng.randint(0, 3)
        for _ in range(n_init):
            idx = order.pop()
            if not np.any(Cover[idx] & uncovered):
                continue
            selected.append(idx)
            uncovered = uncovered & (Cover[idx] == 0)
        while np.any(uncovered) and time.time() < budget:
            gains = Cover @ uncovered
            # Add small random noise to break ties / diversify
            noise = rng.random(n_blocks) * 0.5
            gains_f = gains + noise
            best = int(np.argmax(gains_f))
            if gains[best] == 0:
                # No block covers anything uncovered -> should not happen
                break
            selected.append(best)
            uncovered = uncovered & (Cover[best] == 0)
        return selected

    def remove_redundant(selected, budget):
        cover_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in selected:
            cover_counts += Cover[idx]
        changed = True
        while changed and time.time() < budget:
            changed = False
            new_sel = []
            for idx in selected:
                if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                    new_sel.append(idx)
                else:
                    cover_counts -= Cover[idx]
                    changed = True
            selected = new_sel
        return selected

    def swap_improve(selected, budget):
        # Try to replace a pair of blocks with one better block, or drop one.
        # Simple pass: for each block, see if removing it keeps validity after
        # adding a single alternative. Limited to time budget.
        cover_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in selected:
            cover_counts += Cover[idx]
        improved = True
        while improved and time.time() < budget:
            improved = False
            for pos in range(len(selected)):
                idx = selected[pos]
                # tentatively remove
                cover_counts -= Cover[idx]
                if np.any(cover_counts == 0):
                    # need to add a block covering the now-uncovered tsubs
                    missing = np.where(cover_counts == 0)[0]
                    # find a single block covering all missing
                    cand = None
                    for j in range(n_blocks):
                        if j == idx:
                            continue
                        if np.all(Cover[j][missing] == 1):
                            cand = j
                            break
                    if cand is not None:
                        # replacing idx with cand; counts: remove idx (done), add cand
                        cover_counts += Cover[cand]
                        selected[pos] = cand
                        improved = True
                        break
                    else:
                        cover_counts += Cover[idx]  # restore
                else:
                    # removing idx keeps validity -> drop it
                    selected.pop(pos)
                    improved = True
                    break
        return selected

    best = None
    best_len = 10**9
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    rng_global = random.Random(12345)
    # First: one deterministic greedy to get a valid baseline quickly.
    base = greedy(0, deadline - 30)
    base = remove_redundant(base, deadline - 10)
    if is_valid(base):
        best = base
        best_len = len(base)
    # Multi-start with remaining time.
    for s in seeds:
        if time.time() > deadline - 12:
            break
        # Reserve time for removal + swap of this candidate.
        cand_budget = deadline - 12
        cand = greedy(s, cand_budget)
        if not is_valid(cand):
            continue
        cand = remove_redundant(cand, deadline - 6)
        if is_valid(cand) and len(cand) < best_len:
            cand = swap_improve(cand, deadline - 3)
            if is_valid(cand):
                best = cand
                best_len = len(cand)
    if best is None:
        # Fallback: deterministic greedy, force valid.
        best = greedy(0, deadline - 2)
    # Final safety: ensure validity; if not, do a quick repair.
    if not is_valid(best):
        uncovered = np.zeros(n_tsubs, dtype=np.uint8)
        cov = np.zeros(n_tsubs, dtype=np.uint8)
        for idx in best:
            cov |= Cover[idx]
        uncovered = (cov == 0)
        while np.any(uncovered) and time.time() < deadline - 1:
            gains = Cover @ uncovered
            b = int(np.argmax(gains))
            if gains[b] == 0:
                break
            best.append(b)
            uncovered = uncovered & (Cover[b] == 0)
    # Convert to list of lists of ints.
    blocks = [list(all_kblocks[i]) for i in best]
    # Ensure distinctness (greedy may pick same block twice with noise? gains>0 prevents repeats mostly, but dedupe safely).
    seen = set()
    out = []
    for b in blocks:
        t = tuple(b)
        if t not in seen:
            seen.add(t)
            out.append(b)
    return out
