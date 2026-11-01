"""Fresh design for covering design C(v,k,t): global greedy set cover with
Lagrangian dual-weighted gain + redundancy minimalization, under a deadline guard."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _count_table(v, k, t):
    """Map each t-subset (as a tuple) to an integer index; return (table, n_t)."""
    table = {}
    n_t = 0
    for c in itertools.combinations(range(v), t):
        table[c] = n_t
        n_t += 1
    return table, n_t


def construct(v, k, t):
    start = time.time()
    deadline = start + 130.0  # safety margin vs 150s kill

    # --- precompute t-subset index table and per-point t-subset lists ---
    table, n_t = _count_table(v, k, t)
    ptsub = [[] for _ in range(v)]
    for c in itertools.combinations(range(v), t):
        idx = table[c]
        for x in c:
            ptsub[x].append(idx)

    # Precompute all k-subsets and their t-subset index lists for fast gain lookup
    all_ksubs = list(itertools.combinations(range(v), k))
    ksub_tsets = []
    for comb in all_ksubs:
        tset = []
        for c in itertools.combinations(comb, t):
            tset.append(table[c])
        ksub_tsets.append(tset)
    ksub_tsets_arr = np.array(ksub_tsets, dtype=np.int32)

    # --- Lagrangian-guided greedy set cover ---
    covered = np.zeros(n_t, dtype=np.int8)
    blocks = []
    block_tsets = []  # list of numpy arrays of t-subset indices per block

    # Dual weights: initially uniform, updated via subgradient
    w = np.ones(n_t, dtype=np.float64)
    w_scale = 1.0

    def add_block(bl, tset_arr):
        for idx in tset_arr:
            covered[idx] = 1
        blocks.append(bl)
        block_tsets.append(tset_arr)

    # Number of greedy rounds
    max_rounds = 200
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        unc = int(np.count_nonzero(covered == 0))
        if unc == 0:
            break

        # Compute gain for each k-subset: sum of weights for uncovered t-subsets
        # Vectorized: for each ksub, sum w[covered[tset]==0]
        # Use matrix multiply approach: build a binary matrix M[n_ksub, n_t] would be too large
        # Instead, loop but with numpy inner ops
        gains = np.zeros(len(all_ksubs), dtype=np.float64)
        for i, tset in enumerate(ksub_tsets):
            tset_arr = ksub_tsets_arr[i]
            gains[i] = w[tset_arr][covered[tset_arr] == 0].sum()

        best_idx = int(np.argmax(gains))
        best_gain = gains[best_idx]

        if best_gain <= 0:
            break

        best_comb = all_ksubs[best_idx]
        bl = list(best_comb)
        bl.sort()
        add_block(bl, ksub_tsets_arr[best_idx])

        # Update dual weights via subgradient descent
        # Subgradient: for uncovered t-subsets, increase weight; for covered, decrease
        # Step size: diminishing
        step = 2.0 / (rnd + 1.0)
        unc_mask = covered == 0
        # Gradient of the Lagrangian w.r.t. weights:
        # dL/dw_j = -1 if j is covered (penalty term), +1 if not covered
        # We want to increase w for uncovered, decrease for covered
        w[unc_mask] += step
        w[~unc_mask] -= step * 0.5
        # Clip to positive
        w = np.maximum(w, 0.1)
        # Renormalize to prevent drift
        w_scale = w.mean()
        if w_scale > 0:
            w /= w_scale

    # --- Minimalization: remove any block whose removal leaves coverage intact ---
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        new_blocks = []
        new_tsets = []
        for i, tset_arr in enumerate(block_tsets):
            # temporarily remove this block
            for idx in tset_arr:
                covered[idx] = 0
            ok = True
            for idx in tset_arr:
                if covered[idx] == 0:
                    ok = False
                    break
            if not ok:
                # keep it: restore
                for idx in tset_arr:
                    covered[idx] = 1
                new_blocks.append(blocks[i])
                new_tsets.append(tset_arr)
            else:
                # safe to drop
                changed = True
        blocks = new_blocks
        block_tsets = new_tsets

    return blocks
# EVOLVE-BLOCK-END
