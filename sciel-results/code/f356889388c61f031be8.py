import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin vs 150s kill

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))

    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)

    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Coverage matrix: Cover[i, j] = 1 if t-subset j is in block i
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            Cover[i, tsub_to_idx[ts]] = 1

    # Inverted index: for each t-subset, blocks that cover it
    inverted_index = [np.where(Cover[:, j] == 1)[0] for j in range(n_tsubs)]

    # Bit-packed mask of selected blocks (fast gain masking)
    n_words = (n_blocks + 63) // 64
    block_bits = np.zeros(n_words, dtype=np.uint64)

    def bits_set(i):
        block_bits[i >> 6] |= (np.uint64(1) << (i & 63))

    def bits_clear(i):
        block_bits[i >> 6] &= ~(np.uint64(1) << (i & 63))

    def bits_get(i):
        return bool((block_bits[i >> 6] >> (i & 63)) & np.uint64(1))

    def gains_masked(gains, selected_set):
        # zero out already-selected blocks via bit mask (vectorized)
        sel_mask = (block_bits != 0)
        if np.any(sel_mask):
            idxs = np.where(sel_mask)[0]
            for w in idxs:
                m = block_bits[w]
                while m:
                    b = (m & (-m)).bit_length() - 1
                    i = int(w) * 64 + b
                    if i < n_blocks and i in selected_set:
                        gains[i] = 0
                    m &= m - 1
        return gains

    def greedy_init(weighted=False, randomize=False, max_iter=20000, seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=np.uint8)
        selected = []
        selected_set = set()

        if weighted:
            # weight uncovered subsets by 1/rarity to prioritize scarce ones
            inv_count = np.zeros(n_tsubs, dtype=np.float64)
            for j in range(n_tsubs):
                c = len(inverted_index[j])
                inv_count[j] = 1.0 / c if c > 0 else 0.0
            w = inv_count
        else:
            w = None

        for _ in range(max_iter):
            if time.time() > deadline - 30.0:
                break
            if not uncovered.any():
                break
            if w is not None:
                gains = (Cover @ (uncovered * w).astype(np.float64)).astype(np.float64)
            else:
                gains = (Cover @ uncovered).astype(np.float64)

            if randomize:
                gains = gains + rng.random() * 0.05
                gains[gains < 1e-6] = 0.0

            gains = gains_masked(gains, selected_set)
            best = int(np.argmax(gains))
            if gains[best] <= 1e-9:
                break
            selected.append(best)
            selected_set.add(best)
            bits_set(best)
            uncovered = uncovered & (Cover[best] == 0)

        return selected, selected_set, uncovered

    # --- Phase 1: multiple greedy restarts (plain + scarcity-weighted) ---
    best_selected = None
    best_set = None
    best_count = float('inf')

    restarts = [(False, False, 0), (False, False, 1), (True, False, 2),
                (True, True, 3), (True, True, 4), (False, True, 5), (True, True, 6)]
    for (wt, rnd, sd) in restarts:
        if time.time() > deadline - 25.0:
            break
        sel, ss, unc = greedy_init(weighted=wt, randomize=rnd, seed=sd)
        if unc is not None and not unc.any():
            if len(sel) < best_count:
                best_count = len(sel)
                best_selected = sel
                best_set = ss

    if best_selected is None:
        best_selected, best_set, _ = greedy_init(weighted=False, randomize=False, seed=0)

    selected = best_selected
    selected_set = best_set

    # cover counts
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for i in selected:
        cover_counts += Cover[i]

    def remove_redundants():
        nonlocal cover_counts
        changed = True
        while changed and time.time() < deadline - 12.0:
            changed = False
            new_sel = []
            for idx in selected:
                if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                    new_sel.append(idx)
                else:
                    cover_counts -= Cover[idx]
                    selected_set.discard(idx)
                    bits_clear(idx)
                    changed = True
            selected[:] = new_sel

    remove_redundants()

    # --- Phase 2: local search (1-for-1, 2-for-1, add-minimize) ---
    def try_1for1():
        nonlocal cover_counts
        for i_idx in list(selected):
            if time.time() > deadline - 8.0:
                return False
            critical = (Cover[i_idx] == 1) & (cover_counts == 1)
            cc = int(critical.sum())
            if cc == 0:
                cover_counts -= Cover[i_idx]
                selected.remove(i_idx)
                selected_set.discard(i_idx)
                bits_clear(i_idx)
                remove_redundants()
                return True
            c_idx = int(np.where(critical)[0][0])
            for j_idx in inverted_index[c_idx]:
                if j_idx in selected_set:
                    continue
                if int((Cover[j_idx] & critical).sum()) == cc:
                    cover_counts -= Cover[i_idx]
                    cover_counts += Cover[j_idx]
                    selected.remove(i_idx)
                    selected_set.discard(i_idx)
                    bits_clear(i_idx)
                    selected.append(int(j_idx))
                    selected_set.add(int(j_idx))
                    bits_set(int(j_idx))
                    remove_redundants()
                    return True
        return False

    def try_2for1():
        nonlocal cover_counts
        n_sel = len(selected)
        for a in range(n_sel):
            if time.time() > deadline - 8.0:
                return False
            for b in range(a + 1, n_sel):
                if time.time() > deadline - 8.0:
                    return False
                i_idx = selected[a]
                j_idx = selected[b]
                mi = (Cover[i_idx] == 1) & (cover_counts == 1)
                mj = (Cover[j_idx] == 1) & (cover_counts == 1)
                comb = mi | mj
                cc = int(comb.sum())
                if cc == 0:
                    cover_counts -= Cover[i_idx]
                    cover_counts -= Cover[j_idx]
                    for x in (i_idx, j_idx):
                        selected.remove(x)
                        selected_set.discard(x)
                        bits_clear(x)
                    remove_redundants()
                    return True
                c_idx = int(np.where(comb)[0][0])
                for cand in inverted_index[c_idx]:
                    if cand in selected_set:
                        continue
                    if int((Cover[cand] & comb).sum()) == cc:
                        cover_counts -= Cover[i_idx]
                        cover_counts -= Cover[j_idx]
                        cover_counts += Cover[cand]
                        for x in (i_idx, j_idx):
                            selected.remove(x)
                            selected_set.discard(x)
                            bits_clear(x)
                        selected.append(int(cand))
                        selected_set.add(int(cand))
                        bits_set(int(cand))
                        remove_redundants()
                        return True
        return False

    def try_add_minimize():
        # add one block that covers many currently-single-covered (critical) subsets,
        # then drop blocks that become redundant -> can escape local minima
        nonlocal cover_counts
        if time.time() > deadline - 8.0:
            return False
        # candidate: blocks covering the most critical (count==1) subsets
        crit = (cover_counts == 1)
        if not crit.any():
            return False
        gains = (Cover @ crit.astype(np.uint8)).astype(np.int64)
        gains = gains_masked(gains, selected_set)
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            return False
        # tentatively add
        cover_counts += Cover[best]
        selected.append(best)
        selected_set.add(best)
        bits_set(best)
        before = len(selected)
        remove_redundants()
        if len(selected) < before:
            return True
        else:
            # revert
            cover_counts -= Cover[best]
            selected.remove(best)
            selected_set.discard(best)
            bits_clear(best)
            return False

    # alternating local search
    for _ in range(200):
        if time.time() > deadline - 6.0:
            break
        improved = False
        if try_1for1():
            improved = True
        if try_2for1():
            improved = True
        if try_add_minimize():
            improved = True
        if not improved:
            break

    # --- Phase 3: high-precision polish (tighter margin, more 1-for-1 sweeps) ---
    for _ in range(120):
        if time.time() > deadline - 3.0:
            break
        if try_1for1():
            continue
        if try_add_minimize():
            continue
        if try_2for1():
            continue
        break

    remove_redundants()

    blocks = [all_kblocks[i] for i in selected]

    # Safety verification
    if time.time() < deadline - 1.0:
        final_cover = np.zeros(n_tsubs, dtype=np.uint8)
        for i in selected:
            final_cover |= Cover[i]
        if (final_cover == 0).any():
            sel, ss, _ = greedy_init(weighted=False, randomize=False, seed=0)
            blocks = [all_kblocks[i] for i in sel]

    return blocks
