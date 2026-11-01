import numpy as np
import itertools
import time
import random
import sys


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s

    points = list(range(v))

    # ---------- Precomputation ----------
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_idx = {sub: i for i, sub in enumerate(t_subsets)}

    k_blocks = list(itertools.combinations(points, k))
    n_b = len(k_blocks)

    # For each block: list of t-subset indices it covers
    block_cover = []
    for bi, block in enumerate(k_blocks):
        subs = list(itertools.combinations(block, t))
        idx = np.array([t_idx[s] for s in subs], dtype=np.int32)
        block_cover.append(idx)

    k_choose_t = len(block_cover[0])

    # For each t-subset: which blocks cover it
    tsub_blocks = [[] for _ in range(n_t)]
    for bi, inds in enumerate(block_cover):
        for s in inds:
            tsub_blocks[s].append(bi)
    tsub_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_blocks]

    # 64-bit masks for fast coverage checking
    point_bit = {p: (1 << p) for p in points}
    block_mask = np.zeros(n_b, dtype=np.uint64)
    for bi, block in enumerate(k_blocks):
        m = 0
        for p in block:
            m |= point_bit[p]
        block_mask[bi] = m

    subset_mask = np.zeros(n_t, dtype=np.uint64)
    for i, sub in enumerate(t_subsets):
        m = 0
        for p in sub:
            m |= point_bit[p]
        subset_mask[i] = m

    best_blocks = None
    best_len = float('inf')

    def time_ok():
        return time.time() < deadline

    def greedy(top_k=15):
        """Greedy set cover with randomized tie-breaking. Returns (selected_indices, completed)."""
        gains = np.full(n_b, k_choose_t, dtype=np.int32)
        uncovered = np.ones(n_t, dtype=bool)
        selected = []
        unc_count = n_t
        while unc_count > 0:
            if not time_ok():
                return selected, False
            # Randomized selection from top-k to break symmetry
            if n_b > top_k and random.random() < 0.6:
                top_indices = np.argpartition(gains, -top_k)[-top_k:]
                b = int(random.choice(top_indices))
            else:
                b = int(np.argmax(gains))
            
            if gains[b] == 0:
                return selected, False
            
            selected.append(b)
            inds = block_cover[b]
            unique = inds[uncovered[inds]]
            if len(unique) > 0:
                unc_count -= len(unique)
                uncovered[unique] = False
                to_dec = []
                for s in unique:
                    to_dec.append(tsub_blocks_np[s])
                if to_dec:
                    flat = np.concatenate(to_dec)
                    np.add.at(gains, flat, -1)
        return selected, True

    def compute_cover_count(selected):
        cc = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            cc[block_cover[b]] += 1
        return cc

    def local_search(selected):
        """Redundancy removal + 1-for-1 swaps + 2-for-1 merges + 2-for-2 pair swaps."""
        cc = compute_cover_count(selected)
        improved = True
        max_iterations = 50
        iteration = 0
        
        while improved and time_ok() and iteration < max_iterations:
            improved = False
            iteration += 1

            # --- Redundancy removal ---
            redundant = []
            for b in selected:
                if np.min(cc[block_cover[b]]) > 1:
                    redundant.append(b)
            if redundant:
                for b in redundant:
                    selected.remove(b)
                    cc[block_cover[b]] -= 1
                improved = True
                continue

            # --- Compute unique subsets per block ---
            uniq = {}
            for b in selected:
                subs = block_cover[b]
                u = [s for s in subs if cc[s] == 1]
                uniq[b] = u

            # --- 1-for-1 swap ---
            min_u = min(len(u) for u in uniq.values()) if uniq else 0
            if min_u > 0:
                cands_out = [b for b, u in uniq.items() if len(u) == min_u]
                random.shuffle(cands_out)
                swap_done = False
                for b_out in cands_out:
                    if not time_ok():
                        break
                    s_uniq = uniq[b_out]
                    if not s_uniq:
                        continue
                    # Find blocks that cover ALL unique subsets of b_out
                    cands_in = None
                    for s in s_uniq:
                        if cands_in is None:
                            cands_in = set(tsub_blocks_np[s].tolist())
                        else:
                            cands_in &= set(tsub_blocks_np[s].tolist())
                    
                    if cands_in:
                        exclude = set(selected)
                        valid = [c for c in cands_in if c not in exclude]
                        if valid:
                            b_in = random.choice(valid)
                            selected.remove(b_out)
                            selected.append(b_in)
                            cc[block_cover[b_out]] -= 1
                            cc[block_cover[b_in]] += 1
                            swap_done = True
                            improved = True
                            break
                if swap_done:
                    continue

            # --- 2-for-1 merge ---
            if len(selected) > 2 and time_ok():
                small = [b for b in selected if len(uniq[b]) <= 10]
                random.shuffle(small)
                merge_done = False
                for i in range(len(small)):
                    if not time_ok():
                        break
                    for j in range(i + 1, len(small)):
                        b1, b2 = small[i], small[j]
                        u1 = uniq[b1]
                        u2 = uniq[b2]
                        combined = list(set(u1) | set(u2))
                        if len(combined) == 0:
                            continue
                        
                        # Find a block covering all subsets in combined
                        cands = None
                        for s in combined:
                            if cands is None:
                                cands = set(tsub_blocks_np[s].tolist())
                            else:
                                cands &= set(tsub_blocks_np[s].tolist())
                        
                        if cands:
                            exclude = {b1, b2}
                            valid = [c for c in cands if c not in exclude]
                            if valid:
                                c = random.choice(valid)
                                selected.remove(b1)
                                selected.remove(b2)
                                selected.append(c)
                                cc[block_cover[b1]] -= 1
                                cc[block_cover[b2]] -= 1
                                cc[block_cover[c]] += 1
                                merge_done = True
                                improved = True
                                break
                    if merge_done:
                        break
                if merge_done:
                    continue

            # --- 2-for-2 pair swap ---
            if len(selected) >= 4 and time_ok():
                small = [b for b in selected if len(uniq[b]) <= 8]
                if len(small) >= 2:
                    random.shuffle(small)
                    pair_swap_done = False
                    n_pairs_to_try = min(8, len(small) - 1)
                    for i in range(n_pairs_to_try):
                        if not time_ok():
                            break
                        b1, b2 = small[i], small[i + 1]
                        u1 = uniq[b1]
                        u2 = uniq[b2]
                        if not u1 or not u2:
                            continue
                        
                        # Try: c1 covers u1, c2 covers u2
                        cands1 = None
                        for s in u1:
                            if cands1 is None:
                                cands1 = set(tsub_blocks_np[s].tolist())
                            else:
                                cands1 &= set(tsub_blocks_np[s].tolist())
                        
                        cands2 = None
                        for s in u2:
                            if cands2 is None:
                                cands2 = set(tsub_blocks_np[s].tolist())
                            else:
                                cands2 &= set(tsub_blocks_np[s].tolist())
                        
                        if cands1 and cands2:
                            exclude = {b1, b2}
                            valid1 = [c for c in cands1 if c not in exclude]
                            valid2 = [c for c in cands2 if c not in exclude]
                            
                            if valid1 and valid2:
                                # Try a few random combinations
                                for _ in range(3):
                                    c1 = random.choice(valid1)
                                    c2 = random.choice(valid2)
                                    if c1 != c2:
                                        selected.remove(b1)
                                        selected.remove(b2)
                                        selected.append(c1)
                                        selected.append(c2)
                                        cc[block_cover[b1]] -= 1
                                        cc[block_cover[b2]] -= 1
                                        cc[block_cover[c1]] += 1
                                        cc[block_cover[c2]] += 1
                                        pair_swap_done = True
                                        improved = True
                                        break
                        if pair_swap_done:
                            break
                    if pair_swap_done:
                        continue

            break

        return selected

    # ---------- Main loop: multiple restarts ----------
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 
             7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000,
             17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000]
    si = 0
    while time_ok():
        seed = seeds[si % len(seeds)]
        si += 1
        random.seed(seed)
        np.random.seed(seed)

        # Vary top_k for diversity
        top_k = 10 + (si % 20)
        selected, completed = greedy(top_k=top_k)
        if not completed:
            continue

        selected = local_search(selected)

        cur_len = len(selected)
        if cur_len < best_len:
            best_len = cur_len
            best_blocks = [list(k_blocks[i]) for i in selected]

        if time.time() > deadline - 8.0:
            break

    if best_blocks is None:
        selected, completed = greedy(top_k=15)
        if completed:
            best_blocks = [list(k_blocks[i]) for i in selected]
        else:
            # Fallback: just take all blocks (shouldn't happen)
            best_blocks = [list(b) for b in k_blocks[:200]]

    return best_blocks
