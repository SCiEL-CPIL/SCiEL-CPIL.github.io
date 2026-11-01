import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 148.0

    def comb(n, r):
        if r > n or r < 0:
            return 0
        r = min(r, n - r)
        if r == 0:
            return 1
        res = 1
        for i in range(1, r + 1):
            res = res * (n - i + 1) // i
        return res

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)
    ts_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    block_to_ts_idx = []
    ts_to_block_idx = [[] for _ in range(num_t)]
    for j, blk in enumerate(all_k_blocks):
        idxs = []
        for ts in itertools.combinations(blk, t):
            idx = ts_index[ts]
            idxs.append(idx)
            ts_to_block_idx[idx].append(j)
        block_to_ts_idx.append(idxs)

    # Precompute numpy arrays for each block's t-subsets
    block_ts_arrays = [np.array(x, dtype=np.int32) for x in block_to_ts_idx]
    
    # Precompute t-subset combinations for each block as tuples for faster lookup
    block_t_tuples = []
    for blk in all_k_blocks:
        block_t_tuples.append(tuple(itertools.combinations(blk, t)))

    def greedy(seed_val, alpha=1.0):
        rng = np.random.RandomState(seed_val)
        gain = np.full(num_k, comb(k, t), dtype=np.int32)
        uncovered = np.ones(num_t, dtype=bool)
        selected = np.zeros(num_k, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 12.0:
                return None
            mg = int(np.max(gain))
            if mg <= 0:
                break
            cands = np.where((gain == mg) & (~selected))[0]
            if len(cands) == 0:
                break
            if alpha < 1.0 and len(cands) > 1:
                w = np.ones(len(cands)) + rng.rand(len(cands)) * 0.1
                p = w / w.sum()
                b = cands[rng.choice(len(cands), p=p)]
            else:
                b = cands[rng.randint(0, len(cands))]
            sol.append(int(b))
            selected[b] = True
            gain[b] = -1
            for ti in block_to_ts_idx[b]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    for nb in ts_to_block_idx[ti]:
                        if not selected[nb]:
                            gain[nb] -= 1
        return sol

    best_idx = None
    best_len = float('inf')

    # Deterministic greedy
    s0 = greedy(0, 1.0)
    if s0 is not None:
        best_idx = s0
        best_len = len(s0)

    # Randomized greedy - more diverse
    for s in range(1, 100):
        if time.time() > deadline - 18.0:
            break
        a = 1.0 + (s % 10) * 0.12
        sol = greedy(s, a)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_idx = sol

    def idx_to_blocks(idxs):
        return [list(all_k_blocks[i]) for i in idxs]

    blocks = idx_to_blocks(best_idx)

    # Fast redundancy removal using numpy
    def remove_redundant(blks):
        if not blks:
            return blks
        cc = np.zeros(num_t, dtype=np.int32)
        for b_idx in range(len(blks)):
            cc[block_ts_arrays[b_idx]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(blks)):
                if time.time() > deadline - 4.0:
                    return blks
                # Check if all t-subsets in block i are covered by other blocks
                mask = cc[block_ts_arrays[i]] == 1
                if not mask.any():
                    # This block is redundant
                    cc[block_ts_arrays[i]] -= 1
                    blks.pop(i)
                    changed = True
                    break
        return blks

    # Fast swap search
    def swap_search(blks, passes=30):
        if not blks:
            return blks, False
        cc = np.zeros(num_t, dtype=np.int32)
        for b_idx in range(len(blks)):
            cc[block_ts_arrays[b_idx]] += 1
        sol_set = set()
        for b in blks:
            sol_set.add(tuple(b))
        improved = False
        for _ in range(passes):
            if time.time() > deadline - 5.0:
                break
            found = False
            for i in range(len(blks)):
                if time.time() > deadline - 5.0:
                    break
                # Find t-subsets uniquely covered by block i
                arr_i = block_ts_arrays[i]
                unique_mask = cc[arr_i] == 1
                ut = arr_i[unique_mask]
                if len(ut) == 0:
                    continue
                # Find candidate blocks that cover all unique t-subsets
                cands = set(ts_to_block_idx[ut[0]])
                for ti in ut[1:]:
                    cands.intersection_update(ts_to_block_idx[ti])
                    if not cands:
                        break
                if not cands:
                    continue
                valid = [c for c in cands if tuple(all_k_blocks[c]) not in sol_set]
                if not valid:
                    continue
                ni = random.choice(valid)
                nb = list(all_k_blocks[ni])
                old = blks[i]
                cc[block_ts_arrays[i]] -= 1
                cc[block_ts_arrays[ni]] += 1
                sol_set.discard(tuple(old))
                sol_set.add(tuple(nb))
                blks[i] = nb
                improved = True
                found = True
                break
            if not found:
                break
        return blks, improved

    # 2-exchange (faster version)
    def two_exchange(blks):
        if len(blks) < 2:
            return blks, False
        cc = np.zeros(num_t, dtype=np.int32)
        for b_idx in range(len(blks)):
            cc[block_ts_arrays[b_idx]] += 1
        sol_set = set()
        for b in blks:
            sol_set.add(tuple(b))
        improved = False
        n = len(blks)
        for i in range(n):
            if time.time() > deadline - 3.0:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 3.0:
                    break
                arr_i = block_ts_arrays[i]
                arr_j = block_ts_arrays[j]
                uti = arr_i[cc[arr_i] == 1]
                utj = arr_j[cc[arr_j] == 1]
                au = np.concatenate([uti, utj])
                if len(au) == 0:
                    continue
                # Try to replace both with one block
                cands = set(ts_to_block_idx[au[0]])
                for ti in au[1:]:
                    cands.intersection_update(ts_to_block_idx[ti])
                    if not cands:
                        break
                if cands:
                    valid = [c for c in cands if tuple(all_k_blocks[c]) not in sol_set]
                    if valid:
                        ni = random.choice(valid)
                        nb = list(all_k_blocks[ni])
                        oi, oj = blks[i], blks[j]
                        cc[block_ts_arrays[i]] -= 1
                        cc[block_ts_arrays[j]] -= 1
                        cc[block_ts_arrays[ni]] += 1
                        sol_set.discard(tuple(oi))
                        sol_set.discard(tuple(oj))
                        sol_set.add(tuple(nb))
                        blks.pop(j)
                        blks[i] = nb
                        improved = True
                        break
                # Try to replace with two different blocks
                if len(uti) > 0:
                    ci = set(ts_to_block_idx[uti[0]])
                    for ti in uti[1:]:
                        ci.intersection_update(ts_to_block_idx[ti])
                        if not ci:
                            break
                else:
                    ci = set(range(num_k))
                if len(utj) > 0:
                    cj = set(ts_to_block_idx[utj[0]])
                    for ti in utj[1:]:
                        cj.intersection_update(ts_to_block_idx[ti])
                        if not cj:
                            break
                else:
                    cj = set(range(num_k))
                vi = [c for c in ci if tuple(all_k_blocks[c]) not in sol_set]
                vj = [c for c in cj if tuple(all_k_blocks[c]) not in sol_set and c not in ci]
                if vi and vj:
                    ni, nj = random.choice(vi), random.choice(vj)
                    nb1, nb2 = list(all_k_blocks[ni]), list(all_k_blocks[nj])
                    oi, oj = blks[i], blks[j]
                    cc[block_ts_arrays[i]] -= 1
                    cc[block_ts_arrays[j]] -= 1
                    cc[block_ts_arrays[ni]] += 1
                    cc[block_ts_arrays[nj]] += 1
                    sol_set.discard(tuple(oi))
                    sol_set.discard(tuple(oj))
                    sol_set.add(tuple(nb1))
                    sol_set.add(tuple(nb2))
                    blks[i], blks[j] = nb1, nb2
                    improved = True
                    break
            if improved:
                break
        return blks, improved

    # Main loop - more iterations with faster operations
    for _ in range(40):
        if time.time() > deadline - 6.0:
            break
        blocks, imp = swap_search(blocks, 25)
        if imp:
            blocks = remove_redundant(blocks)
            if len(blocks) < best_len:
                best_len = len(blocks)
        else:
            blocks, imp2 = two_exchange(blocks)
            if imp2:
                blocks = remove_redundant(blocks)
                if len(blocks) < best_len:
                    best_len = len(blocks)
            else:
                break

    if time.time() < deadline - 3.0:
        for _ in range(20):
            if time.time() > deadline - 2.0:
                break
            blocks, imp = swap_search(blocks, 15)
            if imp:
                blocks = remove_redundant(blocks)
                if len(blocks) < best_len:
                    best_len = len(blocks)
            else:
                break

    return blocks
