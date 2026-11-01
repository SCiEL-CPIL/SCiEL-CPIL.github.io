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

    # numpy arrays
    block_ts = [np.array(x, dtype=np.int32) for x in block_to_ts_idx]

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

    # Randomized greedy
    for s in range(1, 80):
        if time.time() > deadline - 18.0:
            break
        a = 1.0 + (s % 7) * 0.15
        sol = greedy(s, a)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_idx = sol

    def idx_to_blocks(idxs):
        return [list(all_k_blocks[i]) for i in idxs]

    blocks = idx_to_blocks(best_idx)

    # Redundancy removal
    def remove_redundant(blks):
        if not blks:
            return blks
        cc = np.zeros(num_t, dtype=np.int32)
        for b in blks:
            for ts in itertools.combinations(b, t):
                cc[ts_index[ts]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(blks)):
                if time.time() > deadline - 4.0:
                    return blks
                red = True
                for ts in itertools.combinations(blks[i], t):
                    if cc[ts_index[ts]] == 1:
                        red = False
                        break
                if red:
                    rem = blks.pop(i)
                    for ts in itertools.combinations(rem, t):
                        cc[ts_index[ts]] -= 1
                    changed = True
                    break
        return blks

    blocks = remove_redundant(blocks)
    best_len = len(blocks)

    # Local search: single swap
    def swap_search(blks, passes=40):
        if not blks:
            return blks, False
        cc = np.zeros(num_t, dtype=np.int32)
        for b in blks:
            for ts in itertools.combinations(b, t):
                cc[ts_index[ts]] += 1
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
                ut = []
                for ts in itertools.combinations(blks[i], t):
                    idx = ts_index[ts]
                    if cc[idx] == 1:
                        ut.append(idx)
                if not ut:
                    continue
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
                for ts in itertools.combinations(old, t):
                    cc[ts_index[ts]] -= 1
                for ts in itertools.combinations(nb, t):
                    cc[ts_index[ts]] += 1
                sol_set.discard(tuple(old))
                sol_set.add(tuple(nb))
                blks[i] = nb
                improved = True
                found = True
                break
            if not found:
                break
        return blks, improved

    # 2-exchange
    def two_exchange(blks):
        if len(blks) < 2:
            return blks, False
        cc = np.zeros(num_t, dtype=np.int32)
        for b in blks:
            for ts in itertools.combinations(b, t):
                cc[ts_index[ts]] += 1
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
                uti = []
                for ts in itertools.combinations(blks[i], t):
                    idx = ts_index[ts]
                    if cc[idx] == 1:
                        uti.append(idx)
                utj = []
                for ts in itertools.combinations(blks[j], t):
                    idx = ts_index[ts]
                    if cc[idx] == 1:
                        utj.append(idx)
                au = uti + utj
                if not au:
                    continue
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
                        for ts in itertools.combinations(oi, t):
                            cc[ts_index[ts]] -= 1
                        for ts in itertools.combinations(oj, t):
                            cc[ts_index[ts]] -= 1
                        for ts in itertools.combinations(nb, t):
                            cc[ts_index[ts]] += 1
                        sol_set.discard(tuple(oi))
                        sol_set.discard(tuple(oj))
                        sol_set.add(tuple(nb))
                        blks.pop(j)
                        blks[i] = nb
                        improved = True
                        break
                if uti:
                    ci = set(ts_to_block_idx[uti[0]])
                    for ti in uti[1:]:
                        ci.intersection_update(ts_to_block_idx[ti])
                        if not ci:
                            break
                else:
                    ci = set(range(num_k))
                if utj:
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
                    for ts in itertools.combinations(oi, t):
                        cc[ts_index[ts]] -= 1
                    for ts in itertools.combinations(oj, t):
                        cc[ts_index[ts]] -= 1
                    for ts in itertools.combinations(nb1, t):
                        cc[ts_index[ts]] += 1
                    for ts in itertools.combinations(nb2, t):
                        cc[ts_index[ts]] += 1
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

    # Main loop
    for _ in range(25):
        if time.time() > deadline - 6.0:
            break
        blocks, imp = swap_search(blocks, 35)
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
        for _ in range(15):
            if time.time() > deadline - 2.0:
                break
            blocks, imp = swap_search(blocks, 20)
            if imp:
                blocks = remove_redundant(blocks)
                if len(blocks) < best_len:
                    best_len = len(blocks)
            else:
                break

    return blocks
