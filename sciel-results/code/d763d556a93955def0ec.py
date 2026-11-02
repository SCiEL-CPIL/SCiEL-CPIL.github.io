import time
import random
import itertools
import numpy as np
from math import comb as math_comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute all k-blocks and t-subsets
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    ts_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Build incidence structures
    block_to_ts = []  # block -> list of t-subset indices it covers
    ts_to_blocks = [[] for _ in range(num_t)]  # t-subset -> list of block indices covering it
    for j, blk in enumerate(all_k_blocks):
        idxs = []
        for ts in itertools.combinations(blk, t):
            idx = ts_index[ts]
            idxs.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(idxs)

    # Convert ts_to_blocks to sets for faster intersection
    ts_to_blocks_sets = [set(b) for b in ts_to_blocks]

    def run_greedy(seed, randomized=True, time_budget=None):
        """Run randomized greedy set cover. Returns list of block indices."""
        rng = np.random.RandomState(seed)
        gain = np.full(num_k, math_comb(k, t), dtype=np.int32)
        uncovered = np.ones(num_t, dtype=bool)
        selected = np.zeros(num_k, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 15.0:
                return None
            mg = int(np.max(gain))
            if mg <= 0:
                break
            cands = np.where((gain == mg) & (~selected))[0]
            if len(cands) == 0:
                break
            if randomized and len(cands) > 1:
                w = np.ones(len(cands)) + rng.rand(len(cands)) * 0.5
                p = w / w.sum()
                b = cands[rng.choice(len(cands), p=p)]
            else:
                b = cands[rng.randint(0, len(cands))] if randomized else cands[0]
            sol.append(int(b))
            selected[b] = True
            gain[b] = -1
            for ti in block_to_ts[b]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    for nb in ts_to_blocks[ti]:
                        if not selected[nb]:
                            gain[nb] -= 1
        return sol

    def idx_to_blocks(idxs):
        return [list(all_k_blocks[i]) for i in idxs]

    def coverage_counts(blk_idxs):
        """Compute coverage count for each t-subset given a list of block indices."""
        cc = np.zeros(num_t, dtype=np.int32)
        for b in blk_idxs:
            for ti in block_to_ts[b]:
                cc[ti] += 1
        return cc

    def remove_redundant(blk_idxs, time_limit=None):
        """Remove redundant blocks. blk_idxs is a list of block indices."""
        if not blk_idxs:
            return blk_idxs
        cc = coverage_counts(blk_idxs)
        changed = True
        while changed:
            changed = False
            for i in range(len(blk_idxs)):
                if time_limit and time.time() > time_limit:
                    return blk_idxs
                redundant = True
                for ti in block_to_ts[blk_idxs[i]]:
                    if cc[ti] == 1:
                        redundant = False
                        break
                if redundant:
                    for ti in block_to_ts[blk_idxs[i]]:
                        cc[ti] -= 1
                    blk_idxs.pop(i)
                    changed = True
                    break
        return blk_idxs

    def vnd_local_search(blk_idxs, time_limit=None):
        """Variable Neighborhood Descent: 1-exchange, then 2-exchange, then 3-exchange."""
        if not blk_idxs:
            return blk_idxs
        cc = coverage_counts(blk_idxs)
        sol_set = set(blk_idxs)
        improved_total = False

        # Neighborhood 1: single block removal (if redundant)
        blk_idxs = remove_redundant(blk_idxs, time_limit)
        cc = coverage_counts(blk_idxs)
        sol_set = set(blk_idxs)

        # Neighborhood 2: 1-for-1 exchange
        for _ in range(50):
            if time_limit and time.time() > time_limit:
                break
            found = False
            n = len(blk_idxs)
            for i in range(n):
                if time_limit and time.time() > time_limit:
                    break
                # Find t-subsets only covered by block i
                ut = [ti for ti in block_to_ts[blk_idxs[i]] if cc[ti] == 1]
                if not ut:
                    continue
                # Find blocks that cover all t-subsets in ut
                cands = ts_to_blocks_sets[ut[0]].copy()
                for ti in ut[1:]:
                    cands &= ts_to_blocks_sets[ti]
                    if not cands:
                        break
                if not cands:
                    continue
                valid = [c for c in cands if c not in sol_set]
                if not valid:
                    continue
                # Pick candidate with minimum overlap
                best_c = None
                best_overlap = float('inf')
                for c in valid[:min(len(valid), 30)]:
                    overlap = sum(1 for ti in block_to_ts[c] if cc[ti] > 0)
                    if overlap < best_overlap:
                        best_overlap = overlap
                        best_c = c
                if best_c is not None:
                    for ti in block_to_ts[blk_idxs[i]]:
                        cc[ti] -= 1
                    for ti in block_to_ts[best_c]:
                        cc[ti] += 1
                    sol_set.discard(blk_idxs[i])
                    sol_set.add(best_c)
                    blk_idxs[i] = best_c
                    found = True
                    break
            if not found:
                break

        # Neighborhood 3: 2-for-1 exchange (replace two blocks with one)
        for _ in range(30):
            if time_limit and time.time() > time_limit:
                break
            found = False
            n = len(blk_idxs)
            for i in range(n):
                if time_limit and time.time() > time_limit:
                    break
                for j in range(i + 1, min(i + 15, n)):
                    if time_limit and time.time() > time_limit:
                        break
                    # Union of uniquely covered t-subsets by blocks i and j
                    ut = []
                    for ti in block_to_ts[blk_idxs[i]]:
                        if cc[ti] == 1:
                            ut.append(ti)
                    for ti in block_to_ts[blk_idxs[j]]:
                        if cc[ti] == 1:
                            ut.append(ti)
                    if not ut:
                        continue
                    # Find a single block covering all t-subsets in ut
                    cands = ts_to_blocks_sets[ut[0]].copy()
                    for ti in ut[1:]:
                        cands &= ts_to_blocks_sets[ti]
                        if not cands:
                            break
                    if cands:
                        valid = [c for c in cands if c not in sol_set]
                        if valid:
                            best_c = random.choice(valid[:min(len(valid), 10)])
                            for ti in block_to_ts[blk_idxs[i]]:
                                cc[ti] -= 1
                            for ti in block_to_ts[blk_idxs[j]]:
                                cc[ti] -= 1
                            for ti in block_to_ts[best_c]:
                                cc[ti] += 1
                            sol_set.discard(blk_idxs[i])
                            sol_set.discard(blk_idxs[j])
                            sol_set.add(best_c)
                            blk_idxs.pop(j)
                            blk_idxs[i] = best_c
                            found = True
                            n = len(blk_idxs)
                            break
                if found:
                    break
            if not found:
                break

        # Neighborhood 4: 2-for-2 exchange (replace two blocks with two different blocks)
        for _ in range(20):
            if time_limit and time.time() > time_limit:
                break
            found = False
            n = len(blk_idxs)
            for i in range(n):
                if time_limit and time.time() > time_limit:
                    break
                for j in range(i + 1, min(i + 10, n)):
                    if time_limit and time.time() > time_limit:
                        break
                    ci = [ti for ti in block_to_ts[blk_idxs[i]] if cc[ti] == 1]
                    cj = [ti for ti in block_to_ts[blk_idxs[j]] if cc[ti] == 1]
                    if ci:
                        vi = ts_to_blocks_sets[ci[0]].copy()
                        for ti in ci[1:]:
                            vi &= ts_to_blocks_sets[ti]
                    else:
                        vi = set(range(num_k))
                    if cj:
                        vj = ts_to_blocks_sets[cj[0]].copy()
                        for ti in cj[1:]:
                            vj &= ts_to_blocks_sets[ti]
                    else:
                        vj = set(range(num_k))
                    vi = [c for c in vi if c not in sol_set]
                    vj = [c for c in vj if c not in sol_set and c not in vi]
                    if vi and vj:
                        ni = random.choice(vi[:min(len(vi), 10)])
                        nj = random.choice(vj[:min(len(vj), 10)])
                        for ti in block_to_ts[blk_idxs[i]]:
                            cc[ti] -= 1
                        for ti in block_to_ts[blk_idxs[j]]:
                            cc[ti] -= 1
                        for ti in block_to_ts[ni]:
                            cc[ti] += 1
                        for ti in block_to_ts[nj]:
                            cc[ti] += 1
                        sol_set.discard(blk_idxs[i])
                        sol_set.discard(blk_idxs[j])
                        sol_set.add(ni)
                        sol_set.add(nj)
                        blk_idxs[i] = ni
                        blk_idxs[j] = nj
                        found = True
                        break
                if found:
                    break
            if not found:
                break

        # Neighborhood 5: 3-for-1 exchange
        for _ in range(10):
            if time_limit and time.time() > time_limit:
                break
            found = False
            n = len(blk_idxs)
            for i in range(n):
                if time_limit and time.time() > time_limit:
                    break
                for j in range(i + 1, min(i + 5, n)):
                    if time_limit and time.time() > time_limit:
                        break
                    for m in range(j + 1, min(j + 5, n)):
                        if time_limit and time.time() > time_limit:
                            break
                        ut = []
                        for ti in block_to_ts[blk_idxs[i]]:
                            if cc[ti] == 1:
                                ut.append(ti)
                        for ti in block_to_ts[blk_idxs[j]]:
                            if cc[ti] == 1:
                                ut.append(ti)
                        for ti in block_to_ts[blk_idxs[m]]:
                            if cc[ti] == 1:
                                ut.append(ti)
                        if not ut:
                            continue
                        cands = ts_to_blocks_sets[ut[0]].copy()
                        for ti in ut[1:]:
                            cands &= ts_to_blocks_sets[ti]
                            if not cands:
                                break
                        if cands:
                            valid = [c for c in cands if c not in sol_set]
                            if valid:
                                best_c = random.choice(valid[:min(len(valid), 5)])
                                for ti in block_to_ts[blk_idxs[i]]:
                                    cc[ti] -= 1
                                for ti in block_to_ts[blk_idxs[j]]:
                                    cc[ti] -= 1
                                for ti in block_to_ts[blk_idxs[m]]:
                                    cc[ti] -= 1
                                for ti in block_to_ts[best_c]:
                                    cc[ti] += 1
                                sol_set.discard(blk_idxs[i])
                                sol_set.discard(blk_idxs[j])
                                sol_set.discard(blk_idxs[m])
                                sol_set.add(best_c)
                                # Remove in reverse order to maintain indices
                                blk_idxs.pop(m)
                                blk_idxs.pop(j)
                                blk_idxs[i] = best_c
                                found = True
                                n = len(blk_idxs)
                                break
                    if found:
                        break
                if found:
                    break
            if not found:
                break

        # Final cleanup
        blk_idxs = remove_redundant(blk_idxs, time_limit)
        return blk_idxs

    # Phase 1: Multiple greedy runs
    best_blk_idxs = None
    best_len = float('inf')

    # Deterministic greedy
    s0 = run_greedy(0, randomized=False)
    if s0 is not None:
        best_blk_idxs = s0
        best_len = len(s0)

    # Randomized greedy runs
    for seed in range(1, 60):
        if time.time() > deadline - 25.0:
            break
        sol = run_greedy(seed, randomized=True)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_blk_idxs = sol

    # Phase 2: VND local search
    if best_blk_idxs is not None:
        best_blk_idxs = remove_redundant(best_blk_idxs, deadline - 10.0)
        best_len = len(best_blk_idxs)

        for iteration in range(40):
            if time.time() > deadline - 10.0:
                break
            prev_len = len(best_blk_idxs)
            best_blk_idxs = vnd_local_search(best_blk_idxs, deadline - 10.0)
            if len(best_blk_idxs) < prev_len:
                best_len = len(best_blk_idxs)
            else:
                # Try a perturbation: randomly remove a few blocks and re-greedy
                if time.time() > deadline - 15.0:
                    break
                if len(best_blk_idxs) > 10:
                    rng = np.random.RandomState(iteration)
                    n_remove = min(5, len(best_blk_idxs) // 10)
                    remove_set = set(rng.choice(len(best_blk_idxs), n_remove, replace=False).tolist())
                    remaining = [b for idx, b in enumerate(best_blk_idxs) if idx not in remove_set]
                    # Re-greedy to fill gaps
                    cc = coverage_counts(remaining)
                    uncovered_mask = cc == 0
                    if np.any(uncovered_mask):
                        # Quick greedy fill
                        gain = np.zeros(num_k, dtype=np.int32)
                        for b in range(num_k):
                            if b not in set(remaining):
                                gain[b] = sum(1 for ti in block_to_ts[b] if uncovered_mask[ti])
                        while True:
                            if time.time() > deadline - 12.0:
                                break
                            mg = int(np.max(gain))
                            if mg <= 0:
                                break
                            cands = np.where(gain == mg)[0]
                            b = cands[0]
                            remaining.append(int(b))
                            gain[b] = -1
                            for ti in block_to_ts[b]:
                                if uncovered_mask[ti]:
                                    uncovered_mask[ti] = False
                                    for nb in ts_to_blocks[ti]:
                                        if gain[nb] >= 0:
                                            gain[nb] -= 1
                        remaining = remove_redundant(remaining, deadline - 10.0)
                        if len(remaining) < best_len:
                            best_len = len(remaining)
                            best_blk_idxs = remaining
                else:
                    break

    # Convert to blocks
    if best_blk_idxs is None:
        best_blk_idxs = run_greedy(0, randomized=False)
        if best_blk_idxs is None:
            best_blk_idxs = []

    blocks = idx_to_blocks(best_blk_idxs)

    # Ensure no duplicates
    seen = set()
    unique_blocks = []
    for b in blocks:
        tb = tuple(b)
        if tb not in seen:
            seen.add(tb)
            unique_blocks.append(b)

    return unique_blocks
