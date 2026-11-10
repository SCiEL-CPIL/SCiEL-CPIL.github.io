import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # ---- Precomputation ----
    points = np.arange(v, dtype=np.int32)
    all_blocks = np.array(list(itertools.combinations(range(v), k)), dtype=np.int32)
    n_blocks = len(all_blocks)

    # Map each t-subset to a unique id
    t_sub_list = list(itertools.combinations(range(v), t))
    t_sub_to_id = {sub: i for i, sub in enumerate(t_sub_list)}
    n_t = len(t_sub_list)

    # For each block, the list of t-subset ids it covers (shape: n_blocks x C(k,t))
    ckt = int(np.prod([k - t + i for i in range(t)])) // (np.prod([i + 1 for i in range(t)]))  # C(k,t)
    # Actually compute properly
    from math import comb
    ckt = comb(k, t)
    block_t_ids = np.zeros((n_blocks, ckt), dtype=np.int32)
    for bi, blk in enumerate(all_blocks):
        subs = list(itertools.combinations(blk, t))
        for si, s in enumerate(subs):
            block_t_ids[bi, si] = t_sub_to_id[s]

    # ---- Vectorized greedy with random tie-breaking ----
    def greedy_random(seed):
        rng = random.Random(seed)
        # Random priority for tie-breaking
        priority = np.arange(n_blocks, dtype=np.float64)
        rng.shuffle(priority.tolist())
        priority = np.array(priority, dtype=np.float64)

        uncovered = np.ones(n_t, dtype=np.int32)
        selected = []
        remaining = np.ones(n_blocks, dtype=bool)

        while True:
            if time.time() > deadline:
                return None
            total_uncov = uncovered.sum()
            if total_uncov == 0:
                break

            # gain[b] = number of currently-uncovered t-subsets covered by block b
            # gain = block_t_ids[remaining] mapped through uncovered
            # Vectorized: for each remaining block, sum uncovered over its t-ids
            gains = np.zeros(n_blocks, dtype=np.int32)
            # Only compute for remaining blocks
            rem_idx = np.where(remaining)[0]
            if len(rem_idx) == 0:
                break
            # For each remaining block, sum uncovered[block_t_ids[bi]]
            # This is a gather + sum over axis=1
            ids = block_t_ids[rem_idx]  # shape (n_rem, ckt)
            g = uncovered[ids].sum(axis=1)  # shape (n_rem,)
            gains[rem_idx] = g

            # Add tiny random perturbation for diversity
            gains[rem_idx] += (priority[rem_idx] * 1e-6)

            best_local = int(np.argmax(gains))
            best_global = int(rem_idx[best_local])
            best_gain = int(gains[best_global])

            if best_gain == 0:
                break

            selected.append(best_global)
            remaining[best_global] = False

            # Update uncovered
            tids = block_t_ids[best_global]
            uncovered[tids] = 0  # set to 0 (already covered)

        if uncovered.sum() != 0:
            return None
        return selected

    # ---- Prune redundant blocks (fast) ----
    def prune(blocks):
        if len(blocks) <= 1:
            return blocks
        # Count coverage
        cover_count = np.zeros(n_t, dtype=np.int32)
        for bi in blocks:
            tids = block_t_ids[bi]
            cover_count[tids] += 1
        essential = []
        for bi in blocks:
            tids = block_t_ids[bi]
            if np.any(cover_count[tids] == 1):
                essential.append(bi)
        return essential

    # ---- Vectorized 2-for-1 local search ----
    def local_2for1(blocks, max_iters=200):
        if len(blocks) < 2:
            return None
        block_set = set(blocks)
        n_b = len(blocks)

        # Precompute coverage mask per block as a set of t-ids for speed
        # Actually use numpy: coverage matrix is too big. Use per-block t-id arrays.
        # For each pair, compute unique coverage, then scan all blocks for a replacement.

        for it in range(max_iters):
            if time.time() > deadline:
                return None
            i, j = random.sample(range(n_b), 2)
            b1, b2 = blocks[i], blocks[j]
            t1 = block_t_ids[b1]
            t2 = block_t_ids[b2]
            # union of t1, t2
            union_set = set(t1.tolist()) | set(t2.tolist())
            # coverage by others
            others_cover = np.zeros(n_t, dtype=bool)
            for idx, bi in enumerate(blocks):
                if idx == i or idx == j:
                    continue
                others_cover[block_t_ids[bi]] = True
            unique = np.array([tid for tid in union_set if not others_cover[tid]], dtype=np.int32)
            if len(unique) == 0:
                # one is redundant
                new_blocks = [b for idx, b in enumerate(blocks) if idx != i]
                return new_blocks
            # Find a block b3 covering all of unique
            for b3 in range(n_blocks):
                if b3 == b1 or b3 == b2:
                    continue
                t3 = block_t_ids[b3]
                # check if unique is subset of t3
                # Fast: check all unique in t3
                t3_set = set(t3.tolist())
                if all(u in t3_set for u in unique):
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(b3)
                    return new_blocks
        return None

    # ---- Main search loop: many fast greedy restarts + local search ----
    best_blocks = None
    best_count = float('inf')

    # Phase 1: many greedy restarts (fast, vectorized)
    n_restarts = 200  # will be limited by time
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:  # leave time for local search
            break
        blocks = greedy_random(r)
        if blocks is None:
            continue
        blocks = prune(blocks)
        # Quick local search
        ls_iter = 0
        improved = True
        while improved and time.time() < deadline - 5.0 and ls_iter < 15:
            improved = False
            ls_iter += 1
            nb = local_2for1(blocks, max_iters=30)
            if nb is not None:
                blocks = prune(nb)
                improved = True
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            if best_count <= 335:
                break

    # Phase 2: deeper local search on the best solution
    if best_blocks is not None and time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            nb = local_2for1(best_blocks, max_iters=100)
            if nb is not None:
                best_blocks = prune(nb)
                improved = True

    if best_blocks is None:
        # Fallback: single greedy
        blocks = greedy_random(0)
        if blocks:
            return [list(all_blocks[i]) for i in blocks]
        return []

    return [list(all_blocks[i]) for i in best_blocks]
