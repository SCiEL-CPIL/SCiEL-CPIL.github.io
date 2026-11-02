import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110  # safety margin vs 150s kill

    # Precompute all t-subsets and their indices
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    # All k-blocks and their t-subset index arrays (vectorized gain possible)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_idx_arrays = np.empty((n_blocks, C(v, t)), dtype=np.int64)
    for bi, blk in enumerate(all_blocks):
        block_idx_arrays[bi] = [subset_to_idx[sub] for sub in itertools.combinations(blk, t)]

    # Per-point block incidence for load tracking
    point_blocks = [[] for _ in range(v)]
    for bi, blk in enumerate(all_blocks):
        for x in blk:
            point_blocks[x].append(bi)

    best_blocks = None
    best_count = float('inf')

    def greedy_unbiased(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_subsets, dtype=bool)
        num_uncovered = n_subsets
        chosen = []
        chosen_set = set()
        load = np.zeros(v, dtype=np.float64)

        while num_uncovered > 0:
            if time.time() > deadline:
                return None
            # pick a random uncovered subset as seed
            seed_idx = -1
            for _ in range(80):
                cand = rng.randint(0, n_subsets - 1)
                if uncovered[cand]:
                    seed_idx = cand
                    break
            if seed_idx == -1:
                for i in range(n_subsets):
                    if uncovered[i]:
                        seed_idx = i
                        break
            if seed_idx == -1:
                break
            seed_sub = all_subsets[seed_idx]

            # candidate blocks must contain the seed subset
            cand_blocks = [bi for bi in point_blocks[seed_sub[0]]
                           if all(x in all_blocks[bi] for x in seed_sub)]
            if not cand_blocks:
                cand_blocks = list(range(n_blocks))

            # score each candidate: gain * (1 / (1+load))  -> load-balancing
            best_bi = None
            best_score = -1.0
            # subsample if too many
            if len(cand_blocks) > 2000:
                cand_blocks = rng.sample(cand_blocks, 2000)
            for bi in cand_blocks:
                idxs = block_idx_arrays[bi]
                gain = int(np.count_nonzero(uncovered[idxs]))
                if gain <= 0:
                    continue
                # load factor: product of (1/(1+load[x])) over x in block
                blk = all_blocks[bi]
                lf = 1.0
                for x in blk:
                    lf /= (1.0 + load[x])
                score = gain * lf
                if score > best_score:
                    best_score = score
                    best_bi = bi
            if best_bi is None:
                # fallback: pick block with max gain
                best_bi = max(cand_blocks, key=lambda bi: int(np.count_nonzero(uncovered[block_idx_arrays[bi]])))
            blk = all_blocks[best_bi]
            chosen.append(blk)
            chosen_set.add(best_bi)
            for x in blk:
                load[x] += 1.0
            idxs = block_idx_arrays[best_bi]
            newly = uncovered[idxs]
            num_new = int(np.count_nonzero(newly))
            num_uncovered -= num_new
            uncovered[idxs] = False

        return [list(blk) for blk in chosen]

    def redundancy_removal(blocks):
        blocks = [list(b) for b in blocks]
        changed = True
        while changed and time.time() < deadline - 8:
            changed = False
            # coverage count
            cov = np.zeros(n_subsets, dtype=np.int64)
            for blk in blocks:
                for sub in itertools.combinations(blk, t):
                    cov[subset_to_idx[sub]] += 1
            for i in range(len(blocks)):
                if time.time() > deadline - 8:
                    break
                blk = blocks[i]
                redundant = True
                for sub in itertools.combinations(blk, t):
                    if cov[subset_to_idx[sub]] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    def swap_improve(blocks):
        blocks = [list(b) for b in blocks]
        attempts = 0
        while time.time() < deadline - 8 and attempts < 400:
            attempts += 1
            i = random.randint(0, len(blocks) - 1)
            old = blocks[i]
            # coverage without block i
            cov = np.zeros(n_subsets, dtype=np.int64)
            for j, blk in enumerate(blocks):
                if j == i:
                    continue
                for sub in itertools.combinations(blk, t):
                    cov[subset_to_idx[sub]] += 1
            # unique subsets of old
            unique = []
            for sub in itertools.combinations(old, t):
                if cov[subset_to_idx[sub]] == 0:
                    unique.append(sub)
            if not unique:
                blocks.pop(i)
                continue
            # candidate new block: must cover all unique subsets
            cand_pts = set()
            for sub in unique:
                cand_pts.update(sub)
            if len(cand_pts) > k:
                continue
            # try to find a block covering unique with low overlap
            best_bi = None
            best_overlap = float('inf')
            for bi, blk in enumerate(all_blocks):
                if all(x in blk for x in cand_pts):
                    # overlap = number of t-subsets of blk already covered (cov>0)
                    idxs = block_idx_arrays[bi]
                    overlap = int(np.count_nonzero(cov[idxs] > 0))
                    if overlap < best_overlap:
                        best_overlap = overlap
                        best_bi = bi
            if best_bi is None:
                continue
            # accept if new block's total new coverage >= old's unique count and overlap lower
            new_idx = block_idx_arrays[best_bi]
            new_new = int(np.count_nonzero(cov[new_idx] == 0))
            if new_new >= len(unique) and best_overlap < len(unique):
                blocks[i] = list(all_blocks[best_bi])
        return blocks

    # Main loop: multiple restarts
    restart = 0
    while time.time() < deadline - 25:
        blocks = greedy_unbiased(restart)
        if blocks is None:
            continue
        if len(blocks) < best_count:
            improved = redundancy_removal(blocks)
            improved = swap_improve(improved)
            if len(improved) < best_count:
                best_count = len(improved)
                best_blocks = improved
        restart += 1

    if best_blocks is None:
        if time.time() < deadline - 5:
            blocks = greedy_unbiased(12345)
            if blocks:
                return blocks
        return []

    return best_blocks
