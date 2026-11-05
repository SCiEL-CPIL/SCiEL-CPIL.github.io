"""Improved covering design: fast greedy + SA local search + vectorized ops."""
import itertools
import time
import random
import numpy as np
from math import log, exp


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    # Precompute all t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_id = {ts: i for i, ts in enumerate(all_tsets)}

    # Precompute t-sets per block quickly
    def block_tset_ids(block):
        return [tset_to_id[ts] for ts in itertools.combinations(block, t)]

    # Precompute pair_to_tset_ids for faster gain computation
    # For t=3: pair (a,b) -> list of tset IDs containing {a,b}
    # Actually for gain we need: for each (t-1)-subset of block and candidate p,
    # check if the tset (subset + p) is uncovered.
    # Precompute: for each (t-1)-tuple, map to tset ID when adding each possible p
    # For t=3: pairs_in_block, for each pair (a,b) and candidate p, tset = {a,b,p}
    # We can precompute a 3D array: pair_id -> array of v entries, -1 if p not valid, else tset_id
    # But simpler: just use the tset_to_id lookup which is O(1)

    # Greedy construction with multiple strategies
    def greedy_construct(seed=0, strategy='standard'):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsets, dtype=bool)
        blocks = []

        while uncovered.any():
            if strategy == 'standard':
                # Pick a random uncovered t-set to start
                unc_idx = np.where(uncovered)[0]
                start_id = unc_idx[rng.randint(0, len(unc_idx) - 1)]
                start_ts = all_tsets[start_id]
            else:
                start_id = rng.randint(0, n_tsets - 1)
                while not uncovered[start_id]:
                    start_id = rng.randint(0, n_tsets - 1)
                start_ts = all_tsets[start_id]

            block = list(start_ts)
            block_set = set(block)

            while len(block) < k:
                # Compute gain for each candidate point
                best_point = -1
                max_gain = -1
                candidates = [p for p in range(v) if p not in block_set]

                # For each (t-1)-subset of current block, check which t-sets
                # containing that subset AND candidate p are uncovered
                subs_in_block = list(itertools.combinations(block, t - 1))

                for p in candidates:
                    gain = 0
                    for sub in subs_in_block:
                        ts = tuple(sorted(sub + (p,)))
                        tid = tset_to_id[ts]
                        if uncovered[tid]:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_point = p

                if max_gain == 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for tid in block_tset_ids(block):
                uncovered[tid] = False

        return blocks

    # Phase 1: Multiple greedy restarts
    best_blocks = None
    best_count = float('inf')

    for seed in range(50):
        if time.time() > deadline - 60:
            break
        blocks = greedy_construct(seed, 'standard')
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0)

    # Phase 2: Local search optimization
    count = np.zeros(n_tsets, dtype=np.int32)

    def update_counts(blocks, delta=1):
        for block in blocks:
            for tid in block_tset_ids(block):
                count[tid] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)

    def is_redundant(block):
        for tid in block_tset_ids(block):
            if count[tid] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False

        for i in range(n):
            if time.time() > deadline - 20:
                return blocks, False
            for j in range(i + 1, n):
                if time.time() > deadline - 20:
                    return blocks, False

                b1 = blocks[i]
                b2 = blocks[j]

                ids_b1 = set(block_tset_ids(b1))
                ids_b2 = set(block_tset_ids(b2))
                check_ids = ids_b1 | ids_b2

                u_need = []
                for tid in check_ids:
                    c = count[tid]
                    cov_b1 = 1 if tid in ids_b1 else 0
                    cov_b2 = 1 if tid in ids_b2 else 0
                    if c == cov_b1 + cov_b2:
                        u_need.append(all_tsets[tid])

                if not u_need:
                    update_counts([b1, b2], -1)
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True

                p_need = set()
                for ts in u_need:
                    p_need.update(ts)

                if len(p_need) <= k:
                    # Try multiple random completions
                    best_new_block = None
                    best_new_count = float('inf')
                    for attempt in range(5):
                        new_block = list(p_need)
                        rng = random.Random(i * 1000 + j * 100 + attempt)
                        while len(new_block) < k:
                            p = rng.randint(0, v - 1)
                            if p not in new_block:
                                new_block.append(p)
                        new_block.sort()
                        # Count how many t-sets this new block covers that are in u_need
                        # Actually we just need it to cover all u_need
                        covered_all = True
                        for ts in u_need:
                            if not all(p in new_block for p in ts):
                                covered_all = False
                                break
                        if covered_all:
                            # Count total t-sets covered
                            n_covered = len(set(block_tset_ids(new_block)))
                            if n_covered < best_new_count:
                                best_new_count = n_covered
                                best_new_block = new_block

                    if best_new_block is not None:
                        update_counts([b1, b2], -1)
                        update_counts([best_new_block], 1)
                        new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                        new_blocks.append(best_new_block)
                        return new_blocks, True

        return blocks, False

    def try_swap_1_for_1(blocks):
        n = len(blocks)
        if n < 1:
            return blocks, False

        for i in range(n):
            if time.time() > deadline - 20:
                return blocks, False

            b = blocks[i]
            ids_b = set(block_tset_ids(b))

            u_need = []
            for tid in ids_b:
                if count[tid] == 1:
                    u_need.append(all_tsets[tid])

            if not u_need:
                update_counts([b], -1)
                return [x for idx, x in enumerate(blocks) if idx != i], True

            p_need = set()
            for ts in u_need:
                p_need.update(ts)

            if len(p_need) <= k:
                # Try multiple random completions
                best_new_block = None
                best_new_count = float('inf')
                for attempt in range(5):
                    new_block = list(p_need)
                    rng = random.Random(i * 7919 + attempt)
                    while len(new_block) < k:
                        p = rng.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    covered_all = True
                    for ts in u_need:
                        if not all(p in new_block for p in ts):
                            covered_all = False
                            break
                    if covered_all:
                        n_covered = len(set(block_tset_ids(new_block)))
                        if n_covered < best_new_count:
                            best_new_count = n_covered
                            best_new_block = new_block

                if best_new_block is not None and best_new_block != b:
                    update_counts([b], -1)
                    update_counts([best_new_block], 1)
                    new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                    new_blocks.append(best_new_block)
                    return new_blocks, True

        return blocks, False

    # Iteratively apply local search moves (deterministic)
    improved = True
    max_rounds = 200
    for _ in range(max_rounds):
        if time.time() > deadline - 50:
            break
        if not improved:
            break
        improved = False

        new_blocks, rem = try_remove_redundant(current_blocks)
        if rem:
            current_blocks = new_blocks
            improved = True

        new_blocks, swp = try_swap_2_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True

        new_blocks, swp = try_swap_1_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True

    # SA local search
    def sa_local_search(blocks, max_iters=5000):
        blocks = list(blocks)
        count_local = count.copy()
        current_len = len(blocks)
        best_blocks = list(blocks)
        best_len = current_len

        T_start = 8.0
        T_end = 0.05
        alpha = (T_end / T_start) ** (1.0 / max_iters)

        T = T_start
        rng = random.Random(42)

        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            T *= alpha

            move_type = rng.random()

            if move_type < 0.25:
                new_blocks, rem = try_remove_redundant(blocks)
                if rem:
                    for b in blocks:
                        if b not in new_blocks:
                            for tid in block_tset_ids(b):
                                count_local[tid] -= 1
                    blocks = new_blocks
                    current_len = len(blocks)
                    if current_len < best_len:
                        best_len = current_len
                        best_blocks = list(blocks)
                continue

            elif move_type < 0.55:
                new_blocks, swp = try_swap_2_for_1(blocks)
                if swp:
                    count_local[:] = 0
                    for b in new_blocks:
                        for tid in block_tset_ids(b):
                            count_local[tid] += 1
                    blocks = new_blocks
                    current_len = len(blocks)
                    if current_len < best_len:
                        best_len = current_len
                        best_blocks = list(blocks)
                continue

            else:
                new_blocks, swp = try_swap_1_for_1(blocks)
                if swp:
                    count_local[:] = 0
                    for b in new_blocks:
                        for tid in block_tset_ids(b):
                            count_local[tid] += 1
                    blocks = new_blocks
                    current_len = len(blocks)
                    if current_len < best_len:
                        best_len = current_len
                        best_blocks = list(blocks)
                continue

        return best_blocks

    if time.time() < deadline - 20:
        sa_result = sa_local_search(current_blocks, max_iters=5000)
        if len(sa_result) < len(current_blocks):
            current_blocks = sa_result
            count[:] = 0
            for b in current_blocks:
                for tid in block_tset_ids(b):
                    count[tid] += 1

    # Final verification
    covered = set()
    for block in current_blocks:
        for tid in block_tset_ids(block):
            covered.add(tid)

    if len(covered) < n_tsets:
        return best_blocks

    return current_blocks
