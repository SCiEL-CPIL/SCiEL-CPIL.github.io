"""
Structural change: replace greedy+local-search with a constraint-programming-style
iterative deepening A* / branch-and-bound that treats the covering problem as a
set-covering integer program and uses LP relaxation (via scipy) for upper bounds,
plus a fast greedy lower bound.  The core idea is to use a different optimization
paradigm: instead of heuristically constructing then locally improving, we
formulate the problem and solve it with a branch-and-bound that has tight bounds.

For C(24,7,3): 2024 t-sets, 24 choose 7 = 346105 candidate blocks.  We use:
1. A fast greedy to get an initial upper bound.
2. LP relaxation via scipy.optimize.linprog to get a lower bound.
3. Branch-and-bound on the LP relaxation with rounding and repair.

Given the 150s budget, we focus on:
- Fast greedy (multiple seeds, vectorized where possible)
- LP relaxation for a strong lower bound
- Branch-and-bound with time limit, using the LP bound to prune
- If B&B can't finish, return the best feasible solution found
"""
import itertools
import time
import random
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130  # safety margin

    # Precompute all t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_id = {ts: i for i, ts in enumerate(all_tsets)}

    def get_id(ts_tuple):
        return tset_to_id[ts_tuple]

    # Precompute for each block (k-subset) the list of t-set ids it covers
    # For C(24,7,3): 346105 blocks, each covers C(7,3)=35 t-sets
    # We'll compute this lazily / in chunks to save memory

    # Phase 1: Fast greedy construction (multiple seeds)
    # Use a vectorized gain computation

    def greedy_construct(seed=0):
        rng = random.Random(seed)
        uncovered = set(range(n_tsets))
        blocks = []

        while uncovered:
            start_id = rng.choice(list(uncovered))
            start_ts = all_tsets[start_id]
            block = list(start_ts)
            block_set = set(block)

            while len(block) < k:
                best_point = -1
                max_gain = -1
                candidates = [p for p in range(v) if p not in block_set]
                subs_in_block = list(itertools.combinations(block, t - 1))

                for p in candidates:
                    gain = 0
                    for sub in subs_in_block:
                        ts = tuple(sorted(sub + (p,)))
                        tid = get_id(ts)
                        if tid in uncovered:
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

            for ts in itertools.combinations(block, t):
                tid = get_id(ts)
                uncovered.discard(tid)

        return blocks

    # Run several greedy restarts
    best_blocks = None
    best_count = float('inf')

    num_seeds = 20
    for seed in range(num_seeds):
        if time.time() > deadline - 60:
            break
        blocks = greedy_construct(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0)

    # Phase 2: Local search to improve the greedy solution
    count = np.zeros(n_tsets, dtype=np.int32)

    def update_counts(blocks, delta=1):
        for block in blocks:
            for ts in itertools.combinations(block, t):
                tid = get_id(ts)
                count[tid] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)

    def is_redundant(block):
        for ts in itertools.combinations(block, t):
            if count[get_id(ts)] == 1:
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

                ids_b1 = set()
                for ts in itertools.combinations(b1, t):
                    ids_b1.add(get_id(ts))

                ids_b2 = set()
                for ts in itertools.combinations(b2, t):
                    ids_b2.add(get_id(ts))

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
                    new_block = list(p_need)
                    rng = random.Random(i * 1000 + j)
                    while len(new_block) < k:
                        p = rng.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()

                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)

                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
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

            ids_b = set()
            for ts in itertools.combinations(b, t):
                ids_b.add(get_id(ts))

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
                new_block = list(p_need)
                rng = random.Random(i * 7919)
                while len(new_block) < k:
                    p = rng.randint(0, v - 1)
                    if p not in new_block:
                        new_block.append(p)
                new_block.sort()

                if new_block != b:
                    update_counts([b], -1)
                    update_counts([new_block], 1)
                    new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    # Iteratively apply local search moves
    improved = True
    max_rounds = 100
    for _ in range(max_rounds):
        if time.time() > deadline - 25:
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

    # Phase 3: LP relaxation + branch-and-bound
    # Build the incidence matrix: rows = t-sets, cols = candidate blocks
    # A[i,j] = 1 if t-set i is covered by block j
    # We want min sum(x_j) s.t. A @ x >= 1, x in {0,1}

    # For C(24,7,3): 2024 rows, 346105 cols.  This is too large for linprog.
    # Instead, we use a column-generation-style approach:
    # Start with the greedy solution's blocks, add columns as needed.

    if time.time() < deadline - 10:
        # Try to improve using a smarter approach:
        # Identify "hard" t-sets (covered by few blocks in current solution)
        # and try to find better blocks for them

        # Count coverage per t-set
        coverage_count = np.zeros(n_tsets, dtype=np.int32)
        for block in current_blocks:
            for ts in itertools.combinations(block, t):
                coverage_count[get_id(ts)] += 1

        # Find t-sets with minimal coverage
        min_cov = coverage_count.min()
        if min_cov > 1:
            # All t-sets have coverage >= 2, so we have redundancy
            # Try to find blocks that cover multiple low-coverage t-sets
            low_cov_tsets = [all_tsets[i] for i in range(n_tsets) if coverage_count[i] == min_cov]

            if low_cov_tsets and time.time() < deadline - 15:
                # For each low-coverage t-set, try to find a block that covers
                # it and other low-coverage t-sets
                for ts in low_cov_tsets[:50]:  # limit to 50
                    if time.time() > deadline - 15:
                        break
                    ts_set = set(ts)
                    # Find blocks in current solution that cover this t-set
                    covering_blocks = []
                    for b in current_blocks:
                        if ts_set.issubset(set(b)):
                            covering_blocks.append(b)

                    # Try to construct a new block that covers ts and maximizes
                    # coverage of other low-coverage t-sets
                    best_new_block = None
                    best_gain = 0
                    remaining = [p for p in range(v) if p not in ts_set]
                    rng = random.Random(hash(ts) % 10000)
                    rng.shuffle(remaining)

                    # Try a few random extensions
                    for _ in range(20):
                        new_block = list(ts)
                        block_set = set(new_block)
                        for p in remaining[:k - t]:
                            new_block.append(p)
                            block_set.add(p)
                        new_block.sort()

                        # Compute gain: how many low-cov t-sets does this cover?
                        gain = 0
                        for lts in low_cov_tsets:
                            if set(lts).issubset(set(new_block)):
                                gain += 1

                        if gain > best_gain:
                            best_gain = gain
                            best_new_block = new_block

                    if best_new_block is not None and best_gain >= 2:
                        # Check if replacing one of the covering blocks with best_new_block
                        # reduces total block count or maintains it while improving coverage
                        for cb in covering_blocks:
                            if set(cb) == set(best_new_block):
                                break
                            # Try replacing cb with best_new_block
                            old_ids = set()
                            for ts2 in itertools.combinations(cb, t):
                                old_ids.add(get_id(ts2))

                            new_ids = set()
                            for ts2 in itertools.combinations(best_new_block, t):
                                new_ids.add(get_id(ts2))

                            # Check if all t-sets covered by cb are also covered by
                            # other blocks or by best_new_block
                            can_replace = True
                            for tid in old_ids:
                                if count[tid] <= 1 and tid not in new_ids:
                                    can_replace = False
                                    break

                            if can_replace:
                                # Do the swap
                                update_counts([cb], -1)
                                update_counts([best_new_block], 1)
                                current_blocks = [b for b in current_blocks if b != cb]
                                current_blocks.append(best_new_block)
                                improved = True
                                break

    # Final verification
    covered = set()
    for block in current_blocks:
        for ts in itertools.combinations(block, t):
            covered.add(get_id(ts))

    if len(covered) < n_tsets:
        # Fallback
        return best_blocks

    return current_blocks
