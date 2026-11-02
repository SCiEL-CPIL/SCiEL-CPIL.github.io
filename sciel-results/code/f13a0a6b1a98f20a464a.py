"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    _HAS_MILP = True
except Exception:
    _HAS_MILP = False


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute: for each point, which t-subsets contain it
    point_tsubs = [[] for _ in range(v)]
    for i, sub in enumerate(all_t_subsets):
        for pt in sub:
            point_tsubs[pt].append(i)

    # Precompute: for each block (tuples of k points), its t-subset indices
    block_tsubs_cache = {}

    def block_tsubs(block):
        key = block
        if key in block_tsubs_cache:
            return block_tsubs_cache[key]
        idxs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_tsubs_cache[key] = idxs
        return idxs

    best_blocks = None
    best_count = float('inf')

    n_restarts = 40

    for restart in range(n_restarts):
        if not time_ok():
            break

        random.seed(12345 + restart * 7919)

        covered = [False] * n_t
        covered_count = 0
        blocks = []

        while covered_count < n_t and time_ok():
            # Find uncovered t-subsets
            uncovered_indices = [i for i in range(n_t) if not covered[i]]
            if not uncovered_indices:
                break

            # Score uncovered t-subsets by scarcity (sum of point rarities)
            point_uncovered_count = [0] * v
            for idx in uncovered_indices:
                for pt in all_t_subsets[idx]:
                    point_uncovered_count[pt] += 1

            # Pick seed candidates (weighted by scarcity, capped)
            if len(uncovered_indices) <= 50:
                candidates = uncovered_indices
            else:
                weights = []
                for idx in uncovered_indices:
                    w = sum(1.0 / max(1, point_uncovered_count[pt])
                            for pt in all_t_subsets[idx])
                    weights.append(w)
                total_w = sum(weights)
                if total_w == 0:
                    candidates = [random.choice(uncovered_indices)]
                else:
                    n_samples = min(10, len(uncovered_indices))
                    candidates = []
                    for _ in range(n_samples):
                        r = random.random() * total_w
                        cum = 0.0
                        for i, w in enumerate(weights):
                            cum += w
                            if cum >= r:
                                candidates.append(uncovered_indices[i])
                                break
                    if not candidates:
                        candidates = [random.choice(uncovered_indices)]

            best_block = None
            best_gain = -1

            for seed_idx in candidates:
                if not time_ok():
                    break
                seed_subset = all_t_subsets[seed_idx]
                block_set = set(seed_subset)

                # Extend block greedily with dynamic gain
                while len(block_set) < k and time_ok():
                    best_x = None
                    best_x_gain = -1
                    block_tuple = tuple(sorted(block_set))
                    candidates_pts = [x for x in all_points if x not in block_set]
                    random.shuffle(candidates_pts)

                    for x in candidates_pts:
                        gain = 0
                        for rest in itertools.combinations(block_tuple, t - 1):
                            sub = tuple(sorted(rest + (x,)))
                            idx = t_sub_to_idx[sub]
                            if not covered[idx]:
                                gain += 1
                        tie = random.random() * 0.01
                        if gain + tie > best_x_gain:
                            best_x_gain = gain + tie
                            best_x = x

                    if best_x is None:
                        break
                    block_set.add(best_x)

                if len(block_set) < k:
                    remaining = [x for x in all_points if x not in block_set]
                    random.shuffle(remaining)
                    for x in remaining[:k - len(block_set)]:
                        block_set.add(x)

                block = tuple(sorted(block_set))

                # Total gain of this block
                gain = 0
                for idx in block_tsubs(block):
                    if not covered[idx]:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_block = block

            if best_block is None or best_gain <= 0:
                block_set = set(random.sample(all_points, k))
                best_block = tuple(sorted(block_set))
                best_gain = sum(1 for idx in block_tsubs(best_block)
                                if not covered[idx])

            if best_gain <= 0:
                break

            for idx in block_tsubs(best_block):
                if not covered[idx]:
                    covered[idx] = True
                    covered_count += 1
            blocks.append(best_block)

        # Deletion phase: remove redundant blocks
        if time_ok() and covered_count == n_t:
            tsub_block_count = [0] * n_t
            for b in blocks:
                for idx in block_tsubs(b):
                    tsub_block_count[idx] += 1

            changed = True
            while changed and time_ok():
                changed = False
                for i in range(len(blocks)):
                    if not time_ok():
                        break
                    removable = True
                    for idx in block_tsubs(blocks[i]):
                        if tsub_block_count[idx] <= 1:
                            removable = False
                            break
                    if removable:
                        for idx in block_tsubs(blocks[i]):
                            tsub_block_count[idx] -= 1
                        blocks.pop(i)
                        changed = True
                        break

        # Local search: swap a block for one covering its unique t-subsets
        if time_ok() and covered_count == n_t:
            tsub_block_count = [0] * n_t
            for b in blocks:
                for idx in block_tsubs(b):
                    tsub_block_count[idx] += 1

            attempts = 0
            max_attempts = 300
            while time_ok() and attempts < max_attempts:
                attempts += 1
                idx = random.randint(0, len(blocks) - 1)
                old_block = blocks[idx]

                unique = [i for i in block_tsubs(old_block)
                          if tsub_block_count[i] == 1]

                if not unique:
                    # redundant, remove
                    for i in block_tsubs(old_block):
                        tsub_block_count[i] -= 1
                    blocks.pop(idx)
                    continue

                union = set(pt for i in unique for pt in all_t_subsets[i])
                if len(union) > k:
                    continue

                candidate = set(union)
                while len(candidate) < k:
                    best_point = None
                    best_gain = -1
                    for x in range(v):
                        if x in candidate:
                            continue
                    # pick point maximizing newly-covered unique t-subsets
                    best_point = None
                    best_gain = -1
                    for x in range(v):
                        if x in candidate:
                            continue
                        g = 0
                        for i in unique:
                            sub = all_t_subsets[i]
                            if x in sub and all(pt in candidate for pt in sub if pt != x):
                                g += 1
                        if g > best_gain:
                            best_gain = g
                            best_point = x
                    if best_point is None:
                        break
                    candidate.add(best_point)

                if len(candidate) != k:
                    continue

                new_block = tuple(sorted(candidate))
                valid = all(all(pt in new_block for pt in all_t_subsets[i])
                            for i in unique)
                if not valid:
                    continue

                for i in block_tsubs(old_block):
                    tsub_block_count[i] -= 1
                for i in block_tsubs(new_block):
                    tsub_block_count[i] += 1
                blocks[idx] = new_block

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)

        if best_count <= 140:  # good enough, stop early
            break

    # MILP polish: select a minimal subset of candidate blocks covering all t-subsets
    if _HAS_MILP and best_blocks is not None and time_ok():
        best_blocks = _milp_polish(v, k, t, best_blocks, t_sub_to_idx,
                                   all_t_subsets, block_tsubs, deadline)

    return best_blocks


def _milp_polish(v, k, t, blocks, t_sub_to_idx, all_t_subsets, block_tsubs, deadline):
    n_t = len(all_t_subsets)
    n_blocks = len(blocks)
    if n_blocks == 0:
        return blocks

    # Build candidate pool: top blocks by unique coverage + some random
    tsub_block_count = [0] * n_t
    for b in blocks:
        for idx in block_tsubs(b):
            tsub_block_count[idx] += 1

    # Score each block by number of t-subsets it uniquely covers
    scores = []
    for i, b in enumerate(blocks):
        s = 0
        for idx in block_tsubs(b):
            if tsub_block_count[idx] == 1:
                s += 1
        scores.append(s)

    # Keep top by score, plus a random sample, capped
    order = sorted(range(n_blocks), key=lambda i: -scores[i])
    top = order[:min(n_blocks, 200)]
    extra = random.sample(range(n_blocks), min(n_blocks, 40))
    pool = list(set(top) | set(extra))
    pool = sorted(pool)
    m = len(pool)

    # Build A matrix (n_t x m) sparse: A[i, j] = 1 if pool[j] covers t-subset i
    from scipy.sparse import lil_matrix
    A = lil_matrix((n_t, m), dtype=np.int8)
    for j, bi in enumerate(pool):
        for idx in block_tsubs(blocks[bi]):
            A[idx, j] = 1
    A = A.tocsr()

    c = np.ones(m)
    constraint = LinearConstraint(A, 1, None)
    bounds = Bounds(0, 1)

    try:
        res = milp(c, integrality=np.ones(m), constraints=constraint,
                   bounds=bounds, options={'time_limit': 40, 'mip_rel_gap': 0.0,
                                           'mip_absgap': 0.0, 'verbose': 0})
    except Exception:
        return blocks

    if res.x is not None:
        sel = [pool[j] for j in range(m) if res.x[j] > 0.5]
        new_blocks = [blocks[i] for i in sel]
        # verify coverage
        if len(new_blocks) < len(blocks):
            cov = [False] * n_t
            for b in new_blocks:
                for idx in block_tsubs(b):
                    cov[idx] = True
            if all(cov):
                return new_blocks

    return blocks
# EVOLVE-BLOCK-END
