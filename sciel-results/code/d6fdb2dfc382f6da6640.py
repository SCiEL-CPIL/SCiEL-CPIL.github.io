"""Structural combination: fast randomized greedy + local search + ILP block-removal."""
import itertools
import time
import random
import numpy as np

def _all_tsubsets(v, t):
    return list(itertools.combinations(range(v), t))

def _greedy_construct(v, k, t, deadline, seed, mode='basic'):
    """Greedy covering. mode in {'basic','div','pair'}.
    Returns (blocks, valid)."""
    all_subs = _all_tsubsets(v, t)
    n_subs = len(all_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}

    # For each point x: list of t-subset indices that contain x
    point_subs = [[] for _ in range(v)]
    for idx, sub in enumerate(all_subs):
        for x in sub:
            point_subs[x].append(idx)
    # For each point x: list of (t-1)-subsets of the other points that with x form a t-subset
    # Represent each (t-1)-subset as a frozenset of points; we map to a key
    # Precompute for each x a dict: frozenset(rest) -> t-subset index (only rest of size t-1)
    # Instead, precompute for each x a set of (t-1)-tuples (sorted) -> idx
    # Build via iterating all t-subsets containing x
    point_rest = [dict() for _ in range(v)]
    for idx, sub in enumerate(all_subs):
        for i, x in enumerate(sub):
            rest = tuple(sub[:i] + sub[i+1:])
            point_rest[x][rest] = idx

    uncovered = np.ones(n_subs, dtype=np.int32)
    n_uncovered = n_subs
    blocks = []
    rng = random.Random(seed)

    # Pair frequency for diversity scoring
    pair_count = np.zeros((v, v), dtype=np.int32)

    while n_uncovered > 0:
        if time.time() > deadline:
            break
        unc_idx = np.where(uncovered > 0)[0]
        if len(unc_idx) == 0:
            break
        # Pick a starting triple
        if mode == 'basic':
            start_sub = all_subs[unc_idx[0]]
        else:
            start_sub = all_subs[rng.choice(unc_idx.tolist())]

        block = list(start_sub)
        block_set = set(block)

        while len(block) < k:
            if time.time() > deadline:
                break
            best_x = -1
            best_key = None
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                if len(block) >= t - 1:
                    for rest in itertools.combinations(block, t - 1):
                        idx = point_rest[x].get(rest)
                        if idx is not None and uncovered[idx] > 0:
                            gain += 1
                if gain <= 0:
                    continue
                if mode == 'div':
                    # diversity: prefer points rarely used
                    div = -0.01 * pair_count[x].sum()
                    key = (gain, div)
                elif mode == 'pair':
                    # pair rarity: prefer points that form rare pairs with block
                    rar = 0
                    for y in block:
                        rar += -pair_count[x, y]
                    key = (gain, rar)
                else:
                    key = (gain,)
                if best_key is None or key > best_key:
                    best_key = key
                    best_x = x
            if best_x == -1:
                # no gain; pick arbitrary
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break
                if best_x == -1:
                    break
            block.append(best_x)
            block_set.add(best_x)

        if len(block) < k:
            break
        block.sort()
        blocks.append(block)
        for x in block:
            for y in block:
                if x != y:
                    pair_count[x, y] += 1
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            if uncovered[idx] > 0:
                uncovered[idx] = 0
                n_uncovered -= 1

    valid = (n_uncovered == 0)
    return blocks, valid

def _local_search(blocks, v, k, t, deadline):
    """Remove redundant blocks and try 2-for-1 merges."""
    if not blocks:
        return blocks
    all_subs = _all_tsubsets(v, t)
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    n_subs = len(all_subs)

    cover_count = np.zeros(n_subs, dtype=np.int32)
    block_subs = []
    for b in blocks:
        subs = [sub_to_idx[s] for s in itertools.combinations(b, t)]
        block_subs.append(subs)
        for idx in subs:
            cover_count[idx] += 1

    if np.any(cover_count == 0):
        return blocks  # invalid

    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(blocks)
        for i in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            redundant = True
            for idx in block_subs[i]:
                if cover_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_subs[i]:
                    cover_count[idx] -= 1
                blocks.pop(i)
                block_subs.pop(i)
                n -= 1
                improved = True

    # 2-for-1 merge
    if time.time() < deadline - 5:
        n = len(blocks)
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(n - 1, i, -1):
                if time.time() > deadline:
                    break
                union = set(block_subs[i]) | set(block_subs[j])
                points = set()
                for idx in union:
                    points.update(all_subs[idx])
                if len(points) <= k:
                    new_block = sorted(points)
                    if len(new_block) < k:
                        for p in range(v):
                            if p not in new_block:
                                new_block.append(p)
                            if len(new_block) == k:
                                break
                    new_block = sorted(new_block[:k])
                    nb_set = set(new_block)
                    if all(set(all_subs[idx]).issubset(nb_set) for idx in union):
                        for idx in block_subs[i]:
                            cover_count[idx] -= 1
                        for idx in block_subs[j]:
                            cover_count[idx] -= 1
                        new_subs = [sub_to_idx[s] for s in itertools.combinations(new_block, t)]
                        for idx in new_subs:
                            cover_count[idx] += 1
                        blocks[i] = new_block
                        block_subs[i] = new_subs
                        blocks.pop(j)
                        block_subs.pop(j)
                        n -= 1
                        improved = True
                        break
            if improved:
                break

    return blocks

def _ilp_reduce(blocks, v, k, t, deadline):
    """Use scipy milp to find a minimum subset of blocks that still covers all triples."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except Exception:
        return blocks
    if not blocks or time.time() > deadline:
        return blocks
    all_subs = _all_tsubsets(v, t)
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    n_subs = len(all_subs)
    n_blocks = len(blocks)
    if n_blocks < 2:
        return blocks
    # A[b, s] = 1 if block b covers t-subset s
    A = np.zeros((n_subs, n_blocks), dtype=np.int8)
    for b, blk in enumerate(blocks):
        for s in itertools.combinations(blk, t):
            A[sub_to_idx[s], b] = 1
    # minimize sum x_b, s.t. A x >= 1, x in {0,1}
    c = np.ones(n_blocks)
    cons = LinearConstraint(A, 1, np.inf)
    integrality = np.ones(n_blocks)
    bounds = Bounds(0, 1)
    try:
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds,
                   options={'time_limit': max(1.0, deadline - time.time() - 2.0)})
    except Exception:
        return blocks
    if res.status not in (0, 2):
        return blocks
    x = res.x
    keep = [b for b in range(n_blocks) if x[b] > 0.5]
    if not keep:
        return blocks
    new_blocks = [blocks[b] for b in keep]
    # verify validity
    covered = set()
    for b in new_blocks:
        for s in itertools.combinations(b, t):
            covered.add(s)
    if len(covered) == n_subs:
        return new_blocks
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # safety margin for 150s kill
    if t > k:
        return []
    if t == 0:
        return []

    best_blocks = None
    best_count = float('inf')

    # First deterministic-ish greedy
    for seed, mode in [(42, 'basic'), (1, 'div'), (2, 'pair')]:
        if time.time() > deadline - 10:
            break
        blocks, valid = _greedy_construct(v, k, t, deadline - 5, seed, mode)
        if valid and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]

    if best_blocks is None:
        # fallback: basic greedy with full time
        blocks, valid = _greedy_construct(v, k, t, deadline, 0, 'basic')
        return blocks

    # Local search on best
    if time.time() < deadline - 20:
        ls = _local_search(best_blocks, v, k, t, deadline - 5)
        if len(ls) < best_count:
            best_count = len(ls)
            best_blocks = ls

    # ILP reduction on best
    if time.time() < deadline - 15:
        il = _ilp_reduce(best_blocks, v, k, t, deadline - 5)
        if len(il) < best_count:
            best_count = len(il)
            best_blocks = il

    # Randomized restarts with local search
    attempt = 0
    while time.time() < deadline - 15:
        attempt += 1
        rng_seed = attempt * 131 + 7
        mode = ['basic', 'div', 'pair'][attempt % 3]
        blocks, valid = _greedy_construct(v, k, t, deadline - 10, rng_seed, mode)
        if valid and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            if time.time() < deadline - 15:
                ls = _local_search(best_blocks, v, k, t, deadline - 5)
                if len(ls) < best_count:
                    best_count = len(ls)
                    best_blocks = ls
                if time.time() < deadline - 15:
                    il = _ilp_reduce(best_blocks, v, k, t, deadline - 5)
                    if len(il) < best_count:
                        best_count = len(il)
                        best_blocks = il

    # Ensure distinct blocks
    seen = set()
    unique = []
    for b in best_blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique.append(b)
    return unique
