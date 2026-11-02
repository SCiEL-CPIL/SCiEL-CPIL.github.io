import time
import random
import itertools
import numpy as np


def _triple_index(a, b, c, v):
    # a<b<c, 0<=a<b<c<v ; bijection to 0..C(v,3)-1
    return (v - 1 - c) * ((v - 1) * (v - 2)) // 2 + (v - 1 - b) * (v - 1) + a


def _build_triple_table(v):
    idx = np.empty((v, v), dtype=np.int32)
    for a in range(v):
        for b in range(a + 1, v):
            idx[a, b] = _triple_index(a, b, v - 1, v)
    pair_sets = {}
    for a in range(v):
        for b in range(a + 1, v):
            lst = []
            for c in range(b + 1, v):
                lst.append(_triple_index(a, b, c, v))
            pair_sets[(a, b)] = np.asarray(lst, dtype=np.int32)
    return idx, pair_sets


def _greedy(v, k, t, idx, pair_sets, order, tie_seed, prefer_covered, deadline):
    rng = random.Random(tie_seed)
    total = len(pair_sets)
    covered = np.zeros(total, dtype=bool)
    blocks = []
    point_set = list(order)
    n_covered = 0
    while n_covered < total:
        if time.time() > deadline:
            break
        # choose seed point
        if len(point_set) < k:
            point_set = list(order)
            rng.shuffle(point_set)
        seed = point_set.pop(0)
        block = [seed]
        while len(block) < k:
            if time.time() > deadline:
                break
            # candidate points
            in_block = set(block)
            candidates = [p for p in range(v) if p not in in_block]
            # current block triple set
            cur = None
            for a, b in itertools.combinations(block, 2):
                ps = pair_sets[(min(a, b), max(a, b))]
                cur = ps if cur is None else np.concatenate((cur, ps))
            if cur is None:
                cur = np.zeros(0, dtype=np.int32)
            cur_u = np.unique(cur)
            new_gain = np.count_nonzero(~covered[cur_u])
            cov_gain = np.count_nonzero(covered[cur_u])
            # score each candidate
            best = -1
            best_score = None
            rng.shuffle(candidates)
            for x in candidates:
                # new triples formed by adding x
                add = None
                for p in block:
                    ps = pair_sets[(min(p, x), max(p, x))]
                    add = ps if add is None else np.concatenate((add, ps))
                add_u = np.unique(add)
                ng = np.count_nonzero(~covered[add_u])
                cg = np.count_nonzero(covered[add_u])
                if prefer_covered:
                    score = (ng, cg)
                else:
                    score = (ng, -cg)
                if best_score is None or score > best_score:
                    best_score = score
                    best = x
            block.append(best)
        # finalize block
        block.sort()
        blocks.append(block)
        # mark covered
        cur = None
        for a, b in itertools.combinations(block, 2):
            ps = pair_sets[(min(a, b), max(a, b))]
            cur = ps if cur is None else np.concatenate((cur, ps))
        if cur is not None:
            cur_u = np.unique(cur)
            newly = cur_u[~covered[cur_u]]
            covered[cur_u] = True
            n_covered += len(newly)
    # validate coverage
    if not covered.all():
        return None
    return blocks


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 120.0
    # sanity: must be able to cover
    if k < t:
        return []
    idx, pair_sets = _build_triple_table(v)
    total = len(pair_sets)

    base_order = list(range(v))
    rng0 = random.Random(12345)

    best = None
    # deterministic baseline first (prefer_covered=False, no shuffle)
    b = _greedy(v, k, t, idx, pair_sets, base_order, 0, False, deadline)
    if b is not None:
        best = b

    # randomized restarts
    attempt = 0
    while time.time() < deadline - 2.0:
        attempt += 1
        order = list(base_order)
        rng0.shuffle(order)
        prefer = (attempt % 3 == 0)
        b = _greedy(v, k, t, idx, pair_sets, order, attempt * 7919, prefer, deadline)
        if b is not None:
            if best is None or len(b) < len(best):
                best = b
    if best is None:
        # should not happen; fall back to a trivial-but-valid construction if possible
        # (only valid if k==v covers everything)
        if k == v:
            return [list(range(v))]
        return []
    return best
