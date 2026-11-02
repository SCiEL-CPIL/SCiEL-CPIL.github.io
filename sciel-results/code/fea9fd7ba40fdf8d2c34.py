import itertools
import time
import random
import numpy as np


def _nCr(n, r):
    if r < 0 or r > n:
        return 0
    r = min(r, n - r)
    res = 1
    for i in range(1, r + 1):
        res = res * (n - r + i) // i
    return res


def _precompute(v, k, t):
    triples = list(itertools.combinations(range(v), t))
    nT = len(triples)
    triple_to_idx = {tr: i for i, tr in enumerate(triples)}

    # For each point, the list of triple indices incident to it
    point_triples = []
    for p in range(v):
        lst = []
        for tr in triples:
            if p in tr:
                lst.append(triple_to_idx[tr])
        point_triples.append(np.array(sorted(lst), dtype=np.int64))

    return triples, triple_to_idx, point_triples


def _peel_construct(v, k, t, seed, triples, triple_to_idx, point_triples):
    """Hypergraph peeling: repeatedly pick the point of minimum degree in the
    residual uncovered-triple hypergraph and cover its incident triples with a
    greedily-built block. Escapes uniform-redundancy traps of set-cover greedy."""
    rng = random.Random(seed)
    nT = len(triples)
    uncovered = np.ones(nT, dtype=np.int32)
    rem = np.array([len(pt) for pt in point_triples], dtype=np.int64)
    blocks = []

    while True:
        if int(uncovered.sum()) == 0:
            break
        # Point of minimum residual degree (tie-broken randomly among the minimum)
        mindeg = int(rem.min())
        cand = np.where(rem == mindeg)[0]
        p = int(cand[rng.randrange(len(cand))])

        # Build a block of size k around p, greedily maximizing newly-covered triples
        block = [p]
        bset = {p}
        while len(block) < k:
            best_gain = -1
            best_x = None
            for x in range(v):
                if x in bset:
                    continue
                tris = [tuple(sorted(tr)) for tr in itertools.combinations(block + [x], t)]
                gain = 0
                for tr in tris:
                    ti = triple_to_idx.get(tr)
                    if ti is not None and uncovered[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            bset.add(best_x)

        block.sort()
        blocks.append(block)

        # Update uncovered and residual degrees
        for tr in itertools.combinations(block, t):
            ti = triple_to_idx[tuple(sorted(tr))]
            if uncovered[ti]:
                uncovered[ti] = 0
        for p2 in block:
            rem[p2] = int(uncovered[point_triples[p2]].sum())

    return blocks


def _local_search(v, k, t, blocks, triples, triple_to_idx, deadline, tag=0):
    """Vectorized local search with smooth soft-min surrogate sum(1/cov)."""
    blocks = [list(b) for b in blocks]
    nT = len(triples)
    nB = len(blocks)
    if nB <= 1:
        return blocks

    btris = [np.array([triple_to_idx[tuple(sorted(tr))] for tr in itertools.combinations(b, t)], dtype=np.int64)
             for b in blocks]

    cov = np.zeros(nT, dtype=np.int32)
    for bt in btris:
        np.add.at(cov, bt, 1)

    def score():
        pos = cov[cov > 0]
        return float(np.sum(1.0 / pos)) + 0.005 * nB

    best_cov = cov.copy()
    best_blocks = [list(b) for b in blocks]
    best_score = score()

    rng = random.Random(1000 + tag)

    while time.time() < deadline:
        order = list(range(nB))
        rng.shuffle(order)
        any_removed = False
        for i in order:
            if time.time() >= deadline:
                break
            if i >= nB:
                continue
            bt = btris[i]
            if np.any(cov[bt] <= 1):
                continue
            np.add.at(cov, bt, -1)
            blocks.pop(i)
            btris.pop(i)
            nB -= 1
            any_removed = True
            s = score()
            if s < best_score:
                best_score = s
                best_cov = cov.copy()
                best_blocks = [list(b) for b in blocks]
        if not any_removed:
            break

    return best_blocks


def _polish(v, k, t, blocks, triples, triple_to_idx, deadline):
    """Final high-precision polish: tighter tolerance, more removal passes."""
    blocks = [list(b) for b in blocks]
    nT = len(triples)
    btris = [np.array([triple_to_idx[tuple(sorted(tr))] for tr in itertools.combinations(b, t)], dtype=np.int64)
             for b in blocks]
    cov = np.zeros(nT, dtype=np.int32)
    for bt in btris:
        np.add.at(cov, bt, 1)
    nB = len(blocks)
    rng = random.Random(99999)
    for _ in range(300):
        if time.time() >= deadline:
            break
        improved = False
        order = list(range(nB))
        rng.shuffle(order)
        for i in order:
            if time.time() >= deadline:
                break
            if i >= nB:
                continue
            bt = btris[i]
            if np.any(cov[bt] <= 1):
                continue
            np.add.at(cov, bt, -1)
            blocks.pop(i)
            btris.pop(i)
            nB -= 1
            improved = True
        if not improved:
            break
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 105.0  # safety margin before 120 s kill

    triples, triple_to_idx, point_triples = _precompute(v, k, t)

    best_blocks = None
    best_len = float('inf')

    seed = 0
    while time.time() < deadline:
        if time.time() + 8.0 > deadline:
            break

        # Hypergraph peeling construction
        blocks = _peel_construct(v, k, t, seed, triples, triple_to_idx, point_triples)

        # Local search with smooth surrogate
        if time.time() + 10.0 < deadline:
            blocks = _local_search(v, k, t, blocks, triples, triple_to_idx,
                                   min(time.time() + 10.0, deadline), tag=seed)

        # Final high-precision polish
        if time.time() + 5.0 < deadline:
            blocks = _polish(v, k, t, blocks, triples, triple_to_idx,
                             min(time.time() + 5.0, deadline))

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]

        if best_len == 69:
            break
        seed += 1

    return best_blocks
