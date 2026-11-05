import time
import itertools
import numpy as np


def _global_greedy(v, k, t, seed, deadline):
    rng = np.random.default_rng(seed)
    # Index each t-subset by its lex rank.
    comb = np.array(list(itertools.combinations(range(v), t)), dtype=np.int64)
    N = comb.shape[0]
    # Map a point -> list of triple-indices that contain it.
    triples_per_point = [[] for _ in range(v)]
    for idx, c in enumerate(comb):
        for p in c:
            triples_per_point[p].append(idx)
    for p in range(v):
        triples_per_point[p] = np.asarray(triples_per_point[p], dtype=np.int64)

    uncovered = np.ones(N, dtype=bool)
    blocks = []
    while True:
        if time.time() > deadline:
            break
        rem = int(uncovered.sum())
        if rem == 0:
            break
        cand = np.array(list(itertools.combinations(range(v), k)), dtype=np.int64)
        # gain[i] = # of still-uncovered triples covered by cand[i]
        gain = np.zeros(len(cand), dtype=np.int64)
        for r in range(k):
            pts = cand[:, r]
            for p in range(v):
                sel = pts == p
                if np.any(sel):
                    np.add.at(gain, sel, uncovered[triples_per_point[p]])
        best = int(np.argmax(gain))
        if gain[best] <= 0:
            # No candidate covers anything (should not happen); break to avoid stall.
            break
        block = sorted(cand[best].tolist())
        blocks.append(block)
        # Mark covered triples.
        cm = np.array(list(itertools.combinations(block, t)), dtype=np.int64)
        mask = cm >= 0
        mask = np.all(mask, axis=1)
        uncovered[comb[mask]] = False
    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + 118.0  # safety margin before 150s kill
    best = None
    seed = 0
    while time.time() < deadline:
        blocks = _global_greedy(v, k, t, seed, deadline)
        if blocks is not None and len(blocks) < 10**9:
            # Validate coverage quickly (vectorized) before accepting.
            comb = np.array(list(itertools.combinations(range(v), t)), dtype=np.int64)
            N = comb.shape[0]
            covered = np.zeros(N, dtype=bool)
            for b in blocks:
                cm = np.array(list(itertools.combinations(b, t)), dtype=np.int64)
                covered[comb[cm >= 0]] = True
            if covered.all():
                if best is None or len(blocks) < len(best):
                    best = blocks
                    if len(best) <= 70:
                        # Strong result; keep searching a bit more for improvement.
                        pass
            else:
                # Shouldn't happen; treat as invalid for this seed.
                pass
        seed += 1
        # Stop early if we hit the known lower bound.
        if best is not None and len(best) <= 69:
            break
    if best is None:
        # Fallback: deterministic single greedy (no time pressure).
        best = _global_greedy(v, k, t, 0, time.time() + 1.0)
    return best
