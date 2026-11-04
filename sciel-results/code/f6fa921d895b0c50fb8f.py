import time
import itertools
import numpy as np


def construct(v, k, t):
    start = time.time()
    DEADLINE = 115.0  # safety margin before the 150 s kill

    # ---- triple id mapping ----
    triples = list(itertools.combinations(range(v), t))
    triple_id = {tr: i for i, tr in enumerate(triples)}
    n_triples = len(triples)

    # ---- all k-blocks ----
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)

    # positions of t-subsets inside a k-block
    kt = list(itertools.combinations(range(k), t))
    n_sub = len(kt)

    # block_triples[bi, j] = triple-id of the j-th t-subset of block bi
    block_triples = np.empty((n_blocks, n_sub), dtype=np.int32)
    for bi, blk in enumerate(blocks):
        row = block_triples[bi]
        for j, pos in enumerate(kt):
            row[j] = triple_id[tuple(blk[p] for p in pos)]

    # ---- global greedy (vectorized gain over ALL blocks) ----
    uncov = np.ones(n_triples, dtype=bool)
    chosen = []
    while uncov.any():
        if time.time() - start > DEADLINE:
            break
        gains = uncov[block_triples].sum(axis=1)
        b = int(np.argmax(gains))
        if gains[b] <= 0:
            break
        chosen.append(b)
        uncov[block_triples[b]] = False

    # ---- defensive: force-cover any leftover triples (should not trigger) ----
    if uncov.any():
        for tri, tr in enumerate(triples):
            if not uncov[tri]:
                continue
            base = list(tr)
            for x in range(v):
                if x not in base and len(base) < k:
                    base.append(x)
            base = sorted(base)
            idx = blocks.index(tuple(base))
            chosen.append(idx)
            for j, pos in enumerate(kt):
                sub = tuple(base[p] for p in pos)
                uncov[triple_id[sub]] = False
            if time.time() - start > DEADLINE:
                break

    # ---- cover counts ----
    cover_count = np.zeros(n_triples, dtype=np.int32)
    for b in chosen:
        cover_count[block_triples[b]] += 1

    # ---- safe block-removal passes ----
    def try_remove():
        changed = True
        while changed and (time.time() - start < DEADLINE):
            changed = False
            for i in range(len(chosen)):
                bt = block_triples[chosen[i]]
                if np.all(cover_count[bt] >= 2):
                    cover_count[bt] -= 1
                    chosen.pop(i)
                    changed = True
                    break

    try_remove()

    result = [list(blocks[b]) for b in chosen]
    return result
