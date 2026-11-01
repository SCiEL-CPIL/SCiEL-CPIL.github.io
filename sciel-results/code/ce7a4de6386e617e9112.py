import itertools
import time
import numpy as np


def _triple_index(a, b, c):
    # lexicographic rank of a<b<c in C(v,3); v fixed per call
    # rank = C(a,3) + C(b,2) + C(c,1)  (combinatorial number system)
    return a * (a - 1) * (a - 2) // 6 + b * (b - 1) // 2 + c


def _precompute_indices(v, t):
    # For t=3, precompute index for every triple as a flat array of length C(v,3)
    # indexed by a linear code. We'll build a 3D array idx[a,b,c] for a<b<c.
    idx = np.zeros((v, v, v), dtype=np.int64)
    rank = 0
    for a in range(v):
        for b in range(a + 1, v):
            for c in range(b + 1, v):
                idx[a, b, c] = rank
                rank += 1
    return idx, rank


def _greedy_build(v, k, t, deadline):
    idx, ntrip = _precompute_indices(v, t)
    cnt = np.zeros(ntrip, dtype=np.int8)
    blocks = []
    start = time.time()
    while True:
        if time.time() > deadline:
            break
        # pick seed from an uncovered triple
        unc = np.nonzero(cnt == 0)[0]
        if len(unc) == 0:
            break
        # choose seed triple: one with maximum "spread" to allow good growth
        # just take the first uncovered; but better: pick a triple that is hard
        seed = int(unc[0])
        # decode seed triple
        a, b, c = _decode(idx, seed)
        block = [a, b, c]
        # grow
        while len(block) < k:
            if time.time() > deadline:
                break
            best_x = -1
            best_new = -1
            best_total = -1
            for x in range(v):
                if x in block:
                    continue
                # compute triples formed by x + (t-1)-subset of block
                # for t=3: pairs in block
                new = 0
                total = 0
                for p in range(len(block)):
                    for q in range(p + 1, len(block)):
                        i = block[p]; j = block[q]
                        # triple (min, mid, max)
                        if x < i:
                            ii, jj, kk = x, i, j
                        elif x > j:
                            ii, jj, kk = i, j, x
                        else:
                            ii, jj, kk = i, x, j
                        ti = idx[ii, jj, kk]
                        total += 1
                        if cnt[ti] == 0:
                            new += 1
                if new > best_new or (new == best_new and total > best_total):
                    best_new = new
                    best_total = total
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
        if len(block) < k:
            # can't complete; break to avoid infinite loop
            break
        block.sort()
        blocks.append(block)
        # update cnt
        for p in range(len(block)):
            for q in range(p + 1, len(block)):
                for r in range(q + 1, len(block)):
                    ti = idx[block[p], block[q], block[r]]
                    cnt[ti] += 1
    return blocks, idx, ntrip


def _decode(idx, seed):
    # find (a,b,c) with idx[a,b,c]==seed; do a linear scan over stored triples
    # We'll precompute a reverse map once. For simplicity, scan.
    # This is called once per block (seed), acceptable.
    for a in range(idx.shape[0]):
        for b in range(a + 1, idx.shape[1]):
            for c in range(b + 1, idx.shape[2]):
                if idx[a, b, c] == seed:
                    return a, b, c
    return None


def _deletion_pass(blocks, idx, ntrip, deadline):
    cnt = np.zeros(ntrip, dtype=np.int8)
    for block in blocks:
        for p in range(len(block)):
            for q in range(p + 1, len(block)):
                for r in range(q + 1, len(block)):
                    cnt[idx[block[p], block[q], block[r]]] += 1
    start = time.time()
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for bi in range(len(blocks)):
            if time.time() > deadline:
                return blocks
            # tentatively remove block bi
            block = blocks[bi]
            min_after = None
            for p in range(len(block)):
                for q in range(p + 1, len(block)):
                    for r in range(q + 1, len(block)):
                        ti = idx[block[p], block[q], block[r]]
                        v = cnt[ti] - 1
                        if v < 0:
                            v = 0
                        if min_after is None or v < min_after:
                            min_after = v
            if min_after >= 1:
                # safe to remove
                for p in range(len(block)):
                    for q in range(p + 1, len(block)):
                        for r in range(q + 1, len(block)):
                            cnt[idx[block[p], block[q], block[r]]] -= 1
                blocks.pop(bi)
                changed = True
                break
    return blocks


def construct(v, k, t):
    # deadline with safety margin
    start = time.time()
    deadline = start + 130.0  # 130s of the 150s wall-clock
    blocks, idx, ntrip = _greedy_build(v, k, t, deadline)
    # deletion pass
    blocks = _deletion_pass(blocks, idx, ntrip, deadline)
    # final verification; if invalid, fall back to greedy-only (already valid by construction)
    # Rebuild cnt to verify
    cnt = np.zeros(ntrip, dtype=np.int8)
    for block in blocks:
        for p in range(len(block)):
            for q in range(p + 1, len(block)):
                for r in range(q + 1, len(block)):
                    cnt[idx[block[p], block[q], block[r]]] += 1
    if np.any(cnt == 0):
        # invalid; rebuild without deletion
        blocks, idx, ntrip = _greedy_build(v, k, t, deadline)
    return blocks
