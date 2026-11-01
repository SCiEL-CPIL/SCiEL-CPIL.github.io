"""Covering design C(24,7,3): randomized bitset-greedy + local search, many restarts."""
import time
import random
import itertools
import numpy as np

_CACHE = {}


def _prep(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]

    n_t = 1
    for i in range(1, t + 1):
        n_t = n_t * (v - t + i) // i          # C(v, t)

    W = (n_t + 63) // 64
    n_pts = v

    # triple -> linear index (lexicographic)
    idx = np.empty(n_t, dtype=np.int64)
    inv_a = np.empty(n_t, dtype=np.int64)
    inv_b = np.empty(n_t, dtype=np.int64)
    inv_c = np.empty(n_t, dtype=np.int64)
    ci = 0
    for a in range(v - 2):
        for b in range(a + 1, v - 1):
            for c in range(b + 1, v):
                idx[a * v * v + b * v + c] = ci
                inv_a[ci] = a; inv_b[ci] = b; inv_c[ci] = c
                ci += 1

    # point masks
    masks = np.zeros((n_pts, W), dtype=np.uint64)
    for x in range(n_pts):
        mx = masks[x]
        for y in range(x + 1, v):
            for z in range(y + 1, v):
                i = int(idx[x * v * v + y * v + z])
                mx[i >> 6] |= (1 << (i & 63))
        for y in range(x):
            for z in range(y + 1, v):
                if z == x:
                    continue
                i = int(idx[y * v * v + x * v + z])
                mx[i >> 6] |= (1 << (i & 63))
            for z in range(x + 1, v):
                i = int(idx[y * v * v + z * v + x])
                mx[i >> 6] |= (1 << (i & 63))

    # block masks for every k-subset
    block_masks = np.empty((0, W), dtype=np.uint64)
    for combo in itertools.combinations(range(v), k):
        m = np.zeros(W, dtype=np.uint64)
        for x in combo:
            m |= masks[x]
        block_masks = np.vstack((block_masks, m))

    full_mask = np.zeros(W, dtype=np.uint64)
    full_mask[:-1] = 0xFFFFFFFFFFFFFFFF
    if n_t & 63:
        full_mask[-1] = (1 << (n_t & 63)) - 1

    _CACHE[key] = (masks, block_masks, full_mask, inv_a, inv_b, inv_c, W, n_t)
    return _CACHE[key]


def construct(v, k, t):
    masks, block_masks, full_mask, inv_a, inv_b, inv_c, W, n_t = _prep(v, k, t)
    n_pts = v
    nb = block_masks.shape[0]
    BM = block_masks
    M = masks
    zero = np.zeros(W, dtype=np.uint64)

    deadline = time.time() + 110.0

    def gain_partial(part, cur):
        if len(part) < t - 1:
            return zero
        acc = np.zeros(W, dtype=np.uint64)
        for x in part:
            acc |= M[x]
        return acc & cur

    def local_search(blocks):
        covered = np.zeros(W, dtype=np.uint64)
        for b in blocks:
            covered |= BM[b]
        improved = True
        while improved:
            improved = False
            for i in range(len(blocks)):
                rest = covered & ~BM[blocks[i]]
                if np.array_equal(rest, covered):
                    covered = rest
                    del blocks[i]
                    improved = True
                    break
            if improved:
                continue
            for j in range(nb):
                newcov = covered | BM[j]
                if np.array_equal(newcov, covered):
                    continue
                for i in range(len(blocks)):
                    rest = newcov & ~BM[blocks[i]]
                    if np.array_equal(rest, newcov):
                        blocks[i] = j
                        covered = newcov
                        improved = True
                        break
                if improved:
                    break
        return blocks

    best = None
    best_len = 10 ** 9
    rng = random.Random(12345)

    while True:
        if time.time() > deadline:
            break
        cur = full_mask.copy()
        blocks = []
        while True:
            nz = np.nonzero(cur)[0]
            if len(nz) == 0:
                break
            w = int(nz[0])
            word = cur[w]
            bit = word & (-word)
            tri = (w << 6) + (bit.bit_length() - 1)
            a = int(inv_a[tri]); b = int(inv_b[tri]); c = int(inv_c[tri])
            part = [a, b, c]
            g = gain_partial(part, cur)
            cur = cur & ~g
            while len(part) < k:
                cands = [x for x in range(n_pts) if x not in part]
                gp = gain_partial(part, cur)
                scored = []
                for x in cands:
                    gg = gp | M[x]
                    scored.append((int(np.count_nonzero(gg & cur)), x))
                scored.sort(reverse=True)
                top = [x for _, x in scored[:8]] or cands
                x = rng.choice(top)
                part.append(x)
                g = gain_partial(part, cur)
                cur = cur & ~g
            part.sort()
            blocks.append(tuple(part))
        blocks = local_search(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best = [list(b) for b in blocks]

    if best is None:
        best = [list(range(k))]
    return best
