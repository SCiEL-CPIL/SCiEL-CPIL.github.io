"""Greedy set-cover for K_3(6,1) with numpy-vectorized coverage bookkeeping."""
import time
import itertools
import numpy as np


def _all_words(q, n):
    """Return np.uint8 array of shape (q**n, n) with every word in {0..q-1}^n."""
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    return words


def _spheres(words, q, n, r):
    """Return array of shape (N, (n*r + 1), n): each word plus all radius<=r neighbors."""
    N = words.shape[0]
    cols = []
    cols.append(words)
    for i in range(n):
        for v in range(q - 1):
            w2 = words.copy()
            w2[:, i] = (w2[:, i] + v + 1) % q
            cols.append(w2)
    return np.concatenate(cols, axis=1)


def _tetracode_words(q, n):
    """Tetracode direct-sum construction: 9 * q**(n-4) words."""
    tet = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    out = []
    for h in tet:
        for tail in itertools.product(range(q), repeat=n - 4):
            out.append(h + list(tail))
    return np.array(out, dtype=np.uint8)


def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 110.0  # safety margin under the 150s kill

    # 1) Enumerate the whole space and index each word.
    words = _all_words(q, n)
    N = words.shape[0]
    index = np.zeros(N, dtype=np.int64)
    for k in range(N):
        v = words[k, 0]
        for i in range(1, n):
            v = v * q + words[k, i]
        index[k] = v

    # 2) Candidate pool: tetracode direct sum + every radius-1 sphere.
    cand = _tetracode_words(q, n)
    spheres = _spheres(words, q, n, radius)
    # map each sphere column to its word index
    cand_sph_idx = np.empty((cand.shape[0], spheres.shape[1]), dtype=np.int64)
    for j in range(spheres.shape[1]):
        v = cand[:, j, 0]
        for i in range(1, n):
            v = v * q + cand[:, j, i]
        cand_sph_idx[:, j] = index[v]
    sph_idx = np.empty((N, spheres.shape[1]), dtype=np.int64)
    for j in range(spheres.shape[1]):
        v = words[:, j, 0]
        for i in range(1, n):
            v = v * q + words[:, j, i]
        sph_idx[:, j] = index[v]
    pool = np.concatenate([cand_sph_idx, sph_idx], axis=0)
    M = pool.shape[0]

    # 3) Greedy set cover: repeatedly take the candidate covering the most NEW words.
    uncovered = np.ones(N, dtype=bool)
    remaining = np.ones(M, dtype=bool)
    selected = []
    while True:
        if not remaining.any():
            break
        # count new words covered by each remaining candidate
        new_count = np.bincount(pool[remaining], weights=uncovered, minlength=N).astype(np.int64)
        best_local = int(np.argmax(new_count))
        if new_count[best_local] == 0:
            break  # nothing left to gain
        best_global = int(np.where(remaining)[0][best_local])
        selected.append(best_global)
        covered = pool[best_global]
        uncovered[covered] = False
        remaining[best_global] = False
        if not uncovered.any():
            break
        if time.time() > DEADLINE:
            break

    # 4) Convert selected pool entries back to codewords (dedupe, keep distinct).
    chosen_ids = []
    seen = set()
    for g in selected:
        v = pool[g, 0]
        for i in range(1, n):
            v = v * q + pool[g, i]
        if v not in seen:
            seen.add(v)
            chosen_ids.append(v)
    codewords = []
    for v in chosen_ids:
        c = [v % q]
        v //= q
        for _ in range(n - 1):
            c.append(v % q)
            v //= q
        codewords.append(c[::-1])
    return codewords
