"""Local search for K2,2,2-free 2-colourings, vectorized with NumPy."""
import time
import numpy as np

DEADLINE = 130.0  # safety margin under the 150 s wall-clock kill


def _paley(n):
    """Paley graph on n vertices (n odd prime, (n-1)/4 a square)."""
    A = np.zeros((n, n), dtype=np.int8)
    for x in range(1, n):
        if (pow(x, (n - 1) // 2, n) == 1):  # x is a nonzero square mod n
            for i in range(n):
                A[i, (i + x) % n] = 1
    np.fill_diagonal(A, 0)
    return A


def _grow(A, n, n2):
    """Extend a valid n-vertex 0/1 adjacency to n2 vertices (n2 <= 31)."""
    B = np.zeros((n2, n2), dtype=np.int8)
    B[:n, :n] = A
    for j in range(n, n2):
        deg = B[:n, j].astype(np.int64)
        if deg.sum() <= (n2 - n) // 2:
            B[j, :n] = deg
        else:
            B[j, :n] = 1 - deg
    for i in range(n, n2):
        for j in range(i + 1, n2):
            B[i, j] = B[j, i] = 0
    return B


def _count(A):
    """Count K2,2,2 copies in a 0/1 adjacency matrix (NumPy, no Python loops)."""
    return int(np.sum((A @ A) ** 2))


def _valid(A):
    return _count(A) == 0


def _flip_gain(A, i, j):
    """Change in total K2,2,2 count if we toggle edge (i,j)."""
    ci = A[i].astype(np.int64)
    cj = A[j].astype(np.int64)
    c = ci @ cj
    s = int(ci.sum()) + int(cj.sum())
    if A[i, j] == 1:  # removing the edge
        return -4 * (c - s + 1)
    return 4 * (c + 1) - 4 * s  # adding the edge


def _local_search(A, n, budget=0.0):
    """Simulated annealing over edge flips; returns best valid A found (or best A)."""
    A = A.copy()
    order = np.arange(n * (n - 1) // 2)
    best_A = A.copy()
    best_cnt = _count(A)
    T = 2.0
    for it in range(4000):
        if budget > 0 and (time.time() - budget) > 2.0:
            break
        idx = np.random.randint(n * (n - 1) // 2)
        i = 0
        while idx >= n - i:
            idx -= n - i
            i += 1
        j = i + 1 + idx
        g = _flip_gain(A, i, j)
        if g < 0 or (np.random.rand() < np.exp(-g / T)):
            A[i, j] = 1 - A[i, j]
            A[j, i] = A[i, j]
            cnt = _count(A)
            if cnt < best_cnt:
                best_cnt = cnt
                best_A = A.copy()
                if cnt == 0:
                    return best_A
        T *= 0.985
    return best_A


def construct():
    start = time.time()
    deadline = start + DEADLINE
    np.random.seed(12345)

    # Build a 25-vertex valid base from the Paley graph on 25 vertices.
    A = _paley(25)
    if not _valid(A):
        A = _local_search(A, 25, budget=start)
    base = A.copy()

    best = (25, base)

    # Try to extend the base to larger n and repair with local search.
    for n2 in range(26, 32):
        if time.time() > deadline - 20:
            break
        B = _grow(base, 25, n2)
        B = _local_search(B, n2, budget=start)
        if _valid(B):
            best = (n2, B)
            if n2 >= 30:
                break

    # If we haven't reached the record, spend spare time on more restarts.
    if best[0] < 29 and time.time() < deadline - 5:
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            B = _grow(base, 25, best[0] + 1)
            # random perturbation of a few edges
            for _ in range(8):
                i = np.random.randint(best[0] + 1)
                j = np.random.randint(best[0] + 1)
                if i < j:
                    B[i, j] = 1 - B[i, j]
                    B[j, i] = B[i, j]
            B = _local_search(B, best[0] + 1, budget=start)
            if _valid(B):
                best = (best[0] + 1, B)
                if best[0] >= 29:
                    break

    n = best[0]
    A = best[1]
    colors = [int(A[i, j]) for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
