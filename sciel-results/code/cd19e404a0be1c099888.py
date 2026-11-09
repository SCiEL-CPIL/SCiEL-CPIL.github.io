# EVOLVE-BLOCK-START
import time
import numpy as np


def _union1d(base, N=8):
    """Total length of the union of N unit-height sheared intervals.

    Interval for index i: [base[i] + c*i/N, base[i] + 1/N + c*i/N], c in [0,1].
    Union length = integral over c of (sum of merged 1-D interval lengths).
    """
    c = np.linspace(0.0, 1.0, N + 1)
    w = np.zeros(N + 1)
    w[0] = 0.5
    w[-1] = 0.5
    w[1:-1] = 1.0
    w *= 1.0 / N
    total = 0.0
    for k in range(N + 1):
        ck = c[k]
        lo = base + ck * (np.arange(N) / N)
        hi = lo + 1.0 / N
        order = np.argsort(lo)
        sl = lo[order]
        sh = hi[order]
        cur = sh[0]
        length = 0.0
        for m in range(1, N):
            if sl[m] > cur:
                length += cur - sh[m - 1]
                cur = sh[m]
            else:
                if sh[m] > cur:
                    cur = sh[m]
        length += cur - sh[-1]
        total += w[k] * length
    return total


def _union2d(base, N=8):
    """Total area of the union of N unit-height sheared squares.

    Square for index i: [base[i]+c*i/N, base[i]+1/N+c*i/N]^2.
    Area = integral over c of (1-D union length)^2.
    """
    c = np.linspace(0.0, 1.0, N + 1)
    w = np.zeros(N + 1)
    w[0] = 0.5
    w[-1] = 0.5
    w[1:-1] = 1.0
    w *= 1.0 / N
    total = 0.0
    for k in range(N + 1):
        ck = c[k]
        lo = base + ck * (np.arange(N) / N)
        hi = lo + 1.0 / N
        order = np.argsort(lo)
        sl = lo[order]
        sh = hi[order]
        cur = sh[0]
        length = 0.0
        for m in range(1, N):
            if sl[m] > cur:
                length += cur - sh[m - 1]
                cur = sh[m]
            else:
                if sh[m] > cur:
                    cur = sh[m]
        length += cur - sh[-1]
        total += w[k] * length * length
    return total


def _local_search(base, N, obj, iters, rng, step, deadline):
    """Coordinate-descent local minimization of a 1-D base vector."""
    best = float(obj(base))
    b = base.copy()
    for _ in range(iters):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            for d in (-step, step):
                nb = b.copy()
                nb[i] += d
                v = obj(nb)
                if v < best - 1e-12:
                    best = v
                    b = nb
                    improved = True
        if not improved:
            step *= 0.7
            if step < 1e-5:
                break
    return b, best


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 620.0
    rng = np.random.default_rng(12345)

    # Independent 1-D problems: minimize Vx = union length of the N sheared intervals.
    best_x = np.zeros(N)
    best_xv = _union1d(best_x, N)
    for _ in range(4):
        b = rng.uniform(-0.5, 0.5, N)
        bb, bv = _local_search(b, N, _union1d, 600, rng, 0.05, deadline)
        if bv < best_xv:
            best_xv = bv
            best_x = bb
        if time.time() > deadline:
            break

    # y-dimension: identical structure, so reuse the optimal 1-D solution.
    best_y = best_x.copy()
    best_yv = best_xv

    # Product construction: x depends only on i, y only on j -> volume = Vx * Vy.
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    base_x = best_x[i_grid.flatten()]
    base_y = best_y[j_grid.flatten()]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
