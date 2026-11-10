# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 580.0   # safety margin well under the 600 s wall-clock kill
_N = 8
_EPS = 1e-12


def _union1d(a, c):
    """1D union length of intervals [a_i + c*i/N, a_i + c*i/N + 1/N]."""
    N = _N
    w = 1.0 / N
    starts = a + c * np.arange(N) / N
    ends = starts + w
    order = np.argsort(starts)
    s = starts[order]
    e = ends[order]
    total = 0.0
    cur_s = s[0]
    cur_e = e[0]
    for k in range(1, N):
        if s[k] > cur_e:
            total += cur_e - cur_s
            cur_s = s[k]
            cur_e = e[k]
        elif e[k] > cur_e:
            cur_e = e[k]
    total += cur_e - cur_s
    return total


def _volx(a, c):
    N = _N
    w = 1.0 / N
    starts = a + c * np.arange(N) / N
    ends = starts + w
    order = np.argsort(starts)
    s = starts[order]
    e = ends[order]
    total = 0.0
    cur_s = s[0]
    cur_e = e[0]
    for k in range(1, N):
        if s[k] > cur_e:
            total += cur_e - cur_s
            cur_s = s[k]
            cur_e = e[k]
        elif e[k] > cur_e:
            cur_e = e[k]
    total += cur_e - cur_s
    return total * w


def _objective(a):
    """Total 3D union volume for a given base vector a (factorized exact value)."""
    c = np.linspace(0.0, 1.0, 201)
    vals = np.empty(201)
    for m in range(201):
        vals[m] = _volx(a, c[m])
    return float(np.trapz(vals, c))


def _polish(a):
    """High-precision local polish: steepest-descent on the exact objective."""
    a = np.array(a, dtype=np.float64)
    f0 = _objective(a)
    h = 1e-5
    step = 1e-3
    for _ in range(400):
        if time.time() > _DEADLINE:
            break
        g = np.empty_like(a)
        for i in range(_N):
            ap = a.copy(); am = a.copy()
            ap[i] += h; am[i] -= h
            g[i] = (_objective(ap) - _objective(am)) / (2.0 * h)
        gnorm = float(np.linalg.norm(g))
        if gnorm < 1e-12:
            break
        d = -g / gnorm
        best_f = f0
        best_a = a
        for _t in range(25):
            cand = a + step * d
            fc = _objective(cand)
            if fc < best_f - 1e-14:
                best_f = fc
                best_a = cand
            step *= 0.5
        a = best_a
        f0 = best_f
        if step < 1e-9:
            break
    return a


def _solve_1d():
    """Global + local optimization of the 1D base vector (8-dim)."""
    rng = np.random.default_rng(7)
    # Candidate seeds: the reference Keich binary construction, plus random starts.
    pos = int(np.log2(_N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(_N, dtype=np.float64)
    for idx in range(_N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    starts = [keich]
    for _ in range(12):
        starts.append(rng.uniform(-0.5, 0.5, _N))

    # Coarse global grid over a plausible box.
    best_a = None
    best_f = np.inf
    gvals = np.linspace(-1.0, 1.0, 21)
    for x0 in gvals:
        for x1 in gvals:
            for x2 in gvals:
                for x3 in gvals:
                    if time.time() > _DEADLINE:
                        return best_a if best_a is not None else keich
                    a = np.array([x0, x1, x2, x3,
                                  x0 + 0.5, x1 + 0.5, x2 + 0.5, x3 + 0.5],
                                 dtype=np.float64)
                    f = _objective(a)
                    if f < best_f:
                        best_f = f
                        best_a = a.copy()

    # Polish the best grid point and each random/Keich start.
    candidates = []
    if best_a is not None:
        candidates.append(best_a)
    candidates.extend(starts)
    for a0 in candidates:
        if time.time() > _DEADLINE:
            break
        a = _polish(a0)
        f = _objective(a)
        if f < best_f:
            best_f = f
            best_a = a

    return best_a


def kakeya_needle_3d8() -> np.ndarray:
    """
    Discrete 3-D Kakeya set: union of 64 sheared tubes (N = 8).

    The 3D union volume factorizes exactly into the product of two independent 1D
    projection unions, V = int_0^1 Lx(c) Ly(c) dc.  We therefore solve two parallel
    8-dimensional 1D interval-packing problems with a deterministic exact solver
    (coarse global grid + high-precision steepest-descent polish on the exact
    objective), then place base x as a function of i and base y as a function of j.

    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube (i, j) with i = k // 8,
            j = k % 8; column 0 = base x, column 1 = base y.
    """
    ax = _solve_1d()
    ay = _solve_1d()

    i_grid, j_grid = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")
    base_x = ax[i_grid.flatten()]
    base_y = ay[j_grid.flatten()]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
