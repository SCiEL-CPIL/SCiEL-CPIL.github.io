"""Photonic-crystal unit cell design: maximise the 4-5 TM band gap of a square lattice."""
import time
import math
import numpy as np

_T_START = None


def _now():
    global _T_START
    if _T_START is None:
        _T_START = time.time()
    return time.time() - _T_START


def _build_cell(size, r_cross, r_disk, r_ring, ring_w):
    """Build a symmetric 32x32 cell from a superposition of symmetric shapes.

    Shapes (all 4-fold symmetric about the centre):
      - cross (diamond):  |x| + |y| < r_cross
      - disk:             x^2 + y^2 < r_disk
      - ring:             r_ring - ring_w/2 < sqrt(x^2+y^2) < r_ring + ring_w/2
    The cross is the dominant 4th-harmonic driver for the TM 4-5 gap.
    """
    half = size / 2.0
    cell = np.zeros((size, size), dtype=np.int8)
    # Precompute coordinates relative to centre (in lattice units, cell = 1)
    # pixel (i,j) centre at (i+0.5)/size - 0.5
    xs = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(xs, xs, indexing='xy')  # X[i,j] = x of pixel (i,j)
    ax = np.abs(X)
    ay = np.abs(Y)
    r2 = X * X + Y * Y

    mask = np.zeros((size, size), dtype=bool)
    if r_cross > 0:
        mask |= (ax + ay) < r_cross
    if r_disk > 0:
        mask |= r2 < r_disk * r_disk
    if r_ring > 0:
        rr = np.sqrt(r2)
        mask |= (rr > r_ring - 0.5 * ring_w) & (rr < r_ring + 0.5 * ring_w)
    cell[mask] = 1
    return cell


def _fourier_eps_inv(cell, eps_low, eps_high, nmax):
    """Return the 1/eps values and the Fourier coefficient array for |kx|,|ky| <= nmax."""
    size = cell.shape[0]
    eps = np.where(cell == 1, eps_high, eps_low)
    eps_inv = 1.0 / eps
    # pixel centres
    xs = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    # Fourier coefficient of eps_inv at (m,n): integral over cell of eps_inv(x,y) e^{2pi i (m x + n y)}
    # approximated by sum over pixels (each pixel area = 1/size^2)
    coefs = {}
    for m in range(-nmax, nmax + 1):
        for n in range(-nmax, nmax + 1):
            phase = np.exp(2j * math.pi * (m * X + n * Y))
            coefs[(m, n)] = np.sum(eps_inv * phase) / (size * size)
    return eps_inv, coefs


def _surrogate_gap(cell, eps_low, eps_high, nmax, npts):
    """Fast TM band-structure surrogate using a plane-wave Galerkin method.

    Returns the 4-5 gap-midgap ratio over a coarse irreducible BZ path.
    TM: operator uses Fourier of 1/eps (eps_inv).
    """
    size = cell.shape[0]
    eps_inv, coefs = _fourier_eps_inv(cell, eps_low, eps_high, nmax)

    # Build the plane-wave basis: all (m,n) with |m|,|n| <= nmax
    kvec = [(m, n) for m in range(-nmax, nmax + 1) for n in range(-nmax, nmax + 1)]
    nk = len(kvec)
    idx = {kv: i for i, kv in enumerate(kvec)}

    # H(k) matrix: H_{pq} = (k+G_p)·(k+G_q) * eps_inv_{G_p - G_q}   (TM)
    # where G are reciprocal lattice vectors = 2pi*(m,n). We factor 2pi later.
    def H_matrix(kx, ky):
        # k in units of 2pi (so G = (m,n) in same units)
        H = np.empty((nk, nk), dtype=complex)
        for p in range(nk):
            mp, np_ = kvec[p]
            kpx = kx + mp
            kpy = ky + np_
            for q in range(nk):
                mq, nq = kvec[q]
                gp = mp - mq
                gn = np_ - nq
                H[p, q] = (kpx * (kx + mq) + kpy * (ky + nq)) * coefs[(gp, gn)]
        return H

    # Irreducible BZ path: from Gamma(0,0) to X(0.5,0) to M(0.5,0.5) to Gamma
    # sample points in units of 2pi (i.e. kx,ky in [0,0.5])
    pts = []
    Nseg = 6
    for t in range(Nseg + 1):
        f = t / Nseg
        pts.append((0.0, f * 0.5))          # Gamma -> X
        pts.append((f * 0.5, 0.5))          # X -> M
        pts.append(((1 - f) * 0.5, (1 - f) * 0.5))  # M -> Gamma
    # dedupe
    uniq = []
    for p in pts:
        if not uniq or (abs(p[0] - uniq[-1][0]) > 1e-9 or abs(p[1] - uniq[-1][1]) > 1e-9):
            uniq.append(p)

    max4 = 0.0
    min5 = np.inf
    for (kx, ky) in uniq:
        H = H_matrix(kx, ky)
        H = 0.5 * (H + H.conj().T)  # symmetrize
        try:
            ev = np.linalg.eigvalsh(H)
        except np.linalg.LinAlgError:
            continue
        ev = ev * (2 * math.pi) ** 2  # restore 2pi factor
        if len(ev) >= 5:
            max4 = max(max4, ev[3])
            min5 = min(min5, ev[4])
    if not np.isfinite(min5) or min5 <= 0:
        return -1.0
    gap = 2.0 * (min5 - max4) / (min5 + max4)
    return gap


def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]

    deadline = 110.0  # safety margin well under 120s / 230s wall clock

    # Start from a strong cross (diamond) design, then refine.
    def score_cell(cell):
        return _surrogate_gap(cell, eps_low, eps_high, nmax=3, npts=0)

    # Initial best: a cross with radius tuned by a quick 1-D sweep.
    best_cell = None
    best_score = -1.0
    for r in np.linspace(0.15, 0.55, 9):
        if _now() > deadline:
            break
        cell = _build_cell(size, r_cross=r, r_disk=0.0, r_ring=0.0, ring_w=0.0)
        s = score_cell(cell)
        if s > best_score:
            best_score = s
            best_cell = cell

    # Refine around the best cross radius with a fine sweep.
    if best_cell is not None:
        # Determine the radius that produced best_cell by re-sweeping finely
        if _now() < deadline - 10:
            for r in np.linspace(0.18, 0.52, 25):
                if _now() > deadline:
                    break
                cell = _build_cell(size, r_cross=r, r_disk=0.0, r_ring=0.0, ring_w=0.0)
                s = score_cell(cell)
                if s > best_score:
                    best_score = s
                    best_cell = cell

    # Try adding a small central disk (helps pull band 5 up) and a thin ring.
    if best_cell is not None and _now() < deadline - 20:
        base_r = 0.34
        for rd in [0.0, 0.05, 0.1, 0.15]:
            for rr, rw in [(0.0, 0.0), (0.30, 0.05), (0.35, 0.05)]:
                if _now() > deadline:
                    break
                cell = _build_cell(size, r_cross=base_r, r_disk=rd, r_ring=rr, ring_w=rw)
                s = score_cell(cell)
                if s > best_score:
                    best_score = s
                    best_cell = cell

    # Fallback: ensure we always return a valid symmetric cell.
    if best_cell is None:
        best_cell = _build_cell(size, r_cross=0.34, r_disk=0.0, r_ring=0.0, ring_w=0.0)

    # Enforce exact symmetry about the centre (row/col reversal and transpose).
    c = best_cell
    c = (c + c[::-1, :]) // 2
    c = (c + c[:, ::-1]) // 2
    c = (c + c.T) // 2
    c = (c > 0.5).astype(int)

    # Convert to nested list of ints.
    return [[int(c[i, j]) for j in range(size)] for i in range(size)]
