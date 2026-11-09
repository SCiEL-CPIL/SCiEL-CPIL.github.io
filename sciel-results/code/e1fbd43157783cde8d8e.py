"""Photonic-crystal TM band-gap design via Fourier-space inverse design with C4v symmetry."""
import time
import math
import numpy as np

_DEADLINE = 210.0  # safety margin before 230 s kill


def _c4v_fold(coeffs, n):
    """Enforce C4v symmetry on an n x n array of Fourier coefficients.
    C4v orbits under rotations (90/180/270) and reflections."""
    out = np.zeros_like(coeffs, dtype=complex)
    cnt = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            vals = []
            # orbit under C4v: (i,j), (j,-i), (-i,-j), (-j,i) and reflections
            for (a, b) in [(i, j), (j, -i), (-i, -j), (-j, i),
                           (i, -j), (-j, i), (-i, -j), (j, i)]:
                ia = a % n
                ib = b % n
                vals.append(coeffs[ia, ib])
            out[i, j] = np.mean(vals)
            cnt[i, j] += 1
    return out


def _pixel_to_eps(cell, eps_low, eps_high):
    return np.array(cell, dtype=float) * (eps_high - eps_low) + eps_low


def _eps_fourier(eps_map, n):
    """2D DFT of eps map to Fourier coefficients (unnormalised, index 0..n-1)."""
    F = np.fft.fft2(eps_map)
    return F / (n * n)


def _fourier_to_pixel(F, n):
    return np.real(np.fft.ifft2(F * (n * n)))


def _band_solver(F, kx, ky, eps_low, eps_high, max_iter=40, tol=1e-8):
    """Approximate plane-wave (FFT) eigenvalue solver for TM bands.
    TM: (k - G)^2 / eps(k-G).  We build the matrix using inverse-eps in G-space."""
    n = F.shape[0]
    # G-vectors: for a unit cell, G = 2*pi*(m, n) with m,n in integers.
    # Use a symmetric window around 0.
    idx = np.arange(n) - n // 2
    gx, gy = np.meshgrid(idx, idx, indexing='ij')
    Gx = 2.0 * math.pi * gx
    Gy = 2.0 * math.pi * gy
    nG = n * n
    # Build (k-G) vectors for each G
    kxg = kx - Gx.ravel()
    kyg = ky - Gy.ravel()
    k2 = kxg * kxg + kyg * kyg
    # eps at each G (inverse for TM)
    eps_g = np.real(_fourier_to_pixel(F, n))
    # For TM, the eigenvalue problem uses 1/eps in real space convolved, but
    # in a simple FFT plane-wave basis we approximate:
    #   A = diag(k-G)^2  *  F(1/eps)  (in G-space, cyclic convolution)
    inv_eps = 1.0 / np.maximum(eps_g, 1e-6)
    inv_eps_F = np.fft.fft2(inv_eps) / (n * n)
    # Build matrix M where M_{ab} = (k-G_a)^2 * delta-like * F(1/eps)_{G_a-G_b}
    # Approximate: M = diag(k2) @ circulant(inv_eps_F)
    # circulant matrix from inv_eps_F: C_{a,b} = inv_eps_F[(G_a - G_b)]
    # We'll form it via FFT of a shifted kernel.
    # inv_eps_F is indexed by G index; C_{a,b} = inv_eps_F[(idx_a - idx_b) mod n]
    # Build full matrix (n x n per dimension -> n^2 x n^2). For n=8 -> 64x64 ok.
    M = np.zeros((nG, nG), dtype=complex)
    for a in range(nG):
        ia = a % n
        ib = a // n
        for b in range(nG):
            ibx = b % n
            iby = b // n
            da = (ia - ibx) % n
            db = (ib - iby) % n
            M[a, b] = inv_eps_F[da, db]
    M = np.diag(k2.astype(complex)) @ M
    # Hermitian? k2 real, inv_eps_F Hermitian-symmetric -> M should be Hermitian-ish.
    w, _ = np.linalg.eigh(M)
    w = np.real(w)
    return w


def _max_band(w, band_idx):
    return w[band_idx - 1] if len(w) >= band_idx else (w[-1] if len(w) > 0 else 0.0)


def _evaluate_gap(cell, eps_low, eps_high, k_points, n, lower_band):
    """Compute gap-midgap ratio between bands lower_band and lower_band+1."""
    eps_map = _pixel_to_eps(cell, eps_low, eps_high)
    F = _eps_fourier(eps_map, n)
    min_omega5 = float('inf')
    max_omega4 = float('-inf')
    for (kx, ky) in k_points:
        w = _band_solver(F, kx, ky, eps_low, eps_high)
        if len(w) < lower_band + 1:
            continue
        o4 = w[lower_band - 1]
        o5 = w[lower_band]
        if o5 < min_omega5:
            min_omega5 = o5
        if o4 > max_omega4:
            max_omega4 = o4
    if min_omega5 == float('inf') or max_omega4 == float('-inf'):
        return -1.0
    gap = min_omega5 - max_omega4
    mid = 0.5 * (min_omega5 + max_omega4)
    if mid <= 0:
        return -1.0
    return 2.0 * gap / mid


def _k_points_irb(n_pts=8):
    """Wave vectors in the irreducible Brillouin zone (X-M-X line and X-M-X-K-M)."""
    pts = []
    for t in np.linspace(0, 1, n_pts):
        pts.append((math.pi * t, 0.0))          # X -> Gamma? use (pi,0)->(0,0)
    for t in np.linspace(0, 1, n_pts):
        pts.append((math.pi * (1 - t), math.pi * t))  # X -> M
    for t in np.linspace(0, 1, n_pts):
        pts.append((math.pi * t, math.pi * t))  # M -> Gamma
    return pts


def _apply_c4v_pixel(cell):
    """Enforce C4v on a pixel cell (32x32): cell == reverse rows, reverse cols, transpose."""
    n = len(cell)
    a = np.array(cell, dtype=float)
    # C4v: average over 8 symmetry operations about centre.
    # Operations on pixel indices (0-indexed): identity, rot90, rot180, rot270,
    # and reflections. For a square cell, rot90 about centre: (i,j) -> (j, n-1-i).
    ops = [
        lambda i, j: (i, j),
        lambda i, j: (j, n - 1 - i),
        lambda i, j: (n - 1 - i, n - 1 - j),
        lambda i, j: (n - 1 - j, i),
        lambda i, j: (i, n - 1 - j),
        lambda i, j: (n - 1 - i, j),
        lambda i, j: (j, i),
        lambda i, j: (n - 1 - j, n - 1 - i),
    ]
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for op in ops:
                ii, jj = op(i, j)
                s += a[ii, jj]
            out[i, j] = s / 8.0
    return out


def _round_cell(cell, threshold=0.5):
    n = len(cell)
    return [[1 if cell[i][j] >= threshold else 0 for j in range(n)] for i in range(n)]


def _enforce_sym(cell):
    """Ensure exact C4v pixel symmetry by averaging, then threshold."""
    n = len(cell)
    a = np.array(cell, dtype=float)
    ops = [
        lambda i, j: (i, j),
        lambda i, j: (j, n - 1 - i),
        lambda i, j: (n - 1 - i, n - 1 - j),
        lambda i, j: (n - 1 - j, i),
        lambda i, j: (i, n - 1 - j),
        lambda i, j: (n - 1 - i, j),
        lambda i, j: (j, i),
        lambda i, j: (n - 1 - j, n - 1 - i),
    ]
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for op in ops:
                ii, jj = op(i, j)
                s += a[ii, jj]
            out[i, j] = s / 8.0
    return _round_cell(out)


def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    t0 = time.time()

    # Use a smaller FFT size for the fast solver (n must be power-of-2-ish, use 8)
    n = 8
    # Map the 32x32 cell onto an n x n coarse grid for the fast solver.
    def downsample(cell):
        block = size // n
        coarse = np.zeros((n, n), dtype=float)
        for ii in range(n):
            for jj in range(n):
                s = 0.0
                for di in range(block):
                    for dj in range(block):
                        s += cell[ii * block + di][jj * block + dj]
                coarse[ii, jj] = s / (block * block)
        return coarse

    # Initial guess: a ring / cross pattern (different from the seed rod).
    def initial_cell():
        c = [[0] * size for _ in range(size)]
        cx = cy = (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                dx = i - cx
                dy = j - cy
                r = math.hypot(dx, dy)
                # A dielectric ring (annulus) plus a central dot
                if 6.0 < r < 11.0:
                    c[i][j] = 1
                elif r < 2.5:
                    c[i][j] = 1
        return c

    cell = initial_cell()
    cell = _enforce_sym(cell)
    k_points = _k_points_irb(6)

    # Precompute coarse for fast eval
    def fast_score(cell):
        coarse = downsample(cell)
        eps_map = coarse * (eps_high - eps_low) + eps_low
        F = _eps_fourier(eps_map, n)
        min5 = float('inf')
        max4 = float('-inf')
        for (kx, ky) in k_points:
            w = _band_solver(F, kx, ky, eps_low, eps_high)
            if len(w) < lower_band + 1:
                continue
            o4 = w[lower_band - 1]
            o5 = w[lower_band]
            if o5 < min5:
                min5 = o5
            if o4 > max4:
                max4 = o4
        if min5 == float('inf') or max4 == float('-inf'):
            return -1.0
        gap = min5 - max4
        mid = 0.5 * (min5 + max4)
        if mid <= 0:
            return -1.0
        return 2.0 * gap / mid

    best_score = fast_score(cell)
    best_cell = [row[:] for row in cell]

    # Gradient-free local search on Fourier coefficients with C4v, then threshold.
    # Represent the coarse dielectric map by a few C4v Fourier modes.
    # We optimise a small set of real amplitudes for symmetric modes.
    # Build basis of C4v-symmetric Fourier modes (low frequency).
    modes = []
    # (dx, dy) frequency pairs; enforce C4v by symmetrising.
    freq_pairs = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2),
                  (2, 2), (3, 0), (0, 3), (3, 1), (1, 3), (3, 2), (2, 3), (3, 3),
                  (4, 0), (0, 4), (4, 1), (1, 4), (4, 2), (2, 4), (4, 3), (3, 4),
                  (4, 4)]
    # Keep only independent C4v orbits; build basis vectors in the n x n pixel space.
    basis = []
    for (fx, fy) in freq_pairs:
        v = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                v[i, j] = math.cos(2 * math.pi * (fx * i + fy * j) / n)
        # C4v symmetrise
        ops = [
            lambda i, j: (i, j),
            lambda i, j: (j, n - 1 - i),
            lambda i, j: (n - 1 - i, n - 1 - j),
            lambda i, j: (n - 1 - j, i),
            lambda i, j: (i, n - 1 - j),
            lambda i, j: (n - 1 - i, j),
            lambda i, j: (j, i),
            lambda i, j: (n - 1 - j, n - 1 - i),
        ]
        vs = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                s = 0.0
                for op in ops:
                    ii, jj = op(i, j)
                    s += v[ii, jj]
                vs[i, j] = s / 8.0
        # Normalise
        nn = np.linalg.norm(vs)
        if nn < 1e-9:
            continue
        vs = vs / nn
        # Check linear independence
        indep = True
        for b in basis:
            if abs(np.dot(vs.ravel(), b.ravel())) > 0.95:
                indep = False
                break
        if indep:
            basis.append(vs)
    basis = np.array(basis)
    nb = basis.shape[0]

    # Initial coefficients from current coarse map
    coarse0 = downsample(best_cell)
    mean0 = coarse0.mean()
    resid0 = coarse0 - mean0
    coeffs = basis @ resid0.ravel()
    coeffs = coeffs.tolist()

    def coeffs_to_cell(coeffs):
        coarse = mean0 + (basis @ np.array(coeffs)).reshape(n, n)
        # Upsample to size x size
        block = size // n
        cell = np.zeros((size, size), dtype=float)
        for ii in range(n):
            for jj in range(n):
                cell[ii * block:(ii + 1) * block, jj * block:(jj + 1) * block] = coarse[ii, jj]
        # C4v symmetrise
        ops = [
            lambda i, j: (i, j),
            lambda i, j: (j, size - 1 - i),
            lambda i, j: (size - 1 - i, size - 1 - j),
            lambda i, j: (size - 1 - j, i),
            lambda i, j: (i, size - 1 - j),
            lambda i, j: (size - 1 - i, j),
            lambda i, j: (j, i),
            lambda i, j: (size - 1 - j, size - 1 - i),
        ]
        out = np.zeros((size, size), dtype=float)
        for i in range(size):
            for j in range(size):
                s = 0.0
                for op in ops:
                    ii, jj = op(i, j)
                    s += cell[ii, jj]
                out[i, j] = s / 8.0
        return _round_cell(out)

    # Hill-climbing with random restarts in coefficient space.
    rng = np.random.default_rng(12345)
    step = 0.3
    iters = 0
    while time.time() - t0 < _DEADLINE:
        iters += 1
        # Try a random perturbation
        for _ in range(4):
            trial = [c + rng.normal(0, step) for c in coeffs]
            cell_t = coeffs_to_cell(trial)
            sc = fast_score(cell_t)
            if sc > best_score:
                best_score = sc
                best_cell = cell_t
                coeffs = trial
                break
        # Occasionally shrink step
        if iters % 50 == 0:
            step *= 0.9
            if step < 0.02:
                step = 0.3
                # restart
                coeffs = (basis @ (downsample(best_cell) - mean0).ravel()).tolist()
                step = 0.3

    # Final: enforce exact C4v and return
    final_cell = _enforce_sym(best_cell)
    return final_cell
