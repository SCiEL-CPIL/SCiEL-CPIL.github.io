"""Photonic-crystal unit cell designer: large TM 4-5 band gap via surrogate-guided search."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# ---------- Symmetry: cell == reversed rows/cols == transpose ----------
# We parameterize the 16x16 top-left quadrant (indices 0..15,0..15) and mirror.
# Mirror rule: cell[i][j] = cell[size-1-i][size-1-j] = cell[j][i].
# So the fundamental region is the quadrant with i<=j (the upper triangle of TL quadrant).

def _expand_quadrant(q, size=32):
    """q is a 16x16 array (TL quadrant). Expand to full 32x32 with 4-fold + reflections."""
    n = size // 2
    # full = mirror of q
    # cell[i][j] for i in [0,n), j in [0,n) = q[i][j]
    # cell[i][j] for i in [0,n), j in [n,size) = q[i][size-1-j]
    # cell[i][j] for i in [n,size), j in [0,n) = q[size-1-i][j]
    # cell[i][j] for i in [n,size), j in [n,size) = q[size-1-i][size-1-j]
    cell = np.zeros((size, size), dtype=np.int8)
    cell[0:n, 0:n] = q[0:n, 0:n]
    cell[0:n, n:size] = q[0:n, 0:n][:, ::-1]
    cell[n:size, 0:n] = q[0:n, 0:n][::-1, :]
    cell[n:size, n:size] = q[0:n, 0:n][::-1, ::-1]
    return cell


def _params_to_quadrant(p):
    """p = [r0, r1, cw, ch, cc, cr, c0, c1] continuous -> 16x16 quadrant (0/1).
    - r0: inner ring radius, r1: outer ring radius (ring dielectric)
    - cw: cross half-width (dielectric cross along axes)
    - ch: cross length (how far cross extends)
    - cc: corner radius (dielectric corner blobs)
    - cr: corner center radius (distance from corner)
    - c0: center radius (dielectric center dot)
    - c1: center hole radius (air hole in center)
    All in pixel units of the 16x16 quadrant.
    """
    n = 16
    r0, r1, cw, ch, cc, cr, c0, c1 = p
    q = np.zeros((n, n), dtype=np.int8)
    # pixel centers in quadrant coordinates: (i+0.5, j+0.5), i,j in 0..15
    # center of full cell is at (16,16) in quadrant-mirror coords -> (8,8) in quadrant
    cx = cy = 8.0  # center of quadrant (maps to cell center)
    for i in range(n):
        for j in range(n):
            x = (i + 0.5) - cx
            y = (j + 0.5) - cy
            r = math.hypot(x, y)
            val = 0
            # ring
            if r0 <= r <= r1:
                val = 1
            # cross along x and y axes
            if abs(y) <= cw and abs(x) <= ch:
                val = 1
            if abs(x) <= cw and abs(y) <= ch:
                val = 1
            # corner blobs (near the quadrant corner, i.e. x,y both negative large)
            # corner of full cell in quadrant is at (0,0) -> x=-8, y=-8
            rx = (i + 0.5) - 0.0
            ry = (j + 0.5) - 0.0
            rc = math.hypot(rx, ry)
            if rc <= cc:
                val = 1
            # center dot
            if r <= c0:
                val = 1
            # center air hole
            if r <= c1:
                val = 0
            q[i, j] = val
    return q


# ---------- Surrogate band solver (coarse PW) ----------
def _fourier_eps(cell, N):
    """Exact Fourier coefficients of the pixel map (binarized, 1=eps_high, 0=eps_low).
    Returns eps_g for g in the N x N grid of reciprocal vectors, indexed [a][b].
    cell is size x size of 0/1. eps_low=1, eps_high=12.
    We compute 1/eps for TM (since TM uses 1/eps)."""
    size = cell.shape[0]
    eps = np.where(cell.astype(np.float64) > 0.5, 12.0, 1.0)
    inv_eps = 1.0 / eps
    # Fourier: eps_g = (1/size^2) sum_{i,j} eps_{i,j} exp(-2pi i (g_i i + g_j j)/size)
    # We only need real part for the Hermitian matrix; use rfft2 for speed.
    # But we need specific g values in the N grid. Use direct DFT for the needed g.
    # inv_eps as 2D array; compute 2D FFT.
    F = np.fft.fft2(inv_eps) / (size * size)
    # F[a][b] corresponds to g = (a, b) with a in [-size/2, size/2)
    # We need g values in range [-N//2, N//2-1]
    def get_F(a, b):
        # map a,b to fft indices
        aa = a % size
        bb = b % size
        return F[aa, bb]
    return get_F, size


def _band_gaps(cell, N, kpoints):
    """Compute TM bands using plane-wave Galerkin. Returns (min_omega5, max_omega4) over kpoints.
    TM: eigenvalue problem with 1/eps. H = sum_g'  (1/eps)_{g-g'} |k+g'><k+g'|
    For TM, the operator is (1/eps) * (k+g)^2 in a sense. Specifically:
    (1/eps) (k+g)^2 coupling. The matrix M_{g,g'} = (1/eps)_{g-g'} (k+g)·(k+g')
    Actually for TM: (1/eps) ∇·(∇×H) -> the eigenvalue is (1/eps)_{g-g'} (k+g)·(k+g').
    Let's use the standard form: for TM, A_{g,g'} = (1/eps)_{g-g'} (k+g)·(k+g').
    We want eigenvalues; omega^2 = lambda.
    """
    get_F, size = _fourier_eps(cell, N)
    # g grid
    half = N // 2
    gs = list(range(-half, half))
    glist = [(a, b) for a in gs for b in gs]
    Ng = len(glist)
    gidx = {g: i for i, g in enumerate(glist)}
    # diff grid for (1/eps)_{g-g'}
    maxdiff = N - 1
    # Precompute (1/eps)_{dx,dy} for dx,dy in [-maxdiff, maxdiff]
    def inv_eps_diff(dx, dy):
        return get_F(dx, dy).real  # Hermitian: use real part for symmetric
    # Actually (1/eps)_{g-g'} where g-g' can be any diff in [-2*half, 2*half]
    # get_F handles mod size, so fine.

    max_w4 = 0.0
    min_w5 = float('inf')
    for kx, ky in kpoints:
        # Build matrix
        M = np.zeros((Ng, Ng), dtype=np.float64)
        for ii, g in enumerate(glist):
            kgx = kx + g[0]
            kgy = ky + g[1]
            kg2 = kgx * kgx + kgy * kgy
            for jj, gp in enumerate(glist):
                kpgx = kx + gp[0]
                kpgy = ky + gp[1]
                dx = g[0] - gp[0]
                dy = g[1] - gp[1]
                e = inv_eps_diff(dx, dy)
                M[ii, jj] = e * (kgx * kpgx + kgy * kpgy)
        # Symmetrize
        M = 0.5 * (M + M.T)
        # Eigenvalues
        try:
            w = np.linalg.eigvalsh(M)
        except np.linalg.LinAlgError:
            continue
        # w sorted ascending. Bands 1..N. We want band 4 (index 3) and band 5 (index 4).
        if len(w) >= 5:
            w4 = w[3]
            w5 = w[4]
            max_w4 = max(max_w4, w4)
            min_w5 = min(min_w5, w5)
    if min_w5 == float('inf'):
        return 0.0
    gap = 2.0 * (min_w5 - max_w4)
    mid = min_w5 + max_w4
    if mid <= 0:
        return 0.0
    return gap / mid


# ---------- k-points in irreducible BZ ----------
def _irreducible_kpoints(nk=6):
    """Irreducible BZ of square lattice: triangle with vertices (0,0),(pi,0),(pi/2,pi/2).
    Boundary: X=(pi,0) [or (0.5,0) in units of 2pi?]. Let's use units where BZ is [-0.5,0.5]^2.
    Irreducible: 0<=ky<=kx, kx+ky<=0.5. Vertices: (0,0), (0.5,0), (0.25,0.25).
    """
    kps = []
    # along X: (t, 0) t in [0, 0.5]
    for t in np.linspace(0, 0.5, nk):
        kps.append((t, 0.0))
    # along M-X: (0.5, t) -> but in irreducible it's (0.25+t, 0.25-t)? 
    # M=(0.25,0.25), X=(0.5,0). Line M-X: (0.5 - s*0.25, 0 + s*0.25) for s in [0,1]? 
    # Actually from X(0.5,0) to M(0.25,0.25): param (0.5 - 0.25s, 0.25s)
    for s in np.linspace(0, 1, nk):
        kps.append((0.5 - 0.25 * s, 0.25 * s))
    # along M-Gamma: (0.25(1-s), 0.25(1-s))
    for s in np.linspace(0, 1, nk):
        kps.append((0.25 * (1 - s), 0.25 * (1 - s)))
    return kps


# ---------- Objective (surrogate) ----------
def _surrogate_score(p, N, kpoints):
    q = _params_to_quadrant(p)
    cell = _expand_quadrant(q)
    return _band_gaps(cell, N, kpoints)


def design(problem):
    t_start = time.time()
    deadline = t_start + 200.0  # safety margin before 230s kill
    size = problem.get("pixels", 32)

    # Coarse surrogate settings
    N_coarse = 11
    kps_coarse = _irreducible_kpoints(nk=5)

    # Initial parameters
    # r0, r1, cw, ch, cc, cr, c0, c1
    p0 = np.array([3.0, 6.0, 1.5, 7.0, 3.0, 3.0, 2.0, 0.5])

    best_p = p0.copy()
    best_score = -1e9
    # Evaluate initial
    s0 = _surrogate_score(p0, N_coarse, kps_coarse)
    best_score = s0

    # Nelder-Mead on surrogate
    from scipy.optimize import minimize
    def neg_obj(p):
        return -_surrogate_score(p, N_coarse, kps_coarse)

    # Bounds: keep params in reasonable ranges
    bounds = [(0.5, 7.0), (2.0, 8.0), (0.5, 4.0), (4.0, 8.0), (1.0, 6.0), (1.0, 6.0), (0.5, 5.0), (0.0, 4.0)]

    # Try a few random starts
    rng = np.random.default_rng(42)
    for restart in range(4):
        if time.time() > deadline - 60.0:
            break
        if restart == 0:
            p_try = p0.copy()
        else:
            p_try = p0 + rng.uniform(-1.5, 1.5, size=8)
            p_try = np.clip(p_try, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            res = minimize(neg_obj, p_try, method='Nelder-Mead',
                           bounds=bounds,
                           options={'maxiter': 200, 'xatol': 0.05, 'fatol': 0.01})
            if res.fun < -best_score:
                best_score = -res.fun
                best_p = res.x
        except Exception:
            pass

    # High-resolution polish
    N_fine = 21
    kps_fine = _irreducible_kpoints(nk=8)

    # Local refinement with finer surrogate
    def neg_obj_fine(p):
        return -_band_gaps(_expand_quadrant(_params_to_quadrant(p)), N_fine, kps_fine)

    # Grid search around best_p with finer surrogate
    if time.time() < deadline - 40.0:
        step = 0.25
        for dim in range(8):
            if time.time() > deadline - 40.0:
                break
            for delta in [-2 * step, -step, step, 2 * step]:
                p_try = best_p.copy()
                p_try[dim] += delta
                p_try = np.clip(p_try, [b[0] for b in bounds], [b[1] for b in bounds])
                s = -neg_obj_fine(p_try)
                if s > best_score:
                    best_score = s
                    best_p = p_try

    # Final Nelder-Mead with fine surrogate
    if time.time() < deadline - 30.0:
        try:
            res = minimize(neg_obj_fine, best_p, method='Nelder-Mead',
                           bounds=bounds,
                           options={'maxiter': 100, 'xatol': 0.02, 'fatol': 0.005})
            if res.fun < -best_score:
                best_score = -res.fun
                best_p = res.x
        except Exception:
            pass

    # Build final cell
    q = _params_to_quadrant(best_p)
    cell = _expand_quadrant(q)

    # Verify symmetry
    cell = cell.astype(int)
    # Ensure exact symmetry (in case of any numerical issue)
    n = size // 2
    final = np.zeros((size, size), dtype=int)
    final[0:n, 0:n] = q[0:n, 0:n]
    final[0:n, n:size] = q[0:n, 0:n][:, ::-1]
    final[n:size, 0:n] = q[0:n, 0:n][::-1, :]
    final[n:size, n:size] = q[0:n, 0:n][::-1, ::-1]

    return final.tolist()
# EVOLVE-BLOCK-END
