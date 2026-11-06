import time
import numpy as np
from numpy.linalg import eigh

_SIZE = 32
_EPS_HIGH = 12.0
_EPS_LOW = 1.0
_EPS_DIFF = _EPS_HIGH - _EPS_LOW


def _eval_deadline():
    return 218.0  # safety margin vs 230s kill


def _make_d4_basis():
    """Build D4-invariant monomial basis (x^4 - 6x^2 y^2 + y^4)-style harmonics up to degree 8.
    Each basis function is symmetric under the full square symmetry group about the cell centre."""
    # (a, b) exponents with a+b <= 8, a+b even, and the term is D4-invariant
    # D4-invariant monomials: use combinations that are invariant under x->-x, y->-y, x<->y
    # We build from the 4 fundamental invariants:
    #   I1 = x^2 + y^2
    #   I2 = x^2 y^2
    #   I3 = x^2 - y^2  (changes sign under x<->y? No: (x^2-y^2) -> (y^2-x^2) = -I3, so NOT invariant alone)
    # For full D4 invariance we need even powers symmetric in x,y. Use:
    #   basis = {1, (x^2+y^2), (x^2+y^2)^2, x^2 y^2, (x^2+y^2)^3, (x^2+y^2)^2 x^2 y^2, (x^2+y^2)^3 x^2 y^2,
    #            (x^4 - 6x^2 y^2 + y^4) [degree4 harmonic], (x^6 - 15x^4 y^2 - 15 x^2 y^4 + y^6) [degree6],
    #            (x^8 - 28x^6 y^2 + 70 x^4 y^4 - 28 x^2 y^6 + y^8) [degree8]}
    # These are the real parts of (x+iy)^n projected to D4-invariant (n divisible by 4) harmonics
    # plus radial polynomials to give flexibility.
    # We'll just use a set of D4-invariant polynomials up to degree 8.
    return [
        (0, 0),            # 1
        (2, 0),            # x^2  (but we combine symmetrically below)
    ]


def _d4_poly_value(coeffs, x, y):
    """Evaluate D4-invariant polynomial field. coeffs length 9.
    Uses the 4 D4-invariant harmonic/radial building blocks:
      r2 = x^2+y^2
      h4 = x^4 - 6x^2 y^2 + y^4
      h8 = x^8 - 28 x^6 y^2 + 70 x^4 y^4 - 28 x^2 y^6 + y^8
      r4 = r2^2, r6 = r2^3, r8 = r2^4
      x2y2 = x^2 y^2
    """
    x2 = x * x
    y2 = y * y
    r2 = x2 + y2
    x2y2 = x2 * y2
    h4 = x2 * x2 - 6.0 * x2y2 + y2 * y2
    r4 = r2 * r2
    r6 = r4 * r2
    r8 = r4 * r4
    h8 = x2 * x2 * x2 * x2 - 28.0 * x2 * x2 * x2 * y2 + 70.0 * x2y2 * x2y2 \
         - 28.0 * x2 * y2 * y2 * y2 + y2 * y2 * y2 * y2
    c = coeffs
    return (c[0] + c[1] * r2 + c[2] * x2y2 + c[3] * h4 + c[4] * r4
            + c[5] * r6 + c[6] * r8 + c[7] * h8 + c[8] * r4 * x2y2)


def _coeffs_to_cell(coeffs, thresh=0.0):
    size = _SIZE
    xs = (np.arange(size) + 0.5) / size - 0.5
    ys = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    field = _d4_poly_value(coeffs, X, Y)
    cell = (field > thresh).astype(int)
    # Enforce exact D4 symmetry (the polynomial is D4-invariant analytically, but
    # numerical ties at 0 can break it; snap to symmetric majority).
    c = cell
    c = np.maximum(c, c.T)
    c = np.maximum(c, c[::-1, :])
    c = np.maximum(c, c[:, ::-1])
    return c.tolist()


def _inv_eps_mat(cell):
    size = _SIZE
    eps = _EPS_LOW + _EPS_DIFF * np.array(cell, dtype=float)
    return 1.0 / eps


def _build_pw_basis(radius):
    G_list = []
    for Gx in range(-radius, radius + 1):
        for Gy in range(-radius, radius + 1):
            if Gx * Gx + Gy * Gy <= radius * radius:
                G_list.append((Gx, Gy))
    G_arr = np.array(G_list, dtype=float)
    return G_arr


def _fourier_inv_eps(cell, radius):
    """Compute Fourier coefficients of inv-eps on a (2*radius+1)^2 grid of G-vectors,
    using the exact pixel map (DFT of the pixel array)."""
    size = _SIZE
    inv_eps = _inv_eps_mat(cell)
    # inv_eps is periodic with period `size`. Fourier coeff at integer (Gx,Gy):
    # eps_G = (1/N^2) sum_{i,j} inv_eps[i,j] exp(-2 pi i (Gx i + Gy j)/N)
    # We only need G in [-radius, radius]. Compute via fft2 on the size x size grid.
    fft_inv = np.fft.fft2(inv_eps) / (size * size)
    # fft2 gives coeff at indices [0..size-1] corresponding to G in [0..size-1] (wrapped).
    # For a negative G, index = G % size.
    def get(Gx, Gy):
        return fft_inv[Gx % size, Gy % size]
    G_list = []
    for Gx in range(-radius, radius + 1):
        for Gy in range(-radius, radius + 1):
            if Gx * Gx + Gy * Gy <= radius * radius:
                G_list.append((Gx, Gy))
    G_arr = np.array(G_list, dtype=float)
    n = len(G_list)
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    size_d = 4 * radius + 1
    off = 2 * radius
    inv_eps_full = np.zeros((size_d, size_d), dtype=complex)
    for i in range(n):
        for j in range(n):
            dqx = int(Gx_diff[i, j])
            dqy = int(Gy_diff[i, j])
            inv_eps_full[dqx + off, dqy + off] = get(dqx, dqy)
    Gx_diff_i = Gx_diff.astype(int)
    Gy_diff_i = Gy_diff.astype(int)
    inv_eps_mat = inv_eps_full[Gx_diff_i + off, Gy_diff_i + off]
    return G_arr, inv_eps_mat


def _k_points(n_k):
    # Irreducible BZ boundary + interior for square lattice (in units of pi/a).
    # We use the full set of representative points.
    pts = [
        (0.0, 0.0),
        (0.25, 0.0), (0.5, 0.0),
        (0.5, 0.25), (0.5, 0.5),
        (0.35, 0.35), (0.4, 0.3), (0.3, 0.2),
        (0.2, 0.1), (0.45, 0.15), (0.15, 0.4),
        (0.4, 0.4), (0.25, 0.15), (0.3, 0.3),
        (0.1, 0.0), (0.2, 0.0), (0.35, 0.0), (0.45, 0.0),
        (0.5, 0.1), (0.5, 0.3), (0.5, 0.4),
        (0.3, 0.1), (0.4, 0.2), (0.2, 0.3),
        (0.15, 0.25), (0.25, 0.35), (0.35, 0.25),
        (0.45, 0.35), (0.2, 0.4), (0.4, 0.45),
    ]
    return pts[:n_k]


def _compute_gap(cell, G_arr, inv_eps_mat, k_pts, pw_radius):
    """Compute gap-midgap ratio for bands 4,5 (indices 3,4) using TM plane-wave method.
    TM: eigenvalue problem M a = omega^2 a where M_{GG'} = (k+G).(k+G') * inv_eps_{G-G'}.
    """
    n_pw = len(G_arr)
    max_w4 = 0.0
    min_w5 = float('inf')
    for k in k_pts:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T
        M = dots * inv_eps_mat
        M = (M + M.conj().T) / 2.0
        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0.0))
            if len(eigvals) >= 5:
                w4 = eigvals[3]
                w5 = eigvals[4]
                if w4 > max_w4:
                    max_w4 = w4
                if w5 < min_w5:
                    min_w5 = w5
        except Exception:
            pass
    if min_w5 == float('inf') or max_w4 <= 0.0:
        return -1.0
    return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)


def _gap_from_coeffs(coeffs, G_arr_cache, k_pts, pw_radius, thresh=0.0):
    cell = _coeffs_to_cell(coeffs, thresh)
    G_arr, inv_eps_mat = _fourier_inv_eps(cell, pw_radius)
    return _compute_gap(cell, G_arr, inv_eps_mat, k_pts, pw_radius)


def _nelder_mead(f, x0, max_iter=200, tol=1e-4, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
    """Simple Nelder-Mead for a small number of variables."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        xi = list(x0)
        xi[i] = x0[i] + 0.1 * (1.0 + abs(x0[i]))
        simplex.append(xi)
    fvals = [f(s) for s in simplex]
    for it in range(max_iter):
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
        fr = f(xr)
        if fr < fvals[0]:
            xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            xc = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(n)]
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    fvals[i] = f(simplex[i])
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]


def _field_driven_refine(cell, G_arr, inv_eps_mat, k_pts, pw_radius, n_iters=3):
    """Topological pixel flips guided by band-4/band-5 field intensities."""
    size = _SIZE
    c = np.array(cell, dtype=float)
    xs = (np.arange(size) + 0.5) / size - 0.5
    ys = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    best_cell = cell
    best_score = _compute_gap(cell, G_arr, inv_eps_mat, k_pts, pw_radius)
    for _ in range(n_iters):
        field4 = np.zeros((size, size))
        field5 = np.zeros((size, size))
        for k in k_pts[:4]:
            kvec = np.array(k, dtype=float)
            kG = kvec + G_arr
            dots = kG @ kG.T
            M = dots * inv_eps_mat
            M = (M + M.conj().T) / 2.0
            try:
                eigvals, eigvecs = eigh(M)
                eigvals = np.sqrt(np.maximum(eigvals, 0.0))
                if len(eigvals) >= 5:
                    v4 = eigvecs[:, 3]
                    v5 = eigvecs[:, 4]
                    # Reconstruct field in real space
                    for idx, (Gx, Gy) in enumerate(G_arr):
                        phase4 = np.exp(2j * np.pi * ((k[0] + Gx) * X + (k[1] + Gy) * Y))
                        phase5 = np.exp(2j * np.pi * ((k[0] + Gx) * X + (k[1] + Gy) * Y))
                        field4 += np.abs(v4[idx] * phase4) ** 2
                        field5 += np.abs(v5[idx] * phase5) ** 2
            except Exception:
                pass
        if field4.max() > 0:
            field4 /= field4.max()
        if field5.max() > 0:
            field5 /= field5.max()
        score_map = field4 - field5
        c_new = c.copy()
        for i in range(size // 2):
            for j in range(size // 2):
                avg = (score_map[i, j] + score_map[size - 1 - i, j] +
                       score_map[i, size - 1 - j] + score_map[size - 1 - i, size - 1 - j]) / 4.0
                if avg > 0.15:
                    for (a, b) in [(i, j), (size - 1 - i, j), (i, size - 1 - j), (size - 1 - i, size - 1 - j)]:
                        c_new[a, b] = 1.0
                elif avg < -0.15:
                    for (a, b) in [(i, j), (size - 1 - i, j), (i, size - 1 - j), (size - 1 - i, size - 1 - j)]:
                        c_new[a, b] = 0.0
        new_cell = (c_new > 0.5).astype(int)
        new_cell = np.maximum(new_cell, new_cell.T)
        new_cell = np.maximum(new_cell, new_cell[::-1, :])
        new_cell = np.maximum(new_cell, new_cell[:, ::-1])
        new_cell = new_cell.tolist()
        G_arr2, inv_eps_mat2 = _fourier_inv_eps(new_cell, pw_radius)
        s = _compute_gap(new_cell, G_arr2, inv_eps_mat2, k_pts, pw_radius)
        if s > best_score:
            best_score = s
            best_cell = new_cell
            c = np.array(new_cell, dtype=float)
            G_arr = G_arr2
            inv_eps_mat = inv_eps_mat2
        else:
            break
    return best_cell, best_score


def design(problem):
    t0 = time.time()
    deadline = _eval_deadline()

    def remaining():
        return deadline - time.time()

    size = problem["pixels"]
    assert size == 32

    # Use a moderate plane-wave radius for the fast proxy.
    pw_radius = 6
    k_pts = _k_points(10)

    # Precompute a G-basis for a reference cell (G-list depends only on radius).
    ref_cell = _coeffs_to_cell([0.0] * 9, 0.0)
    G_arr, _ = _fourier_inv_eps(ref_cell, pw_radius)

    best_cell = ref_cell
    best_score = -1.0

    # ---- Phase 1: Nelder-Mead over continuous D4 polynomial coefficients ----
    # Good starting coefficient sets (hand-crafted to produce diverse shapes).
    starts = [
        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # central rod (r2 < c)
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # ring
        [0.0, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # cross-ish (h4)
        [0.0, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # square-ish (x2y2)
        [0.0, -0.3, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.5, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],   # h8 shape
        [0.0, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
    ]

    def objective(coeffs):
        if time.time() > deadline - 25:
            return 1e9
        try:
            s = _gap_from_coeffs(coeffs, None, k_pts, pw_radius, 0.0)
            return -s  # minimize negative gap
        except Exception:
            return 1e9

    for s0 in starts:
        if remaining() < 40:
            break
        try:
            x_opt, f_opt = _nelder_mead(objective, s0, max_iter=60, tol=1e-3)
        except Exception:
            continue
        if f_opt >= 1e8:
            continue
        cell = _coeffs_to_cell(x_opt, 0.0)
        G_arr2, inv_eps_mat2 = _fourier_inv_eps(cell, pw_radius)
        s = _compute_gap(cell, G_arr2, inv_eps_mat2, k_pts, pw_radius)
        if s > best_score:
            best_score = s
            best_cell = cell

    # ---- Phase 2: topological field-driven refinement ----
    if best_cell is not None and remaining() > 30:
        try:
            G_arr_b, inv_eps_mat_b = _fourier_inv_eps(best_cell, pw_radius)
            refined, s_ref = _field_driven_refine(
                best_cell, G_arr_b, inv_eps_mat_b, k_pts, pw_radius, n_iters=4)
            if s_ref > best_score:
                best_score = s_ref
                best_cell = refined
        except Exception:
            pass

    # ---- Phase 3: a few random-coefficient restarts if time remains ----
    if remaining() > 25:
        rng = np.random.default_rng(12345)
        for _ in range(4):
            if remaining() < 20:
                break
            s0 = list(rng.normal(0.0, 0.4, size=9))
            try:
                x_opt, f_opt = _nelder_mead(objective, s0, max_iter=40, tol=1e-3)
                if f_opt >= 1e8:
                    continue
                cell = _coeffs_to_cell(x_opt, 0.0)
                G_arr2, inv_eps_mat2 = _fourier_inv_eps(cell, pw_radius)
                s = _compute_gap(cell, G_arr2, inv_eps_mat2, k_pts, pw_radius)
                if s > best_score:
                    best_score = s
                    best_cell = cell
            except Exception:
                pass

    if best_cell is None:
        best_cell = _coeffs_to_cell([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0)

    return best_cell
