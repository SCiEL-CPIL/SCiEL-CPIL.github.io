"""Photonic-crystal unit cell design: maximize TM band-4/5 gap via local search."""
import time
import math
import numpy as np

_DEADLINE = time.time() + 110.0  # safety margin before 120s sandbox / 230s wall


def _time_left():
    return _DEADLINE - time.time()


def _symmetrize(cell):
    """Enforce square-symmetry about the centre (reverse rows/cols + transpose)."""
    cell = np.array(cell, dtype=np.float64)
    cell = (cell + cell[::-1, :]) * 0.5
    cell = (cell + cell[:, ::-1]) * 0.5
    cell = (cell + cell.T) * 0.5
    return np.round(cell)


def _build_epsilon(cell, eps_low, eps_high):
    """Return the dielectric array (float) on the pixel grid."""
    cell = np.asarray(cell, dtype=np.float64)
    return eps_low + (eps_high - eps_low) * cell


def _fourier_coefficients(eps, N, nmax):
    """Fourier coefficients eps_G for G indices in [-nmax, nmax] (2D square lattice).

    eps(x) = sum_G eps_G exp(i G . r).  For a pixel grid with cell size 1,
    pixel (i,j) occupies [i/N,(i+1)/N) x [j/N,(j+1)/N).
    eps_G = N^2 sum_{i,j} eps[i,j] * (1/N^2) * integral over pixel of exp(-i G.r)
          = sum_{i,j} eps[i,j] * (1/N^2) * I_i(Gx) * I_j(Gy)
    where I_i(g) = integral_{i/N}^{(i+1)/N} exp(-i g x) dx.
    """
    Gx = np.arange(-nmax, nmax + 1) * 2.0 * math.pi
    # integral of exp(-i g x) over [a,b]
    def integ(g, a, b):
        if abs(g) < 1e-12:
            return b - a
        return (math.exp(-1j * g * b) - math.exp(-1j * g * a)) / (-1j * g)

    # Precompute per-pixel integrals for each Gx
    # I_i(g) for pixel i in [i/N,(i+1)/N]
    Nf = float(N)
    # Build matrix M[i, k] = integral over pixel i for Gx[k]
    Mx = np.empty((N, len(Gx)), dtype=np.complex128)
    for k, g in enumerate(Gx):
        if abs(g) < 1e-12:
            Mx[:, k] = 1.0 / Nf
        else:
            a = np.arange(N) / Nf
            b = (np.arange(N) + 1) / Nf
            Mx[:, k] = (np.exp(-1j * g * b) - np.exp(-1j * g * a)) / (-1j * g)
    My = Mx  # same grid, square
    # eps_G[kx, ky] = sum_{i,j} eps[i,j] Mx[i,kx] My[j,ky] / N^2
    # = (1/N^2) * Mx^T @ eps @ My
    coeff = (Mx.T @ eps @ My) / (Nf * Nf)
    return coeff  # shape (len(Gx), len(Gx))


def _plane_wave_grid(nmax):
    """Return list of (gx, gy) indices and the G vectors (in units of 2pi)."""
    idx = []
    for a in range(-nmax, nmax + 1):
        for b in range(-nmax, nmax + 1):
            idx.append((a, b))
    return idx


def _tm_bands(cell, problem, nmax, kpts):
    """Compute lowest bands (omega) at each k-point for TM polarization.

    TM: solve  (1/eps_G) * (|k+G|^2) coupling.  Eigenvalue = omega^2.
    Returns array of shape (len(kpts), nmax*2+1) with sorted omega (sqrt of eigenvalue).
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    eps = _build_epsilon(cell, eps_low, eps_high)
    coeff = _fourier_coefficients(eps, size, nmax)
    inv_coeff = 1.0 / coeff  # 1/eps_G
    idx = _plane_wave_grid(nmax)
    n = len(idx)
    idx_arr = np.array(idx, dtype=np.int64)  # (n,2)

    # Precompute |G| for each plane wave
    Gvals = idx_arr * 2.0 * math.pi
    Gmag2 = np.sum(Gvals ** 2, axis=1)

    # Build the coupling matrix A (n x n): A[p,q] = inv_coeff[gp,gq] * |k+Gq|^2
    # For TM: (1/eps_{Gp-Gq}) |k+Gq|^2  U_q = omega^2 U_p
    # Actually the standard form: sum_q (1/eps_{Gp-Gq}) |k+Gq|^2 U_q = omega^2 U_p
    # The matrix is symmetric? 1/eps is Hermitian (coeff is Hermitian), |k+Gq|^2 depends on q.
    # M[p,q] = inv_coeff[p-q] * |k+Gq|^2.  This is not symmetric but the eigenvalue problem
    # is self-adjoint in the appropriate inner product. We'll just use the matrix as is.

    results = np.empty((len(kpts), n), dtype=np.float64)
    for ki, k in enumerate(kpts):
        kx, ky = k
        # |k + Gq|^2 for each q
        kvec = np.array([kx, ky]) * 2.0 * math.pi
        kG = kvec[None, :] + Gvals
        kGmag2 = np.sum(kG ** 2, axis=1)
        # Build matrix M[p,q] = inv_coeff[p-q] * kGmag2[q]
        # inv_coeff is indexed by (pa-pb, qa-qb) where indices offset by nmax
        # diff index: p - q in range [-n, n]; offset by nmax
        # Build diff matrix D[p,q] = p - q  (in idx index space)
        # We need inv_coeff at (idx_p - idx_q) in G-index space = (pa-pb, qa-qb)
        # coeff array is indexed [a+bidx, c+didx] where bidx = a+nmax
        # So for difference (da, dc) we use coeff[da+nmax, dc+nmax]
        # Compute da = idx_arr[p,0]-idx_arr[q,0], dc = idx_arr[p,1]-idx_arr[q,1]
        # Vectorize: da_matrix = idx_arr[:,0][:,None] - idx_arr[:,0][None,:]
        da = idx_arr[:, 0][:, None] - idx_arr[:, 0][None, :]
        dc = idx_arr[:, 1][:, None] - idx_arr[:, 1][None, :]
        # gather inv_coeff[da+nmax, dc+nmax]
        invM = inv_coeff[da + nmax, dc + nmax]
        M = invM * kGmag2[None, :]
        # Solve eigenvalue problem. M is not symmetric; use eigh only if symmetric.
        # Check: is M symmetric? M[p,q] vs M[q,p]: inv_coeff[p-q]*|k+Gq|^2 vs inv_coeff[q-p]*|k+Gp|^2.
        # inv_coeff[p-q] = conj(inv_coeff[q-p]) (Hermitian). |k+Gq|^2 != |k+Gp|^2 generally.
        # So not symmetric. Use eigvals.
        try:
            w = np.linalg.eigvalsh((M + M.T) * 0.5)  # symmetrize as a heuristic
        except Exception:
            w = np.linalg.eigvals(M)
            w = np.real(w)
        w = np.sort(w.real)
        # omega = sqrt(omega^2)
        w = np.sqrt(np.clip(w, 0, None))
        results[ki, :] = w
    return results


def _irreducible_kpts(nk=30):
    """30 wave vectors over the irreducible BZ (Gamma-X-M boundary + interior).

    Irreducible wedge: 0 <= ky <= kx, kx in [0, 0.5] (in units of 1, i.e. G=2pi).
    We'll use normalized coordinates where the BZ is [-0.5,0.5]^2 and the wedge is
    0<=ky<=kx<=0.5.
    """
    kpts = []
    # Boundary Gamma-X: (t, 0) for t in [0,0.5]
    for t in np.linspace(0.0, 0.5, 10):
        kpts.append((t, 0.0))
    # Boundary X-M: (0.5, t) for t in [0,0.5]
    for t in np.linspace(0.0, 0.5, 10):
        kpts.append((0.5, t))
    # Boundary M-Gamma: (t, t) for t in [0,0.5]
    for t in np.linspace(0.0, 0.5, 10):
        kpts.append((t, t))
    # Deduplicate and ensure 30
    seen = set()
    uniq = []
    for k in kpts:
        key = (round(k[0], 6), round(k[1], 6))
        if key not in seen:
            seen.add(key)
            uniq.append(k)
    return uniq[:30]


def _gap_ratio(bands, lower_band=4):
    """Compute gap-midgap ratio between band lower_band and lower_band+1.

    bands: (nk, nbands) omega values.  min omega_{b+1} and max omega_b across k.
    """
    b = lower_band  # 1-based band index
    omega_b = bands[:, b - 1]
    omega_b1 = bands[:, b]
    max_b = np.max(omega_b)
    min_b1 = np.min(omega_b1)
    gap = min_b1 - max_b
    mid = min_b1 + max_b
    if mid <= 0:
        return -1.0
    return 2.0 * gap / mid


def _make_square_pillar(half_side, size=32):
    """A centred square pillar of half-side (in pixels)."""
    cell = np.zeros((size, size))
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            if abs(i - center) < half_side and abs(j - center) < half_side:
                cell[i, j] = 1.0
    return cell


def _make_cross_pillar(arm_half, size=32):
    """A centred cross: horizontal and vertical arms of half-width arm_half."""
    cell = np.zeros((size, size))
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            if abs(i - center) < arm_half or abs(j - center) < arm_half:
                cell[i, j] = 1.0
    return cell


def _make_ring(inner, outer, size=32):
    """A centred square ring: dielectric where inner <= |i-c| or |j-c| < outer (L-shape ring)."""
    cell = np.zeros((size, size))
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            dx = abs(i - center)
            dy = abs(j - center)
            # ring: max(dx,dy) in [inner, outer)
            if inner <= max(dx, dy) < outer:
                cell[i, j] = 1.0
    return cell


def _evaluate(cell, problem, nmax, kpts):
    """Return gap ratio for a given cell."""
    cell = _symmetrize(cell)
    bands = _tm_bands(cell, problem, nmax, kpts)
    return _gap_ratio(bands, problem.get("lower_band", 4))


def design(problem):
    """Search for a symmetric 32x32 unit cell maximizing the TM band-4/5 gap."""
    size = problem.get("pixels", 32)
    lower_band = problem.get("lower_band", 4)

    # Use a modest plane-wave count for speed during search
    nmax = 5  # 11x11 = 121 plane waves; fast enough
    kpts = _irreducible_kpts(30)

    # Initial candidates: square pillars of various sizes, crosses, rings
    candidates = []
    # Square pillars: half-side from 2 to 10 pixels
    for hs in [2, 3, 4, 5, 6, 7, 8]:
        candidates.append(_make_square_pillar(hs, size))
    # Crosses
    for ah in [2, 3, 4, 5]:
        candidates.append(_make_cross_pillar(ah, size))
    # Rings
    for inner, outer in [(2, 6), (3, 8), (4, 10), (2, 8)]:
        candidates.append(_make_ring(inner, outer, size))

    # Evaluate all candidates
    best_cell = None
    best_score = -1e9
    for cell in candidates:
        if _time_left() < 5.0:
            break
        try:
            score = _evaluate(cell, problem, nmax, kpts)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_cell = np.array(cell, dtype=np.float64)

    if best_cell is None:
        # Fallback: square pillar
        best_cell = _make_square_pillar(5, size)
        best_score = 0.0

    # Local refinement: perturb the best cell with symmetric flips
    # We'll do a simple hill-climbing / random search over pixel flips that preserve symmetry.
    # Symmetry orbits: a pixel (i,j) is equivalent to (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j),
    # and transposes (j,i), (j,size-1-i), (size-1-j,i), (size-1-j,size-1-i).
    # To preserve symmetry, we flip all pixels in the orbit together.

    # Precompute orbits
    orbits = []
    visited = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in visited:
                continue
            orbit = set()
            # generate orbit under the square symmetry group
            pts = [(i, j)]
            for _ in range(8):
                new_pts = []
                for (a, b) in pts:
                    for (c, d) in [(size - 1 - a, b), (a, size - 1 - b), (size - 1 - a, size - 1 - b),
                                    (b, a), (b, size - 1 - a), (size - 1 - b, a), (size - 1 - b, size - 1 - a)]:
                        if (c, d) not in orbit:
                            orbit.add((c, d))
                            new_pts.append((c, d))
                pts = new_pts
            orbits.append(orbit)
            visited.update(orbit)

    # Random hill-climbing over orbit flips
    rng = np.random.default_rng(12345)
    current_cell = np.array(best_cell, dtype=np.float64)
    current_score = best_score
    n_orbits = len(orbits)

    max_iters = 400
    for it in range(max_iters):
        if _time_left() < 10.0:
            break
        # Pick a random orbit and flip it
        oi = rng.integers(0, n_orbits)
        trial = current_cell.copy()
        for (a, b) in orbits[oi]:
            trial[a, b] = 1.0 - trial[a, b]
        try:
            score = _evaluate(trial, problem, nmax, kpts)
        except Exception:
            continue
        if score > current_score:
            current_cell = trial
            current_score = score
            best_cell = trial.copy()
            best_score = score
        # Occasionally try multiple flips
        if it % 20 == 0 and _time_left() > 15.0:
            trial2 = current_cell.copy()
            for _ in range(3):
                oi2 = rng.integers(0, n_orbits)
                for (a, b) in orbits[oi2]:
                    trial2[a, b] = 1.0 - trial2[a, b]
            try:
                score2 = _evaluate(trial2, problem, nmax, kpts)
            except Exception:
                score2 = -1e9
            if score2 > current_score:
                current_cell = trial2
                current_score = score2
                best_cell = trial2.copy()
                best_score = score2

    # Final: ensure the cell is valid 0/1 and symmetric
    final_cell = _symmetrize(best_cell)
    # Convert to nested list of ints
    result = [[int(round(final_cell[i, j])) for j in range(size)] for i in range(size)]
    return result
