# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import eigh

EPS_HIGH = 12.0

# ---------------------------------------------------------------------------
# Symmetry handling: keep the 32x32 cell invariant under the square symmetry
# group about the cell centre (row-reversal, column-reversal and transpose).
# ---------------------------------------------------------------------------
def _symmetrize(cell):
    c = np.asarray(cell, dtype=np.int32)
    r = c[::-1, :]
    t = c.T
    tr = t[::-1, :]
    # consensus: a pixel is dielectric if a majority of the 4 symmetric copies are
    return (c + r + t + tr >= 3).astype(np.int32)

def _free_indices(cell):
    """Pixels that can be flipped without breaking the 4-fold symmetry.
    Returns (indices, group_size)."""
    n = cell.shape[0]
    c = np.asarray(cell, dtype=np.int32)
    idx = []
    for i in range(n):
        for j in range(n):
            if c[i, j] != c[n - 1 - i, j] or c[i, j] != c[j, i] or c[i, j] != c[n - 1 - i, n - 1 - j]:
                continue
            # canonical representative: lexicographically smallest orbit member
            orbit = [(i, j), (n - 1 - i, j), (j, i), (n - 1 - i, n - 1 - j)]
            if (i, j) == min(orbit):
                idx.append((i, j))
    return np.array(idx, dtype=int), 4

# ---------------------------------------------------------------------------
# Fast, accurate TM plane-wave (Galerkin) solver.
# ---------------------------------------------------------------------------
def _fourier_inv_eps(cell):
    """Fourier coefficients of 1/eps on the 2D grid via separable FFT."""
    n = cell.shape[0]
    eps = 1.0 + (EPS_HIGH - 1.0) * cell.astype(float)
    inv = 1.0 / eps
    F = np.fft.fft2(inv) / (n * n)
    return F  # F[m, k] with m, k in 0..n-1

def _band_eigs(F, n_pw, kx, ky, n_bands):
    """Return the lowest n_bands eigenvalues (omega^2) at reduced k=(kx,ky)."""
    n = F.shape[0]
    ks = list(range(-n_pw, n_pw + 1))
    n_g = len(ks)
    G = np.array([[m, kk] for m in ks for kk in ks], dtype=float)  # (n_g,2)
    # fourier_inv_eps[d] for d in -2n_pw..2n_pw
    diff = G[:, 0:1] - G[:, 1:]  # (n_g, n_g, 2)
    dm = diff[:, :, 0]
    dn = diff[:, :, 1]
    dmw = dm % n
    dnw = dn % n
    Fdiff = F[np.ix_(dmw, dnw)]  # (n_g, n_g)
    K = 2.0 * np.pi * (G + np.array([kx, ky]))  # (n_g,2)
    # A[i,j] = K[i].K[j] * Fdiff[i,j]
    A = (K @ K.T) * Fdiff
    A = 0.5 * (A + A.T)  # enforce Hermitian
    w = eigh(A, subset_by_index=[0, n_bands - 1])
    return np.real(w[0])

def _bands_at(F, n_pw, kpts, n_bands):
    out = np.zeros((len(kpts), n_bands))
    for i, (kx, ky) in enumerate(kpts):
        out[i] = _band_eigs(F, n_pw, kx, ky, n_bands)
    return out

def _gap_ratio(bands, lb, ub):
    """gap-midgap ratio between band lb (1-indexed) and ub (1-indexed)."""
    max_low = np.max(bands[:, lb - 1])
    min_up = np.min(bands[:, ub - 1])
    denom = min_up + max_low
    if denom < 1e-12:
        return -1.0
    return 2.0 * (min_up - max_low) / denom

def _field_intensity(F, n_pw, kx, ky, band_idx):
    """|E|^2 (plane-wave amplitude squared) at reduced k for a given band index."""
    n = F.shape[0]
    ks = list(range(-n_pw, n_pw + 1))
    n_g = len(ks)
    G = np.array([[m, kk] for m in ks for kk in ks], dtype=float)
    diff = G[:, 0:1] - G[:, 1:]
    Fdiff = F[np.ix_((diff[:, :, 0]) % n, (diff[:, :, 1]) % n)]
    K = 2.0 * np.pi * (G + np.array([kx, ky]))
    A = (K @ K.T) * Fdiff
    A = 0.5 * (A + A.T)
    w, v = eigh(A, subset_by_index=[0, band_idx])
    vec = v[:, band_idx]
    inten = np.abs(vec) ** 2
    inten = inten / (np.sum(inten) + 1e-30)
    return inten, G

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def design(problem):
    t0 = time.time()
    DEADLINE = t0 + 210.0  # safety margin well under the 230s kill

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", EPS_HIGH)
    lb = problem.get("lower_band", 4)
    ub = lb + 1

    # --- K-points: boundary + interior of the irreducible BZ of a square lattice
    # Irreducible BZ: 0 <= ky <= kx <= 0.5 (reduced units, a=1)
    kpts = []
    # high-symmetry points
    kpts += [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.25, 0.25)]
    # boundary X->M (kx=ky, 0..0.5) and Gamma->X (ky=0, 0..0.5)
    for t in [0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.45]:
        kpts.append((t, 0.0))
        kpts.append((t, t))
    # interior lines
    for t in [0.15, 0.25, 0.35, 0.45]:
        kpts.append((t, 0.5 * t))
        kpts.append((0.5, t))
    kpts = list(dict.fromkeys(kpts))  # dedupe, keep order

    # --- accuracy settings (289 plane waves = 17^2, matching the evaluator)
    n_pw = 17
    n_bands = 7  # need up to band ub (=5) with a couple of slack

    def now():
        return time.time()

    def solve(cell):
        F = _fourier_inv_eps(cell)
        bands = _bands_at(F, n_pw, kpts, n_bands)
        return F, bands

    # --- seed: centred square-ish rod
    cell = np.zeros((size, size), dtype=np.int32)
    c = size // 2
    for i in range(size):
        for j in range(size):
            if abs(i - c + 0.5) < 7.5 and abs(j - c + 0.5) < 7.5:
                cell[i, j] = 1
    cell = _symmetrize(cell)

    F, bands = solve(cell)
    best = _gap_ratio(bands, lb, ub)
    best_cell = cell.copy()
    best_F = F
    best_bands = bands

    # --- field-guided topological optimization
    free, gsize = _free_indices(cell)
    if len(free) == 0:
        return best_cell.tolist()

    # candidate k-points where band lb is maximal / band ub is minimal
    k_low = kpts[int(np.argmax(bands[:, lb - 1]))]
    k_up = kpts[int(np.argmin(bands[:, ub - 1]))]

    inten_low, G_low = _field_intensity(best_F, n_pw, k_low[0], k_low[1], lb - 1)
    inten_up, G_up = _field_intensity(best_F, n_pw, k_up[0], k_up[1], ub - 1)

    def recompute_intens(F, cell, k_low, k_up):
        il, Gl = _field_intensity(F, n_pw, k_low[0], k_low[1], lb - 1)
        iu, Gu = _field_intensity(F, n_pw, k_up[0], k_up[1], ub - 1)
        return il, Gl, iu, Gu

    def sample_field(inten, G, cell):
        """Map plane-wave intensities back onto pixels by inverse-DFT magnitude."""
        n = cell.shape[0]
        grid = np.zeros((n, n), dtype=complex)
        for idx, (m, kk) in enumerate(G):
            grid[m % n, kk % n] = inten[idx]
        # inverse FFT to get real-space field profile
        prof = np.abs(np.fft.ifft2(grid))
        prof = prof / (np.max(prof) + 1e-30)
        return prof

    prof_low = sample_field(inten_low, G_low, cell)
    prof_up = sample_field(inten_up, G_up, cell)

    # score for flipping pixel: add dielectric where band lb is strong,
    # remove where band ub is strong.
    def flip_gain(i, j, cur):
        # cur==1 means currently dielectric
        if cur == 1:
            # removing dielectric: good if band ub strong here, bad if band lb strong
            return prof_up[i, j] - prof_low[i, j]
        else:
            # adding dielectric: good if band lb strong here
            return prof_low[i, j] - prof_up[i, j]

    # greedy hill-climb
    iters = 0
    while now() < DEADLINE - 25.0:
        iters += 1
        gains = np.array([flip_gain(i, j, cell[i, j]) for i, j in free])
        order = np.argsort(-gains)
        improved = False
        for pos in order:
            if now() > DEADLINE - 25.0:
                break
            i, j = free[pos]
            if gains[pos] <= 1e-6:
                break
            # flip whole orbit
            orbit = [(i, j), (size - 1 - i, j), (j, i), (size - 1 - i, size - 1 - j)]
            for (oi, oj) in orbit:
                cell[oi, oj] = 1 - cell[oi, oj]
            F, bands = solve(cell)
            score = _gap_ratio(bands, lb, ub)
            if score > best:
                best = score
                best_cell = cell.copy()
                best_F = F
                best_bands = bands
                improved = True
                # refresh field guides
                k_low = kpts[int(np.argmax(bands[:, lb - 1]))]
                k_up = kpts[int(np.argmin(bands[:, ub - 1]))]
                inten_low, G_low, inten_up, G_up = recompute_intens(F, cell, k_low, k_up)
                prof_low = sample_field(inten_low, G_low, cell)
                prof_up = sample_field(inten_up, G_up, cell)
                gains = np.array([flip_gain(i, j, cell[i, j]) for i, j in free])
                order = np.argsort(-gains)
            else:
                # revert
                for (oi, oj) in orbit:
                    cell[oi, oj] = 1 - cell[oi, oj]
        if not improved:
            break

    # final safety: ensure symmetry and return
    best_cell = _symmetrize(best_cell)
    return best_cell.tolist()
# EVOLVE-BLOCK-END
