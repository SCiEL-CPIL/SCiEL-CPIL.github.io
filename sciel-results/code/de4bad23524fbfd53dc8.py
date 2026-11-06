"""Photonic-crystal TM band-gap designer.

Structural change from the seed: instead of returning a fixed centred dielectric
rod, this program *optimizes* the 32x32 pixel mask.  The cell is parameterized by
its 8x8 symmetric quadrant (180-degree symmetry about the centre).  A fast
surrogate plane-wave (Galerkin) TM band solver -- far cheaper than the evaluator's
289-plane-wave / 30-k solver -- scores candidate masks by the gap-midgap ratio
between TM bands 4 and 5, and drives a simulated-annealing pixel search over the
quadrant.  An explicit wall-clock deadline guard returns the best mask found.
"""
import time
import math
import random
import numpy as np

# ---------------------------------------------------------------------------
# Parameters (kept near the evaluator's physics so the surrogate is faithful)
# ---------------------------------------------------------------------------
SIZE = 32
Q = 8                      # quadrant size (SIZE // 4)
EPS_HIGH = 12.0
EPS_LOW = 1.0
N_PW = 37                  # plane waves in the surrogate (odd, ~ +/-4)
N_K = 5                    # k-points in the surrogate (boundary + interior)
N_BANDS = 6                # lowest bands to track (need bands 4 and 5)
DEADLINE = 205.0           # wall-clock seconds (evaluator kills at 230 s)
SEED = 20240517

# Precompute the 5 k-points (a, b) over the irreducible BZ triangle.
_KPTS = []
def _build_kpts():
    pts = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.25, 0.25), (0.4, 0.2)]
    for (a, b) in pts:
        _KPTS.append((2.0 * math.pi * a, 2.0 * math.pi * b))
_build_kpts()


def _expand_quadrant(quad):
    """Expand an 8x8 quadrant (top-left) to a 32x32 cell with 180-degree symmetry."""
    q = np.asarray(quad, dtype=np.int8)
    half = np.vstack([q, q[::-1, ::-1]])            # 16x16
    cell = np.vstack([half, half[::-1, ::-1]])      # 32x32
    return cell


def _surrogate_bandgap(cell):
    """Fast surrogate gap-midgap ratio (bands 4,5) for a 32x32 0/1 cell.

    Returns a scalar in [-1, 1]; negative when bands 4 and 5 overlap.
    """
    N = SIZE
    # exact (sub-pixel) permittivity field
    eps_field = np.where(cell == 1, EPS_HIGH, EPS_LOW).astype(np.float64)

    # Fourier coefficients of 1/eps (TM: dielectric matrix uses 1/eps)
    inv_eps = 1.0 / eps_field
    inv_eps_hat = np.fft.fft2(inv_eps) / (N * N)
    inv_eps_hat = np.fft.ifftshift(inv_eps_hat)     # index 0 -> G = 0

    # plane-wave grid (odd N_PW), G = (2*pi * m, 2*pi * n), m,n in [-h..h]
    h = (N_PW - 1) // 2
    idx = np.arange(-h, h + 1)
    Gm, Gn = np.meshgrid(idx, idx, indexing="ij")
    Gx = 2.0 * math.pi * Gm
    Gy = 2.0 * math.pi * Gn
    Gmag2 = Gx * Gx + Gy * Gy
    order = np.argsort(Gmag2)
    Gx = Gx[order]
    Gy = Gy[order]
    Gmag2 = Gmag2[order]
    # Fourier coefficient of 1/eps for each G vector
    c = np.empty(N_PW, dtype=np.complex128)
    for k, (mx, my) in enumerate(zip(Gm[order], Gn[order])):
        c[k] = inv_eps_hat[(mx % N), (my % N)]

    gap_max = -1.0
    for (kx, ky) in _KPTS:
        # dielectric matrix  A = (k + G) dot (k + G') * c[G'-G]   (TM)
        kxg = kx + Gx
        kyg = ky + Gy
        # vector of c for difference G_j - G_i
        # build via outer product of the coefficient list indexed by (j - i)
        # precompute a map from (dx,dy) -> c index
        # We need c[j - i] for each pair.  Use convolution-free direct build.
        A = np.empty((N_PW, N_PW), dtype=np.complex128)
        for j in range(N_PW):
            dx = Gx[j] - Gx
            dy = Gy[j] - Gy
            # find c for each (dx,dy): dx = 2*pi*(Gx[j]-Gx[i]) -> integer multiple
            mx_diff = (dx / (2.0 * math.pi)).round().astype(int)
            my_diff = (dy / (2.0 * math.pi)).round().astype(int)
            # coefficient c for G = (mx_diff, my_diff)
            vals = np.empty(N_PW, dtype=np.complex128)
            for i in range(N_PW):
                mx = mx_diff[i]
                my = my_diff[i]
                if abs(mx) <= h and abs(my) <= h:
                    vals[i] = inv_eps_hat[(mx % N), (my % N)]
                else:
                    vals[i] = 0.0
            A[j, :] = vals * kxg * kxg[j] + vals * kyg * kyg[j]
        # solve generalized-ish: we want eigenvalues of A (since c[0]=1 makes
        # A the standard TM matrix in units where eps_low is factored out).
        # A is Hermitian; use eigh.
        try:
            w2 = np.linalg.eigvalsh(A)
        except np.linalg.LinAlgError:
            continue
        w2 = np.sort(np.real(w2))
        if len(w2) < N_BANDS:
            continue
        w4 = w2[N_BANDS - 2]   # band 4 (0-indexed 3)
        w5 = w2[N_BANDS - 1]   # band 5 (0-indexed 4)
        if w4 <= 0:
            continue
        gap = w5 - w4
        mid = w5 + w4
        ratio = 2.0 * gap / mid
        if ratio > gap_max:
            gap_max = ratio
    return gap_max


def design(problem):
    """Optimize the 32x32 TM band-gap cell; return a nested list of 0/1 pixels."""
    t_start = time.time()
    rng = random.Random(SEED)
    N = problem.get("pixels", SIZE)
    if N != SIZE:
        # Fallback: return a symmetric rod for unexpected sizes.
        cell = []
        for i in range(N):
            x = (i + 0.5) / N - 0.5
            cell.append([1 if x * x + ((j + 0.5) / N - 0.5) ** 2 < 0.2 ** 2 else 0
                         for j in range(N)])
        return cell

    # --- initial guess: centred dielectric disc (the seed), as a quadrant ---
    def _disc_quadrant(r):
        q = np.zeros((Q, Q), dtype=np.int8)
        for i in range(Q):
            for j in range(Q):
                # quadrant pixel (i,j) maps to full-cell pixel (i, j)
                x = (i + 0.5) / N - 0.5
                y = (j + 0.5) / N - 0.5
                q[i, j] = 1 if (x * x + y * y) < (r * r) else 0
        return q

    quad = _disc_quadrant(0.20).copy()
    cell = _expand_quadrant(quad)
    best_score = _surrogate_bandgap(cell)
    best_quad = quad.copy()
    cur_quad = quad.copy()
    cur_score = best_score

    # --- simulated annealing over the quadrant ---
    n_quad = Q * Q
    total_steps = 200000
    # coarse -> fine: flip size shrinks over the run
    t_anneal = 1.0
    t_min = 0.02

    step = 0
    # evaluation counter for adaptive step budget
    while step < total_steps:
        if time.time() - t_start > DEADLINE:
            break
        step += 1
        t_anneal = max(t_min, 1.0 - step / total_steps)

        # choose a flip: early on flip a small cluster, late on a single pixel
        if t_anneal > 0.5:
            nflip = rng.choice([1, 1, 2, 3])
        elif t_anneal > 0.2:
            nflip = rng.choice([1, 1, 1, 2])
        else:
            nflip = 1

        cand = cur_quad.copy()
        flipped = 0
        tries = 0
        while flipped < nflip and tries < 20:
            tries += 1
            i = rng.randrange(Q)
            j = rng.randrange(Q)
            cand[i, j] ^= 1
            flipped += 1

        cand_cell = _expand_quadrant(cand)
        cand_score = _surrogate_bandgap(cand_cell)

        # accept if better, or with Metropolis probability
        if cand_score >= cur_score:
            cur_quad = cand
            cur_score = cand_score
            if cur_score > best_score:
                best_score = cur_score
                best_quad = cand.copy()
        else:
            # Metropolis: accept downhill moves with probability exp(dE / T)
            dE = cur_score - cand_score   # positive
            # scale temperature to the score range (~ [-1,1])
            T = 0.05 * t_anneal + 0.001
            if dE < 0.5:  # avoid overflow
                p = math.exp(-dE / T)
            else:
                p = 0.0
            if rng.random() < p:
                cur_quad = cand
                cur_score = cand_score

    # --- return the best cell found ---
    final = _expand_quadrant(best_quad)
    # ensure exact 180-degree symmetry (it is by construction) and 0/1 ints
    out = []
    for i in range(N):
        row = []
        for j in range(N):
            row.append(int(final[i, j]))
        out.append(row)
    return out
