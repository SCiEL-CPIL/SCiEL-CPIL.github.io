"""Spectral inverse design: internal PWE solver + symmetric 4-quadrant search."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def _pwe_gap(cell, n_pw=15, n_wv=12, eps_low=1.0, eps_high=12.0):
    """Return (score, min_w5, max_w4). cell: N x N (0/1). TM bands via PWE."""
    N = cell.shape[0]
    eps = eps_low + (eps_high - eps_low) * cell.astype(float)
    inv_eps = 1.0 / eps
    # Fourier coefficients of 1/eps
    if N > 0 and cell[0, 0] == 1 and cell[0, 1] == 1 and cell[1, 0] == 1:
        inv_eps_f = np.zeros(1)  # uniform dielectric
    else:
        inv_eps_f = np.fft.fft2(inv_eps) / (N * N)
    # plane-wave grid
    Np = 2 * n_pw + 1
    g = np.arange(-n_pw, n_pw + 1)
    Gx, Gy = np.meshgrid(g, g, indexing='ij')
    G2 = Gx * Gx + Gy * Gy
    G2 = np.where(G2 > 0, G2, 1.0)
    # fourier index lookup
    n_off = n_pw
    idx = (Gy + n_off) + (Gx + n_off) * Np
    idx = idx.ravel().astype(int)
    G2f = G2.ravel()
    n_pw_count = Np * Np
    # irreducible BZ: Gx>=0, Gx>=Gy
    kx_grid = np.linspace(0.0, 0.5, n_wv)
    ky_grid = np.linspace(0.0, 0.5, n_wv)
    max_w4 = 0.0
    min_w5 = np.inf
    for kx in kx_grid:
        for ky in ky_grid:
            if kx < ky - 1e-9:
                continue
            if kx < 0 or kx > 0.5 + 1e-9 or ky < 0 or ky > 0.5 + 1e-9:
                continue
            Kx = kx + Gx
            Ky = ky + Gy
            K2 = Kx * Kx + Ky * Ky
            K2 = np.where(K2 > 0, K2, 1.0).ravel()
            # A[m,n] = delta(m,n)/G2[m] + 1/eps_f[m-n] / K2[m]
            A = np.zeros((n_pw_count, n_pw_count))
            for m in range(n_pw_count):
                A[m, m] = 1.0 / G2f[m]
                # sum over n
                for n in range(n_pw_count):
                    dm = idx[m] - idx[n]
                    if -n_off <= dm <= n_off:
                        A[m, n] += inv_eps_f[dm + n_off] / K2[m]
            # symmetric Hermitian
            A = 0.5 * (A + A.T.conj())
            try:
                vals = np.linalg.eigvalsh(A)
            except np.linalg.LinAlgError:
                continue
            vals = vals[vals > 0]
            if len(vals) < 5:
                continue
            w4 = math.sqrt(vals[3])
            w5 = math.sqrt(vals[4])
            if w4 > max_w4:
                max_w4 = w4
            if w5 < min_w5:
                min_w5 = w5
    if not (min_w5 < np.inf and max_w4 > 0):
        return 0.0, 0.0, 0.0
    score = 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)
    return score, min_w5, max_w4


def _build_cell(size, a, b, c, d, e):
    """Symmetric 4-quadrant rod: each quadrant has its own radius.
    Quadrant boundaries at centre. r in [0, 0.5]."""
    half = size // 2
    cell = np.zeros((size, size), dtype=int)
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    radii = [a, b, c, d]
    for i in range(size):
        for j in range(size):
            dx = j - cx
            dy = i - cy
            # quadrant: (dx>=0, dy>=0)=Q0 ; (dx<0,dy>=0)=Q1 ; (dx<0,dy<0)=Q2 ; (dx>=0,dy<0)=Q3
            if dx >= 0 and dy >= 0:
                r = radii[0]
            elif dx < 0 and dy >= 0:
                r = radii[1]
            elif dx < 0 and dy < 0:
                r = radii[2]
            else:
                r = radii[3]
            if dx * dx + dy * dy < r * r:
                cell[i, j] = 1
    # enforce full square symmetry by averaging quadrants
    cell = np.maximum(cell, cell[:, ::-1])
    cell = np.maximum(cell, cell[::-1, :])
    cell = np.maximum(cell, cell.T)
    return cell


def design(problem):
    """Search over symmetric 4-quadrant rod radii using an internal PWE scorer."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    deadline = time.time() + 150.0
    best_cell = None
    best_score = -np.inf

    def consider(cell, score):
        nonlocal best_cell, best_score
        if score > best_score:
            best_score = score
            best_cell = cell.copy()

    # seed: uniform radius
    for r0 in [0.2, 0.22, 0.25, 0.28, 0.3, 0.32, 0.35]:
        if time.time() > deadline - 20.0:
            break
        cell = _build_cell(size, r0, r0, r0, r0, 0.0)
        s, _, _ = _pwe_gap(cell, n_pw=15, n_wv=12, eps_low=eps_low, eps_high=eps_high)
        consider(cell, s)

    # local refinement around best: vary each quadrant radius
    if best_cell is not None:
        # reconstruct base radii from best (we stored cell; re-derive by trying)
        # Use a coarse grid over (r_center, r_off)
        for rc in [0.2, 0.24, 0.28, 0.32, 0.36]:
            for ro in [0.0, 0.03, 0.06, 0.09]:
                if time.time() > deadline - 15.0:
                    break
                # alternate quadrants
                cell = _build_cell(size, rc, rc - ro, rc, rc - ro, 0.0)
                s, _, _ = _pwe_gap(cell, n_pw=15, n_wv=12, eps_low=eps_low, eps_high=eps_high)
                consider(cell, s)

    if best_cell is None:
        # fallback: seed rod
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
    return best_cell.tolist()
# EVOLVE-BLOCK-END
