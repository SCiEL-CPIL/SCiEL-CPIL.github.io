"""Photonic-crystal unit-cell designer: superellipse-based TM band-gap optimization."""
import time
import math
import numpy as np
from scipy.linalg import eigh

# ---------- Deadline guard ----------
_DEADLINE = time.time() + 220  # leave 10s margin before 230s kill

# ---------- Symmetry helpers ----------
def _apply_c4v(cell):
    """Enforce C4v symmetry (full square symmetry) on a 32x32 cell."""
    n = len(cell)
    # Make a float array for averaging
    arr = np.array(cell, dtype=np.float64)
    # All 8 symmetry operations
    ops = [
        arr,
        arr[::-1, :],        # horizontal flip
        arr[:, ::-1],        # vertical flip
        arr[::-1, ::-1],     # 180 rotation
        arr.T,               # transpose
        arr[::-1, :].T,      # anti-diagonal
        arr[:, ::-1].T,      # main diagonal other way
        arr[::-1, ::-1].T,
    ]
    averaged = np.mean(ops, axis=0)
    # Threshold at 0.5
    return (averaged >= 0.5).astype(int)

def _superellipse_mask(n, a, b, m, cx, cy):
    """
    Return an n x n boolean mask for a superellipse centered at (cx, cy) in [0,1)^2,
    with semi-axes a, b and exponent m.
    Superellipse equation: |x/a|^m + |y/b|^m <= 1
    """
    xs = np.arange(n) / n  # 0 to (n-1)/n
    ys = np.arange(n) / n
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    dx = np.abs(X - cx) / a if a > 0 else np.inf
    dy = np.abs(Y - cy) / b if b > 0 else np.inf
    return (dx ** m + dy ** m) <= 1.0

def _build_cell(params, n=32):
    """
    Build a 32x32 cell from a list of superellipse parameters.
    params: list of (a, b, m, cx, cy, sign) where sign=+1 means dielectric, -1 means air (hole).
    """
    cell = np.zeros((n, n), dtype=np.int64)
    for (a, b, m, cx, cy, sign) in params:
        mask = _superellipse_mask(n, a, b, m, cx, cy)
        if sign > 0:
            cell[mask] = 1
        else:
            cell[mask] = 0
    # Enforce C4v
    cell = _apply_c4v(cell.tolist())
    return cell

# ---------- Band structure solver ----------
def _compute_gap(cell, n_pw=17):
    """
    Compute the gap-midgap ratio between TM bands 4 and 5 using plane-wave expansion.
    n_pw: number of plane-wave orders in each direction (total = (2*n_pw+1)^2)
    Returns the score (gap-midgap ratio), or -1 on failure.
    """
    n = len(cell)
    eps = np.array(cell, dtype=np.float64) * 11.0 + 1.0  # epsilon map: 1 or 12

    # Fourier coefficients of epsilon
    # G vectors
    g_max = n_pw
    G = []
    for gx in range(-g_max, g_max + 1):
        for gy in range(-g_max, g_max + 1):
            G.append((gx, gy))
    nG = len(G)

    # Compute Fourier coefficients of epsilon: eps[G] = sum_{i,j} eps[i,j] * exp(-i*2*pi*(Gx*i+Gy*j)/n) / n^2
    eps_F = np.zeros(nG, dtype=np.complex128)
    for idx, (gx, gy) in enumerate(G):
        # Use FFT for efficiency: compute DFT of eps at frequency (gx, gy)
        # Actually, let's just do direct sum for correctness
        s = 0.0
        for i in range(n):
            for j in range(n):
                s += eps[i, j] * np.exp(-2j * np.pi * (gx * i + gy * j) / n)
        eps_F[idx] = s / (n * n)

    # Build the TM band-structure matrix for a given k-point
    # For TM polarization: (k - G)^2 * eps_F[G' - G] * c_{G'}
    # Eigenvalue lambda = (omega * a / c)^2

    def _bands_at_k(kx, ky):
        # Build matrix M[G, G'] = |k - G|^2 * eps_F[G' - G]
        M = np.zeros((nG, nG), dtype=np.complex128)
        for i, (gx, gy) in enumerate(G):
            kx_i = kx - gx
            ky_i = ky - gy
            k2 = kx_i * kx_i + ky_i * ky_i
            for j, (gx2, gy2) in enumerate(G):
                # eps_F index for (G' - G) = (gx2 - gx, gy2 - gy)
                dgx = gx2 - gx
                dgy = gy2 - gy
                # Find index of (dgx, dgy) in G
                # We need a lookup
                pass
        # This is slow; let's use a dict for G indices
        G_dict = {(gx, gy): idx for idx, (gx, gy) in enumerate(G)}
        for i, (gx, gy) in enumerate(G):
            kx_i = kx - gx
            ky_i = ky - gy
            k2 = kx_i * kx_i + ky_i * ky_i
            for j, (gx2, gy2) in enumerate(G):
                dgx = gx2 - gx
                dgy = gy2 - gy
                if (dgx, dgy) in G_dict:
                    eps_idx = G_dict[(dgx, dgy)]
                    M[i, j] = k2 * eps_F[eps_idx]
        # The matrix should be Hermitian; symmetrize
        M = (M + M.conj().T) / 2.0
        # Solve eigenvalue problem
        try:
            w = np.linalg.eigvalsh(M)
            w = np.sort(w.real)
            return w
        except:
            return None

    # Sample k-points in the irreducible BZ of the square lattice
    # Irreducible BZ: 0 <= ky <= kx, kx in [0, pi], ky in [0, pi]
    # Actually for square lattice with a=1, BZ is [-pi, pi]^2, irreducible is triangle: 0<=ky<=kx<=pi
    # Let's sample along high-symmetry lines and interior points
    k_points = []
    # Gamma to X: (k, 0) for k in [0, pi]
    for t in np.linspace(0, np.pi, 10):
        k_points.append((t, 0))
    # X to M: (pi, k) for k in [0, pi]
    for t in np.linspace(0, np.pi, 10):
        k_points.append((np.pi, t))
    # M to Gamma: (k, k) for k in [0, pi]
    for t in np.linspace(0, np.pi, 10):
        k_points.append((t, t))
    # Some interior points
    for kx in np.linspace(0.3, 2.5, 5):
        for ky in np.linspace(0.1, kx, 5):
            if ky <= kx:
                k_points.append((kx, ky))

    max_w4 = 0.0
    min_w5 = np.inf
    for (kx, ky) in k_points:
        w = _bands_at_k(kx, ky)
        if w is None or len(w) < 5:
            continue
        # TM bands: index 0 is lowest. Bands 4 and 5 are indices 3 and 4 (0-based)
        w4 = w[3]
        w5 = w[4]
        max_w4 = max(max_w4, w4)
        min_w5 = min(min_w5, w5)

    if min_w5 <= max_w4:
        return -1.0
    gap = min_w5 - max_w4
    midgap = (min_w5 + max_w4) / 2.0
    if midgap <= 0:
        return -1.0
    return gap / midgap

# ---------- Optimization ----------
def _random_params(n_params=3):
    """Generate random superellipse parameters."""
    params = []
    for _ in range(n_params):
        a = np.random.uniform(0.1, 0.45)
        b = np.random.uniform(0.1, 0.45)
        m = np.random.uniform(1.0, 4.0)  # 2=circle, 4=square
        cx = np.random.uniform(0.2, 0.8)
        cy = np.random.uniform(0.2, 0.8)
        sign = 1 if np.random.random() < 0.6 else -1
        params.append((a, b, m, cx, cy, sign))
    return params

def _mutate_params(params, rate=0.3):
    """Mutate parameters."""
    new_params = []
    for p in params:
        a, b, m, cx, cy, sign = p
        if np.random.random() < rate:
            a = max(0.05, min(0.48, a + np.random.normal(0, 0.05)))
        if np.random.random() < rate:
            b = max(0.05, min(0.48, b + np.random.normal(0, 0.05)))
        if np.random.random() < rate:
            m = max(1.0, min(6.0, m + np.random.normal(0, 0.5)))
        if np.random.random() < rate:
            cx = max(0.1, min(0.9, cx + np.random.normal(0, 0.05)))
        if np.random.random() < rate:
            cy = max(0.1, min(0.9, cy + np.random.normal(0, 0.05)))
        if np.random.random() < rate * 0.5:
            sign = -sign
        new_params.append((a, b, m, cx, cy, sign))
    return new_params

def _crossover(p1, p2):
    """Uniform crossover."""
    child = []
    for a, b in zip(p1, p2):
        child.append(a if np.random.random() < 0.5 else b)
    return child

def design(problem):
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    eps_low = problem.get("eps_low", 1)

    # Start with a reasonable initial design: centered superellipse
    best_params = [(0.25, 0.25, 2.0, 0.5, 0.5, 1)]
    best_cell = _build_cell(best_params, size)
    best_score = -1.0

    # Evaluate initial
    try:
        s = _compute_gap(best_cell, n_pw=8)
        if s > best_score:
            best_score = s
    except:
        pass

    # Genetic algorithm
    pop_size = 20
    population = [best_params]
    for _ in range(pop_size - 1):
        population.append(_random_params(n_params=3))

    gen = 0
    max_gen = 30
    while gen < max_gen:
        if time.time() > _DEADLINE - 30:
            break

        # Evaluate population
        scores = []
        for p in population:
            cell = _build_cell(p, size)
            try:
                s = _compute_gap(cell, n_pw=8)
            except:
                s = -1.0
            scores.append(s)

        # Find best in population
        idx_best = np.argmax(scores)
        if scores[idx_best] > best_score:
            best_score = scores[idx_best]
            best_params = population[idx_best]
            best_cell = _build_cell(best_params, size)

        # Create next generation
        new_pop = []
        # Elitism
        new_pop.append(population[idx_best])

        for _ in range(pop_size - 1):
            if np.random.random() < 0.2:
                # Random new
                child = _random_params(n_params=3)
            else:
                # Crossover + mutation
                i1, i2 = np.random.choice(pop_size, 2, replace=False)
                child = _crossover(population[i1], population[i2])
                child = _mutate_params(child)
            new_pop.append(child)

        population = new_pop
        gen += 1

    # Final refinement: local search around best
    if time.time() < _DEADLINE - 15:
        for _ in range(50):
            if time.time() > _DEADLINE - 10:
                break
            p = _mutate_params(best_params, rate=0.1)
            cell = _build_cell(p, size)
            try:
                s = _compute_gap(cell, n_pw=10)
            except:
                s = -1.0
            if s > best_score:
                best_score = s
                best_params = p
                best_cell = cell

    return best_cell
