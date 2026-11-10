"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def _enforce_symmetry(cell):
    """Enforce square symmetry: 180° rotational, mirror x, mirror y."""
    n = len(cell)
    c = np.array(cell, dtype=int)
    # 180° rotational symmetry: c[i,j] == c[n-1-i, n-1-j]
    c = (c + c[::-1, ::-1]) // 2
    # Mirror x: c[i,j] == c[n-1-i, j]
    c = (c + c[::-1, :]) // 2
    # Mirror y: c[i,j] == c[i, n-1-j]
    c = (c + c[:, ::-1]) // 2
    return c.tolist()


def _compute_gap_fast(cell, n_pw=9, k_points=6):
    """
    Fast approximate band-structure evaluation using a small plane-wave basis.
    Returns the gap-midgap ratio between bands 4 and 5 (TM polarization).
    This is a proxy for the full 289-PW evaluator.
    """
    size = len(cell)
    eps_high = 12.0
    eps_low = 1.0
    eps_mean = (eps_high + eps_low) / 2.0
    eps_var = (eps_high - eps_low) / 2.0

    # Build dielectric map as a 2D array
    eps_map = np.array(cell, dtype=float)
    eps_map = eps_mean + eps_var * (2 * eps_map - 1)  # 1->eps_high, 0->eps_low
    # Actually: 1 = dielectric (eps 12), 0 = air (eps 1)
    # So eps_map[i,j] = 12 if cell[i,j]==1 else 1
    eps_map = np.where(np.array(cell, dtype=int) == 1, eps_high, eps_low)

    # Compute inverse epsilon for TM: (1/eps) is the operator
    inv_eps_map = 1.0 / eps_map

    # Compute Fourier coefficients of inv_eps up to order n_pw
    # G vectors: G = (2π/size)*(m, n) for |m|,|n| <= n_pw
    max_g = n_pw
    g_list = []
    for m in range(-max_g, max_g + 1):
        for n in range(-max_g, max_g + 1):
            if m * m + n * n <= max_g * max_g:
                g_list.append((m, n))

    # Fourier coefficients: eps_inv_G = (1/N^2) sum_{i,j} inv_eps[i,j] * exp(-i G . r_{i,j})
    # r_{i,j} = (i+0.5, j+0.5)/size in lattice units
    N = size
    fourier_coeffs = {}
    for (m, n) in g_list:
        Gx = 2.0 * math.pi * m / N
        Gy = 2.0 * math.pi * n / N
        s = 0.0
        for i in range(N):
            for j in range(N):
                s += inv_eps_map[i, j] * math.cos(Gx * (i + 0.5) / N + Gy * (j + 0.5) / N) * 2.0
        fourier_coeffs[(m, n)] = s / (N * N) * 2.0  # factor of 2 for real part; we need complex

    # Actually let's do it properly with complex exponentials
    fourier_coeffs = {}
    for (m, n) in g_list:
        Gx = 2.0 * math.pi * m
        Gy = 2.0 * math.pi * n
        s = 0.0 + 0.0j
        for i in range(N):
            xi = (i + 0.5) / N
            for j in range(N):
                xj = (j + 0.5) / N
                s += inv_eps_map[i, j] * math.exp(-1j * (Gx * xi + Gy * xj))
        fourier_coeffs[(m, n)] = s / (N * N)

    # K points in irreducible BZ: (0,0), (0.5,0), (0.5,0.5), (0.25,0.25), (0.4,0.1), (0.3,0.2)
    k_points_list = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.25, 0.25), (0.4, 0.1), (0.3, 0.2)]

    max_omega4 = 0.0
    min_omega5 = float('inf')

    for (kx, ky) in k_points_list:
        Kx = 2.0 * math.pi * kx
        Ky = 2.0 * math.pi * ky

        # Build Hamiltonian matrix H[G1, G2] = |K+G1|^2 * eps_inv_{G1-G2}
        # For TM: (1/eps) * |k+G|^2
        nw = len(g_list)
        H = np.zeros((nw, nw), dtype=complex)

        for idx1, (m1, n1) in enumerate(g_list):
            G1x = 2.0 * math.pi * m1
            G1y = 2.0 * math.pi * n1
            kpx = Kx + G1x
            kpy = Ky + G1y
            k2 = kpx * kpx + kpy * kpy

            for idx2, (m2, n2) in enumerate(g_list):
                dm = m1 - m2
                dn = n1 - n2
                if (dm, dn) in fourier_coeffs:
                    H[idx1, idx2] = k2 * fourier_coeffs[(dm, dn)]

        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(np.real(eigvals))
            # omega = sqrt(eigenvalue) / (2*pi) roughly; but we just need ratios
            # Take bands 4 and 5 (0-indexed: 3 and 4)
            if len(eigvals) >= 5:
                w4 = math.sqrt(max(eigvals[3], 1e-12))
                w5 = math.sqrt(max(eigvals[4], 1e-12))
                max_omega4 = max(max_omega4, w4)
                min_omega5 = min(min_omega5, w5)
        except:
            pass

    if min_omega5 == float('inf') or max_omega4 == 0.0:
        return -1.0

    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    if midgap <= 0:
        return -1.0
    return 2.0 * gap / midgap


def design(problem):
    """
    Design a symmetric 32x32 unit cell maximizing the TM band gap between bands 4 and 5.
    Uses a fast proxy band-structure calculation to guide a local search.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    polarization = problem.get("polarization", "TM")

    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill

    # Start with a reasonable initial design: a dielectric ring or cross pattern
    # Try a cross pattern (dielectric in middle row and column)
    cell = [[0] * size for _ in range(size)]
    mid = size // 2
    for i in range(size):
        for j in range(size):
            # Cross: middle 3 rows and columns
            if abs(i - (mid - 0.5)) < 2 or abs(j - (mid - 0.5)) < 2:
                cell[i][j] = 1

    cell = _enforce_symmetry(cell)

    # Evaluate initial design
    best_score = _compute_gap_fast(cell)
    best_cell = [row[:] for row in cell]

    # Local search: flip pixels one at a time
    rng = random.Random(42)
    n_iter = 0

    while time.time() < deadline - 30.0:  # Leave 30s margin
        # Try random single-pixel flips
        improved = False
        for _ in range(50):  # 50 flips per iteration
            if time.time() > deadline - 30.0:
                break

            i = rng.randint(0, size - 1)
            j = rng.randint(0, size - 1)
            cell[i][j] = 1 - cell[i][j]
            cell = _enforce_symmetry(cell)

            score = _compute_gap_fast(cell)
            if score > best_score:
                best_score = score
                best_cell = [row[:] for row in cell]
                improved = True
            else:
                # Revert
                cell[i][j] = 1 - cell[i][j]
                cell = _enforce_symmetry(cell)

        n_iter += 1
        if not improved and n_iter > 10:
            # Try a different initial perturbation
            for _ in range(20):
                i = rng.randint(0, size - 1)
                j = rng.randint(0, size - 1)
                best_cell[i][j] = 1 - best_cell[i][j]
            best_cell = _enforce_symmetry(best_cell)
            cell = [row[:] for row in best_cell]
            best_score = _compute_gap_fast(cell)
            n_iter = 0

    return best_cell
# EVOLVE-BLOCK-END
