"""Photonic-crystal unit cell design: large TM band gap between bands 4 and 5."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    polarization = problem.get("polarization", "TM")

    deadline = time.time() + 110.0
    best_cell = _make_rod(size, 0.2)
    best_score = -1.0

    candidates = []
    for r in [0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28]:
        candidates.append(_make_rod(size, r))
    for frac in [0.25, 0.3, 0.35, 0.4]:
        candidates.append(_make_square(size, frac))
    candidates.append(_make_cross(size, 0.35))
    candidates.append(_make_diamond(size, 0.25))
    candidates.append(_make_diamond(size, 0.3))
    for r in [0.15, 0.2, 0.25, 0.3]:
        candidates.append(_make_air_hole(size, r))

    for cell in candidates:
        if time.time() > deadline:
            break
        score = _evaluate_gap(cell, size, eps_low, eps_high, lower_band)
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in cell]

    if time.time() < deadline - 20:
        best_cell = _local_search(best_cell, size, eps_low, eps_high, lower_band, deadline)

    return best_cell


def _make_rod(size, radius):
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            row.append(1 if x * x + y * y < radius * radius else 0)
        cell.append(row)
    return cell


def _make_square(size, frac):
    half = max(1, int(size * frac / 2))
    start = (size - 2 * half) // 2
    cell = [[0] * size for _ in range(size)]
    for i in range(start, start + 2 * half):
        for j in range(start, start + 2 * half):
            cell[i][j] = 1
    return cell


def _make_cross(size, frac):
    half = max(1, int(size * frac / 2))
    mid = size // 2
    cell = [[0] * size for _ in range(size)]
    for j in range(size):
        for i in range(mid - half, mid + half):
            cell[i][j] = 1
    for i in range(size):
        for j in range(mid - half, mid + half):
            cell[i][j] = 1
    return cell


def _make_diamond(size, radius):
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            row.append(1 if abs(x) + abs(y) < radius else 0)
        cell.append(row)
    return cell


def _make_air_hole(size, radius):
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            row.append(0 if x * x + y * y < radius * radius else 1)
        cell.append(row)
    return cell


def _evaluate_gap(cell, size, eps_low, eps_high, lower_band):
    """Efficient TM band gap calculation using plane-wave method."""
    eps_map = np.array(cell, dtype=float)
    eps_map = eps_map * (eps_high - eps_low) + eps_low
    inv_eps = 1.0 / eps_map

    # Fourier coefficients of 1/eps via FFT
    inv_eps_fft = np.fft.fftshift(np.fft.fft2(inv_eps)) / (size * size)

    # Use a smaller plane-wave set for speed: 9x9 = 81 plane waves
    N = 4  # half-width, giving 9x9 = 81 plane waves
    G_vals = np.arange(-N, N + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    n_pw = len(G_vals) ** 2

    # Precompute c_{G-G'} matrix
    c_mat = np.zeros((n_pw, n_pw), dtype=complex)
    for i in range(n_pw):
        for j in range(n_pw):
            dx = int(Gx_flat[i] - Gx_flat[j])
            dy = int(Gy_flat[i] - Gy_flat[j])
            if -N <= dx <= N and -N <= dy <= N:
                c_mat[i, j] = inv_eps_fft[N + dx, N + dy]

    # k-points in irreducible BZ (units of 2pi/a, so Gamma=(0,0), X=(0.5,0), M=(0.5,0.5))
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.35, 0.15),
        (0.4, 0.3),
        (0.45, 0.45),
        (0.15, 0.05),
        (0.3, 0.1),
    ]

    max_omega4 = 0.0
    min_omega5 = float('inf')

    for kx, ky in k_points:
        # k in actual units: multiply by 2pi
        kx_a = 2 * np.pi * kx
        ky_a = 2 * np.pi * ky

        kx_diff = kx_a - 2 * np.pi * Gx_flat
        ky_diff = ky_a - 2 * np.pi * Gy_flat
        dot_prod = np.outer(kx_diff, kx_diff) + np.outer(ky_diff, ky_diff)

        mat = dot_prod * c_mat
        mat = (mat + mat.T.conj()) / 2  # Hermitian

        try:
            eigvals = np.linalg.eigvalsh(mat)
            eigvals = np.sort(np.maximum(eigvals, 0))
            omegas = np.sqrt(eigvals)

            if len(omegas) >= 5:
                o4 = omegas[3]
                o5 = omegas[4]
                max_omega4 = max(max_omega4, o4)
                min_omega5 = min(min_omega5, o5)
        except:
            continue

    if min_omega5 == float('inf') or max_omega4 == 0:
        return -1.0

    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2
    if midgap <= 0:
        return -1.0
    return gap / midgap


def _local_search(cell, size, eps_low, eps_high, lower_band, deadline):
    best = [row[:] for row in cell]
    best_score = _evaluate_gap(best, size, eps_low, eps_high, lower_band)
    c = np.array(best, dtype=int)

    np.random.seed(42)
    for iteration in range(100):
        if time.time() > deadline:
            break

        i = np.random.randint(0, size)
        j = np.random.randint(0, size)

        orbit = set()
        for (oi, oj) in [(i, j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j),
                         (j, i), (size-1-j, i), (j, size-1-i), (size-1-j, size-1-i)]:
            orbit.add((oi, oj))

        new_c = c.copy()
        old_val = c[i, j]
        for (oi, oj) in orbit:
            new_c[oi, oj] = 1 - old_val

        new_cell = new_c.tolist()
        score = _evaluate_gap(new_cell, size, eps_low, eps_high, lower_band)

        if score > best_score:
            best_score = score
            best = [row[:] for row in new_cell]
            c = new_c

    return best
# EVOLVE-BLOCK-END
