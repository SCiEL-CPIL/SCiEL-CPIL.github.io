"""Design a 32x32 symmetric photonic crystal unit cell for maximum TM band gap."""
import time
import math
import numpy as np

_DEADLINE = 215.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _enforce_d4(cell):
    """Enforce full D4 symmetry on the cell."""
    size = len(cell)
    result = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            positions = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, i),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i),
            ]
            vals = [cell[pi][pj] for pi, pj in positions]
            result[i][j] = 1 if sum(vals) >= 4 else 0
    return result


def _make_cell_disk(r):
    size = 32
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < r * r:
                cell[i][j] = 1
    return _enforce_d4(cell)


def _make_cell_ring(r_in, r_out):
    size = 32
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = math.sqrt(x * x + y * y)
            if r_in <= r < r_out:
                cell[i][j] = 1
    return _enforce_d4(cell)


def _make_cell_cross(bar_w, cross_l):
    size = 32
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < bar_w / 2.0 and abs(y) < cross_l / 2.0:
                cell[i][j] = 1
            if abs(y) < bar_w / 2.0 and abs(x) < cross_l / 2.0:
                cell[i][j] = 1
    return _enforce_d4(cell)


def _compute_tm_gap_fast(cell, N_pw=5, n_kpts=8):
    """
    Fast TM band gap computation using a small plane-wave basis.
    """
    size = len(cell)
    eps_high = 12.0
    eps_low = 1.0

    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            inv_eps[i][j] = 1.0 / (eps_high if cell[i][j] == 1 else eps_low)

    fft_coeffs = np.fft.fft2(inv_eps)
    N = size

    def get_coeff(m, n):
        idx_m = m % N
        idx_n = n % N
        return fft_coeffs[idx_m, idx_n] / (N * N)

    g_indices = []
    for m in range(-N_pw + 1, N_pw):
        for n in range(-N_pw + 1, N_pw):
            g_indices.append((m, n))

    npw = len(g_indices)

    S = np.zeros((npw, npw), dtype=float)
    for i, (mi, ni) in enumerate(g_indices):
        for j, (mj, nj) in enumerate(g_indices):
            gm = mi - mj
            gn = ni - nj
            S[i, j] = get_coeff(gm, gn).real

    # k-points in the irreducible BZ (coarse)
    kpts = [
        (0.0, 0.0),
        (0.25, 0.0),
        (0.5, 0.0),
        (0.75, 0.0),
        (1.0, 0.0),
        (0.25, 0.25),
        (0.5, 0.5),
        (0.75, 0.25),
    ]

    max_omega4 = 0.0
    min_omega5 = float('inf')

    for kx, ky in kpts:
        km2 = np.zeros(npw)
        for idx, (m, n) in enumerate(g_indices):
            km2[idx] = (2.0 * m + kx) ** 2 + (2.0 * n + ky) ** 2

        A = S * km2[np.newaxis, :]

        try:
            eigenvalues = np.linalg.eigvalsh(A)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = np.sort(eigenvalues)

            if len(eigenvalues) >= 5:
                omega4_sq = eigenvalues[3]
                omega5_sq = eigenvalues[4]

                if omega4_sq > 0 and omega5_sq > 0:
                    omega4 = np.sqrt(omega4_sq)
                    omega5 = np.sqrt(omega5_sq)

                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
        except:
            continue

    if min_omega5 == float('inf') or max_omega4 == 0:
        return 0.0

    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0

    if midgap <= 0:
        return 0.0

    return 2.0 * gap / midgap


def _compute_tm_gap_full(cell, N_pw=17, n_kpts=15):
    """
    Full TM band gap computation using a larger plane-wave basis.
    """
    size = len(cell)
    eps_high = 12.0
    eps_low = 1.0

    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            inv_eps[i][j] = 1.0 / (eps_high if cell[i][j] == 1 else eps_low)

    fft_coeffs = np.fft.fft2(inv_eps)
    N = size

    def get_coeff(m, n):
        idx_m = m % N
        idx_n = n % N
        return fft_coeffs[idx_m, idx_n] / (N * N)

    g_indices = []
    for m in range(-N_pw + 1, N_pw):
        for n in range(-N_pw + 1, N_pw):
            g_indices.append((m, n))

    npw = len(g_indices)

    S = np.zeros((npw, npw), dtype=float)
    for i, (mi, ni) in enumerate(g_indices):
        for j, (mj, nj) in enumerate(g_indices):
            gm = mi - mj
            gn = ni - nj
            S[i, j] = get_coeff(gm, gn).real

    # k-points in the irreducible BZ
    kpts = []
    # Along kx axis (Gamma to X)
    for i in range(n_kpts // 3 + 1):
        kx = i / (n_kpts // 3) * 1.0
        kpts.append((kx, 0.0))
    # Along kx=ky diagonal (Gamma to M)
    for i in range(1, n_kpts // 3 + 1):
        kx = i / (n_kpts // 3) * 0.5
        kpts.append((kx, kx))
    # Along the line kx+ky=1 (X to M)
    for i in range(1, n_kpts // 3):
        kx = 1.0 - i / (n_kpts // 3) * 0.5
        ky = i / (n_kpts // 3) * 0.5
        if ky >= 0 and kx >= ky:
            kpts.append((kx, ky))
    # Add some interior points
    kpts.append((0.5, 0.25))
    kpts.append((0.75, 0.25))
    kpts.append((0.6, 0.3))

    # Remove duplicates
    unique_kpts = []
    seen = set()
    for kx, ky in kpts:
        if ky < 0:
            ky = -ky
        if kx < ky:
            kx, ky = ky, kx
        if kx + ky > 1.0:
            scale = 1.0 / (kx + ky)
            kx *= scale
            ky *= scale
        key = (round(kx, 6), round(ky, 6))
        if key not in seen:
            seen.add(key)
            unique_kpts.append((kx, ky))

    kpts = unique_kpts[:n_kpts]

    max_omega4 = 0.0
    min_omega5 = float('inf')

    for kx, ky in kpts:
        km2 = np.zeros(npw)
        for idx, (m, n) in enumerate(g_indices):
            km2[idx] = (2.0 * m + kx) ** 2 + (2.0 * n + ky) ** 2

        A = S * km2[np.newaxis, :]

        try:
            eigenvalues = np.linalg.eigvalsh(A)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = np.sort(eigenvalues)

            if len(eigenvalues) >= 5:
                omega4_sq = eigenvalues[3]
                omega5_sq = eigenvalues[4]

                if omega4_sq > 0 and omega5_sq > 0:
                    omega4 = np.sqrt(omega4_sq)
                    omega5 = np.sqrt(omega5_sq)

                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
        except:
            continue

    if min_omega5 == float('inf') or max_omega4 == 0:
        return 0.0

    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0

    if midgap <= 0:
        return 0.0

    return 2.0 * gap / midgap


def _evolve_population(population, scores, n_gen=30, n_offspring=10, n_elite=5):
    """
    Simple evolutionary algorithm to optimize the cell.
    """
    best_idx = np.argmax(scores)
    best_cell = population[best_idx]
    best_score = scores[best_idx]

    for gen in range(n_gen):
        if _remaining() < 30:
            break

        # Create offspring
        offspring = []
        offspring_scores = []

        # Keep elite
        elite_indices = np.argsort(scores)[-n_elite:]
        for idx in elite_indices:
            offspring.append(population[idx])
            offspring_scores.append(scores[idx])

        # Generate new offspring via mutation
        while len(offspring) < n_offspring:
            if _remaining() < 30:
                break

            # Pick a parent (weighted by score)
            parent_idx = np.random.choice(len(population), p=np.exp(scores / 0.01) / np.sum(np.exp(scores / 0.01)))
            parent = population[parent_idx]

            # Mutate: flip a few random pixels
            cell_arr = np.array(parent, dtype=int)
            n_flips = np.random.randint(1, 5)
            for _ in range(n_flips):
                i = np.random.randint(0, 32)
                j = np.random.randint(0, 32)
                cell_arr[i, j] = 1 - cell_arr[i, j]

            # Enforce D4 symmetry
            new_cell = _enforce_d4(cell_arr.tolist())

            # Evaluate with fast method
            score = _compute_tm_gap_fast(new_cell, N_pw=5, n_kpts=8)

            offspring.append(new_cell)
            offspring_scores.append(score)

        # Update population
        population = offspring
        scores = np.array(offspring_scores)

        # Update best
        best_idx = np.argmax(scores)
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_cell = population[best_idx]

    return best_cell, best_score


def design(problem):
    """
    Design a 32x32 symmetric photonic crystal unit cell.
    Returns a nested list of 0/1 pixels.
    """
    global _start_time
    _start_time = time.time()

    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    polarization = problem.get("polarization", "TM")

    best_cell = None
    best_score = -float('inf')

    # Phase 1: Generate diverse initial candidates
    candidates = []

    # Central disks
    for r in [0.1, 0.15, 0.2, 0.25, 0.3]:
        candidates.append(_make_cell_disk(r))

    # Rings
    for r_in, r_out in [(0.1, 0.2), (0.15, 0.25), (0.2, 0.3), (0.25, 0.35), (0.3, 0.4)]:
        candidates.append(_make_cell_ring(r_in, r_out))

    # Cross shapes
    for bar_w, cross_l in [(0.1, 0.5), (0.15, 0.6), (0.2, 0.7), (0.1, 0.8), (0.2, 0.9)]:
        candidates.append(_make_cell_cross(bar_w, cross_l))

    # Score all candidates with fast method
    scores = []
    for cell in candidates:
        if _remaining() < 60:
            break
        score = _compute_tm_gap_fast(cell, N_pw=5, n_kpts=8)
        scores.append(score)

    scores = np.array(scores)

    # Phase 2: Evolutionary optimization
    if best_cell is None or _remaining() > 60:
        # Find best initial candidate
        best_idx = np.argmax(scores)
        best_cell = candidates[best_idx]
        best_score = scores[best_idx]

        # Run evolution
        evolved_cell, evolved_score = _evolve_population(candidates, scores, n_gen=20, n_offspring=15, n_elite=5)
        if evolved_score > best_score:
            best_score = evolved_score
            best_cell = evolved_cell

    # Phase 3: Refine best candidate with full method
    if best_cell is not None and _remaining() > 30:
        refined_score = _compute_tm_gap_full(best_cell, N_pw=17, n_kpts=15)
        if refined_score > best_score:
            best_score = refined_score

    # Fallback
    if best_cell is None:
        best_cell = _make_cell_disk(0.2)

    return best_cell
