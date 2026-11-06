"""Photonic crystal unit cell design for large TM 4-5 band gap."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 unit cell maximizing TM 4-5 band gap.
    
    Uses moiré interference of counter-rotating square-wave gratings
    to generate complex high-symmetry microstructures, evaluated with
    a fast plane-wave TM solver.
    """
    t_start = time.time()
    DEADLINE = 210.0  # safety margin before 230s kill

    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]

    # --- Fast TM plane-wave solver ---

    def compute_eps_inv_fft(cell):
        """Compute normalized Fourier coefficients of 1/eps."""
        f = cell.astype(np.float64)
        eps_inv = np.where(f > 0.5, 1.0 / eps_high, 1.0 / eps_low)
        F = np.fft.fft2(eps_inv)
        F /= (size * size)
        return F

    def setup_plane_waves(npw):
        """Generate G-vectors within circular cutoff."""
        G_list = []
        for gx in range(-npw, npw + 1):
            for gy in range(-npw, npw + 1):
                if gx * gx + gy * gy <= npw * npw:
                    G_list.append((gx, gy))
        return np.array(G_list, dtype=np.float64)

    def tm_eigenvalues_fast(G_arr, eps_inv_F, kx, ky):
        """Solve TM eigenvalue problem using vectorized matrix construction."""
        n = len(G_arr)
        kxG = kx + G_arr[:, 0]
        kyG = ky + G_arr[:, 1]
        kG_sq = kxG ** 2 + kyG ** 2

        # Build matrix A[i,j] = eps_inv_F[G_i - G_j] * (k+G_j)^2
        # Vectorized: compute all differences
        diff_x = G_arr[:, 0:1] - G_arr[0:1, :]  # n x n
        diff_y = G_arr[:, 1:2] - G_arr[1:2, :]  # n x n

        # Map to FFT indices
        idx_x = (diff_x.astype(int)) % size
        idx_y = (diff_y.astype(int)) % size

        eps_mat = eps_inv_F[idx_x, idx_y].real
        M = eps_mat * kG_sq[None, :]  # broadcast (k+G_j)^2 across rows

        eigvals = np.linalg.eigvalsh(M)
        return eigvals

    def compute_band_ratio(cell, G_arr, n_bands=6, n_kpts=15):
        """Compute gap-midgap ratio for TM bands 4-5 over irreducible BZ."""
        eps_inv_F = compute_eps_inv_fft(cell)

        # Irreducible BZ: triangle (0,0)-(0.5,0)-(0.5,0.5)
        k_points = []
        # Along X direction: (0,0) to (0.5, 0)
        for t in np.linspace(0, 0.5, 5):
            k_points.append((t, 0.0))
        # Along M direction: (0.5, 0) to (0.5, 0.5)
        for t in np.linspace(0, 0.5, 5):
            k_points.append((0.5, t))
        # Along X-M diagonal: (0,0) to (0.5, 0.5)
        for t in np.linspace(0, 0.5, 5):
            k_points.append((t, t))

        min_omega_5 = np.inf
        max_omega_4 = -np.inf

        for kx, ky in k_points:
            eigvals = tm_eigenvalues_fast(G_arr, eps_inv_F, kx, ky)
            omegas = np.sqrt(np.maximum(eigvals, 0))
            if len(omegas) >= n_bands:
                omega_4 = omegas[3]
                omega_5 = omegas[4]
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)

        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap <= 0:
            return -1.0
        return 2.0 * gap / midgap

    # --- Symmetry enforcement ---

    def apply_symmetry(cell):
        """Enforce full D4 symmetry (row-reverse, col-reverse, transpose)."""
        cell = cell.copy()
        # Row-reverse
        for i in range(size // 2):
            for j in range(size):
                avg = (cell[i, j] + cell[size - 1 - i, j]) / 2.0
                cell[i, j] = cell[size - 1 - i, j] = 1 if avg >= 0.5 else 0
        # Col-reverse
        for i in range(size):
            for j in range(size // 2):
                avg = (cell[i, j] + cell[i, size - 1 - j]) / 2.0
                cell[i, j] = cell[i, size - 1 - j] = 1 if avg >= 0.5 else 0
        # Transpose
        for i in range(size):
            for j in range(i + 1, size):
                avg = (cell[i, j] + cell[j, i]) / 2.0
                cell[i, j] = cell[j, i] = 1 if avg >= 0.5 else 0
        return cell

    # --- Moiré-based cell generation ---

    def make_moire_cell(f1, f2, phase1, phase2, threshold):
        """Generate cell from interference of two counter-rotating square waves.
        
        f1, f2: spatial frequencies of the two gratings
        phase1, phase2: phase offsets
        threshold: binarization threshold
        """
        cell = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                # First grating: square wave along direction (1,1)
                v1 = np.sin(2 * np.pi * f1 * (x + y) + phase1)
                sq1 = 1.0 if v1 > 0 else 0.0
                # Second grating: square wave along direction (1,-1)
                v2 = np.sin(2 * np.pi * f2 * (x - y) + phase2)
                sq2 = 1.0 if v2 > 0 else 0.0
                # Moiré: product creates complex pattern
                val = sq1 * sq2
                if val >= threshold:
                    cell[i, j] = 1
        return apply_symmetry(cell)

    def make_moire_plus_ring_cell(f1, f2, phase1, phase2, threshold, r_inner, r_outer):
        """Moiré pattern combined with central ring."""
        cell = make_moire_cell(f1, f2, phase1, phase2, threshold)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r = np.sqrt(x * x + y * y)
                if r_inner < r < r_outer:
                    cell[i, j] = 1
        return apply_symmetry(cell)

    def make_four_rod_cell(radius, offset):
        """Four dielectric rods at (±offset, ±offset)."""
        cell = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for dx, dy in [(offset, offset), (offset, -offset),
                               (-offset, offset), (-offset, -offset)]:
                    if (x - dx) ** 2 + (y - dy) ** 2 < radius ** 2:
                        cell[i, j] = 1
                        break
        return apply_symmetry(cell)

    def mutate_cell(cell, n_mutations=3):
        """Randomly flip symmetric pixel orbits."""
        cell = cell.copy()
        for _ in range(n_mutations):
            i = random.randint(0, size // 2 - 1)
            j = random.randint(0, size // 2 - 1)
            orbit = [(i, j), (i, size - 1 - j), (size - 1 - i, j),
                     (size - 1 - i, size - 1 - j),
                     (j, i), (j, size - 1 - i), (size - 1 - i, j),
                     (size - 1 - i, size - 1 - i)]
            new_val = 1 - cell[i, j]
            for (oi, oj) in orbit:
                cell[oi, oj] = new_val
        return cell

    # --- Main search ---

    # Use npw=4 for balance of accuracy and speed (49 G-vectors)
    G_arr = setup_plane_waves(4)

    best_score = -np.inf
    best_cell = None

    def evaluate_cell(cell):
        nonlocal best_score, best_cell
        if time.time() > DEADLINE:
            return -np.inf
        score = compute_band_ratio(cell, G_arr)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
        return score

    # Phase 1: Evaluate moiré family
    print(f"Phase 1: Moiré family search... (G-vectors: {len(G_arr)})")

    for f1 in [2, 3, 4, 5, 6]:
        if time.time() > DEADLINE:
            break
        for f2 in [2, 3, 4, 5, 6]:
            if time.time() > DEADLINE:
                break
            if f1 == f2:
                continue
            for p1 in [0, np.pi / 2, np.pi]:
                if time.time() > DEADLINE:
                    break
                for p2 in [0, np.pi / 2, np.pi]:
                    if time.time() > DEADLINE:
                        break
                    for thr in [0.5, 0.7]:
                        if time.time() > DEADLINE:
                            break
                        cell = make_moire_cell(f1, f2, p1, p2, thr)
                        evaluate_cell(cell)

    # Phase 2: Evaluate moiré + ring
    if time.time() < DEADLINE - 40:
        print(f"Phase 2: Moiré + ring...")
        for f1 in [3, 4, 5]:
            if time.time() > DEADLINE:
                break
            for f2 in [3, 4, 5]:
                if time.time() > DEADLINE:
                    break
                if f1 == f2:
                    continue
                for p1 in [0, np.pi / 2]:
                    if time.time() > DEADLINE:
                        break
                    for p2 in [0, np.pi / 2]:
                        if time.time() > DEADLINE:
                            break
                        for ri in [0.06, 0.10]:
                            if time.time() > DEADLINE:
                                break
                            for ro in [0.20, 0.28]:
                                if time.time() > DEADLINE:
                                    break
                                cell = make_moire_plus_ring_cell(f1, f2, p1, p2, 0.5, ri, ro)
                                evaluate_cell(cell)

    # Phase 3: Four-rod family as backup
    if time.time() < DEADLINE - 30:
        print(f"Phase 3: Four-rod family...")
        for r in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]:
            if time.time() > DEADLINE:
                break
            for d in [0.08, 0.12, 0.16, 0.20, 0.24]:
                if time.time() > DEADLINE:
                    break
                cell = make_four_rod_cell(r, d)
                evaluate_cell(cell)

    # Phase 4: Local search around best
    if best_cell is not None and time.time() < DEADLINE - 20:
        print(f"Phase 4: Local search (best={best_score:.4f})...")
        current = best_cell.copy()
        current_score = best_score

        for iteration in range(300):
            if time.time() > DEADLINE:
                break
            mutated = mutate_cell(current, n_mutations=random.randint(1, 3))
            score = evaluate_cell(mutated)
            if score > current_score:
                current = mutated
                current_score = score

    # Fallback
    if best_cell is None:
        best_cell = make_four_rod_cell(0.12, 0.16)

    best_cell = apply_symmetry(best_cell)
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    return result
# EVOLVE-BLOCK-END
