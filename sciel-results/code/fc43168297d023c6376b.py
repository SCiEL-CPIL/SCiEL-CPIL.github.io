"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def design(problem):
    """
    Design a 32x32 unit cell using optimized Fourier-harmonic superposition
    with a plane-wave proxy model for local refinement.
    """
    start_time = time.time()
    deadline = 180  # Safety margin before 230s kill

    size = problem["pixels"]
    N = size
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)

    # Coordinate grids centered at 0
    x = np.arange(N) - N / 2.0 + 0.5
    y = np.arange(N) - N / 2.0 + 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    X_norm = X / (N / 2.0)
    Y_norm = Y / (N / 2.0)

    def enforce_d4(cell):
        """Enforce D4 symmetry by averaging over all 8 symmetry operations."""
        c = np.array(cell, dtype=float)
        sym = c.copy()
        sym += np.rot90(c, 1)
        sym += np.rot90(c, 2)
        sym += np.rot90(c, 3)
        sym += np.fliplr(c)
        sym += np.flipud(c)
        sym += np.fliplr(np.flipud(c))
        sym += c.T
        avg = sym / 8.0
        return (avg >= 0.5).astype(int)

    def generate_cell(params):
        """
        Generate a binary cell from harmonic parameters.
        params: list of (amp, phase, mx, my) tuples
        """
        field = np.zeros((N, N))
        for amp, phase, mx, my in params:
            field += amp * np.cos(2 * np.pi * (mx * X_norm + my * Y_norm) + phase)
        threshold = np.median(field)
        binary = (field > threshold).astype(int)
        binary = enforce_d4(binary)
        return binary

    def compute_proxy_gap(cell, max_order=6, g_range=4, n_k=15):
        """
        Compute approximate gap ratio between bands 4 and 5 using plane-wave method.
        Returns gap-midgap ratio (2*(min_w5 - max_w4)/(min_w5 + max_w4)).
        """
        eps = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                eps[i, j] = eps_high if cell[i][j] == 1 else eps_low
        inv_eps = 1.0 / eps

        # FFT for Fourier coefficients
        fft_2d = np.fft.fft2(inv_eps)

        def get_coeff(m, n):
            m_idx = m % N
            n_idx = n % N
            return fft_2d[m_idx, n_idx] / (N * N)

        # G-vectors
        g_vectors = [(m, n) for m in range(-g_range, g_range + 1)
                     for n in range(-g_range, g_range + 1)]
        Npw = len(g_vectors)

        # K-points in irreducible BZ (square: triangle from (0,0) to (0.5,0) to (0.5,0.5))
        k_points = []
        # Along x-axis
        for t in np.linspace(0, 0.5, n_k):
            k_points.append((t, 0.0))
        # Along diagonal
        for t in np.linspace(0, 0.5, n_k):
            k_points.append((t, t))
        # Along y-axis (symmetric to x, but include for completeness)
        for t in np.linspace(0, 0.5, n_k):
            k_points.append((0.0, t))

        min_omega_5 = float('inf')
        max_omega_4 = 0.0

        for kx, ky in k_points:
            H = np.zeros((Npw, Npw), dtype=complex)
            for i, Gp in enumerate(g_vectors):
                kx_t = kx + Gp[0]
                ky_t = ky + Gp[1]
                k_sq = kx_t ** 2 + ky_t ** 2
                for j, G in enumerate(g_vectors):
                    dm = Gp[0] - G[0]
                    dn = Gp[1] - G[1]
                    if abs(dm) <= max_order and abs(dn) <= max_order:
                        coeff = get_coeff(dm, dn)
                    else:
                        coeff = 0.0
                    H[i, j] = k_sq * coeff

            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)

            if len(eigvals) >= 5:
                w4 = eigvals[3]
                w5 = eigvals[4]
                if w5 < min_omega_5:
                    min_omega_5 = w5
                if w4 > max_omega_4:
                    max_omega_4 = w4

        if min_omega_5 == float('inf') or max_omega_4 <= 0:
            return -1.0

        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return -1.0
        return 2.0 * gap / midgap

    # Initial harmonic parameters: (amp, phase, mx, my)
    # Include harmonics important for band 4-5 coupling in square lattice
    initial_params = [
        (0.8, 0.0, 2, 0),    # (2,0)
        (0.8, 0.0, 0, 2),    # (0,2)
        (0.5, np.pi / 4, 2, 2),  # (2,2)
        (0.3, np.pi / 2, 4, 0),  # (4,0)
        (0.3, np.pi / 2, 0, 4),  # (0,4)
        (0.2, 0.0, 4, 2),    # (4,2)
        (0.2, 0.0, 2, 4),    # (2,4)
        (0.15, np.pi / 2, 4, 4),  # (4,4)
        (0.1, 0.0, 6, 0),    # (6,0)
        (0.1, 0.0, 0, 6),    # (0,6)
    ]

    # Generate initial cell
    best_cell = generate_cell(initial_params)
    best_score = compute_proxy_gap(best_cell)

    if time.time() - start_time > deadline:
        return [[int(best_cell[i][j]) for j in range(N)] for i in range(N)]

    # Local optimization: perturb amplitudes and phases
    rng = np.random.default_rng(42)
    n_params = len(initial_params)
    current_params = [list(p) for p in initial_params]
    current_cell = best_cell
    current_score = best_score

    # Coordinate descent on amplitudes and phases
    for iteration in range(80):
        if time.time() - start_time > deadline:
            break

        improved = False
        for p_idx in range(n_params):
            # Try perturbing amplitude
            for delta_a in [-0.05, 0.05, -0.1, 0.1, -0.2, 0.2]:
                trial = [list(p) for p in current_params]
                trial[p_idx][0] = max(0.0, trial[p_idx][0] + delta_a)
                trial_cell = generate_cell(trial)
                trial_score = compute_proxy_gap(trial_cell)
                if trial_score > current_score + 1e-6:
                    current_params = trial
                    current_cell = trial_cell
                    current_score = trial_score
                    improved = True
                    break

            # Try perturbing phase
            for delta_p in [-0.1, 0.1, -0.2, 0.2, -0.5, 0.5]:
                trial = [list(p) for p in current_params]
                trial[p_idx][1] = trial[p_idx][1] + delta_p
                trial_cell = generate_cell(trial)
                trial_score = compute_proxy_gap(trial_cell)
                if trial_score > current_score + 1e-6:
                    current_params = trial
                    current_cell = trial_cell
                    current_score = trial_score
                    improved = True
                    break

        if not improved:
            # Try random perturbation
            trial = [list(p) for p in current_params]
            for p_idx in range(n_params):
                trial[p_idx][0] = max(0.0, trial[p_idx][0] + rng.normal(0, 0.1))
                trial[p_idx][1] += rng.normal(0, 0.2)
            trial_cell = generate_cell(trial)
            trial_score = compute_proxy_gap(trial_cell)
            if trial_score > current_score + 1e-6:
                current_params = trial
                current_cell = trial_cell
                current_score = trial_score
                improved = True

        if not improved:
            # Try adding a new harmonic
            new_harmonics = [
                (0.1, 0.0, 6, 2),
                (0.1, 0.0, 2, 6),
                (0.1, 0.0, 6, 4),
                (0.1, 0.0, 4, 6),
                (0.08, np.pi/4, 3, 1),
                (0.08, np.pi/4, 1, 3),
            ]
            added = False
            for nh in new_harmonics:
                if len(current_params) >= 15:
                    break
                trial = [list(p) for p in current_params] + [list(nh)]
                trial_cell = generate_cell(trial)
                trial_score = compute_proxy_gap(trial_cell)
                if trial_score > current_score + 1e-6:
                    current_params = trial
                    current_cell = trial_cell
                    current_score = trial_score
                    added = True
                    break
            if not added:
                break

        if current_score > best_score:
            best_score = current_score
            best_cell = current_cell

    # Final polish: try pixel-level flips in the most uncertain regions
    if time.time() - start_time < deadline - 30:
        # Identify pixels where the field is close to the threshold
        field = np.zeros((N, N))
        for amp, phase, mx, my in current_params:
            field += amp * np.cos(2 * np.pi * (mx * X_norm + my * Y_norm) + phase)
        threshold = np.median(field)
        uncertainty = np.abs(field - threshold)
        # Get the most uncertain pixels
        flat_idx = np.argsort(uncertainty.ravel())[:20]
        for idx in flat_idx:
            i, j = divmod(idx, N)
            # Try flipping this pixel and its D4 symmetric partners
            trial_cell = best_cell.copy()
            positions = [
                (i, j), (N-1-i, j), (i, N-1-j), (N-1-i, N-1-j),
                (j, i), (N-1-j, N-1-i), (j, N-1-i), (N-1-j, i)
            ]
            for pi, pj in positions:
                trial_cell[pi, pj] = 1 - trial_cell[pi, pj]
            trial_score = compute_proxy_gap(trial_cell)
            if trial_score > best_score + 1e-6:
                best_score = trial_score
                best_cell = trial_cell

    # Ensure D4 symmetry is exactly enforced
    best_cell = enforce_d4(best_cell)

    # Convert to nested list
    result = [[int(best_cell[i][j]) for j in range(N)] for i in range(N)]
    return result
# EVOLVE-BLOCK-END
