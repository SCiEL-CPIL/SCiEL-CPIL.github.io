"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def design(problem):
    size = problem["pixels"]  # 32
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    t0 = time.time()
    deadline = 200.0  # Safety margin before 230s kill

    def enforce_d4(cell):
        """Enforce D4 symmetry by averaging orbits."""
        n = len(cell)
        cell_new = [[0] * n for _ in range(n)]
        visited = set()
        for i in range(n):
            for j in range(n):
                if (i, j) in visited:
                    continue
                orbit = [
                    (i, j),
                    (n - 1 - i, j),
                    (i, n - 1 - j),
                    (n - 1 - i, n - 1 - j),
                    (j, i),
                    (n - 1 - j, i),
                    (j, n - 1 - i),
                    (n - 1 - j, n - 1 - i),
                ]
                unique_orbit = list(set(orbit))
                val = sum(cell[p[0]][p[1]] for p in unique_orbit) / len(unique_orbit)
                val_int = 1 if val > 0.5 else 0
                for p in unique_orbit:
                    cell_new[p[0]][p[1]] = val_int
                    visited.add(p)
        return cell_new

    def compute_eps_inv_fourier(cell):
        """Compute Fourier coefficients of 1/epsilon using FFT."""
        eps_map = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if cell[i][j] == 1:
                    eps_map[i, j] = eps_high
                else:
                    eps_map[i, j] = eps_low

        inv_eps_map = 1.0 / eps_map
        # Compute 2D FFT
        fft_vals = np.fft.fft2(inv_eps_map)
        # Normalize
        fft_vals = fft_vals / (size * size)
        return fft_vals

    def get_eps_G(fft_vals, gx, gy):
        """Get the Fourier coefficient for G = (2pi/a)*(gx, gy) from FFT values."""
        # FFT index for gx, gy: the FFT gives coefficients for frequencies k = 0, 1, ..., N-1
        # which correspond to G = 2pi*(k/N) in each dimension
        # We want G = 2pi*(gx, gy), so we need k such that k/N = gx mod 1
        # Since gx is integer, k = gx mod N
        # But FFT indexing: index k corresponds to frequency k (if k <= N/2) or k-N (if k > N/2)
        # So for integer gx, we want the coefficient at frequency gx (which is in range [-N/2, N/2] for typical gx)
        # Actually, our G-vectors are integer (gx, gy), so we need coefficients at those integer frequencies
        # FFT[0,0] = G=(0,0)
        # FFT[1,0] = G=(1,0) [if we interpret correctly]
        # Actually, np.fft.fft2 gives:
        # result[m,n] = sum_{p,q} f[p,q] * exp(-2j*pi*(m*p/N + n*q/N))
        # So result[m,n] corresponds to G = (2pi*m/N, 2pi*n/N)
        # We want G = (2pi*gx, 2pi*gy) where gx, gy are integers
        # So we need m/N = gx mod 1 => m = gx mod N (since gx is integer, m = gx if gx in [0,N), else gx mod N)
        # But note: for negative gx, m should be gx mod N = N + gx (if gx < 0)
        # Let's just use: m = gx % size, n = gy % size
        m = gx % size
        n = gy % size
        return fft_vals[m, n]

    def solve_bands_at_k(kx, ky, fft_vals, G_order=4):
        """
        Solve for TM bands at a specific k-point.
        Uses plane-wave expansion with G-vectors up to order G_order.
        Returns sorted list of omega values.
        """
        # Generate G vectors: (gx, gy) where |gx|, |gy| <= G_order
        G_vectors = []
        for gx in range(-G_order, G_order + 1):
            for gy in range(-G_order, G_order + 1):
                # Only include if within a circle of radius G_order (circular truncation)
                if gx * gx + gy * gy <= G_order * G_order:
                    G_vectors.append((gx, gy))
        N_G = len(G_vectors)

        # Build the eigenvalue matrix
        # A[i,j] = (K + G_i)^2 * delta_ij - (1/eps)_{G_i - G_j}
        # where (1/eps)_G is the Fourier coefficient
        A = np.zeros((N_G, N_G), dtype=complex)

        for i in range(N_G):
            Gi = G_vectors[i]
            kx_i = kx + 2 * np.pi * Gi[0]
            ky_i = ky + 2 * np.pi * Gi[1]
            term1 = kx_i**2 + ky_i**2

            for j in range(N_G):
                Gj = G_vectors[j]
                G_diff = (Gi[0] - Gj[0], Gi[1] - Gj[1])
                # Get (1/eps)_{G_diff}
                term2 = get_eps_G(fft_vals, G_diff[0], G_diff[1])

                if i == j:
                    A[i, j] = term1 - term2
                else:
                    A[i, j] = -term2

        # Solve eigenvalue problem
        eigenvalues = np.linalg.eigvalsh(A)
        eigenvalues = np.where(eigenvalues > 1e-10, eigenvalues, 0)
        omegas = np.sqrt(eigenvalues)
        return np.sort(omegas)

    def evaluate_cell(cell):
        """
        Calculate the gap-midgap ratio for bands 4 and 5 (1-indexed).
        """
        fft_vals = compute_eps_inv_fourier(cell)

        # Key k-points in the irreducible Brillouin zone
        # Square BZ: [-pi, pi] x [-pi, pi]
        # Irreducible region: 0 <= kx <= ky <= pi
        # High symmetry points and some interior points
        k_points = [
            (0.0, 0.0),           # Gamma
            (np.pi, 0.0),         # M (or X)
            (np.pi, np.pi),       # Corner
            (np.pi/2, 0.0),       # Midpoint
            (np.pi/2, np.pi/2),   # Center of BZ edge
            (np.pi/4, np.pi/4),   # Interior
            (np.pi/3, np.pi/3),   # Interior
            (np.pi/2, np.pi/4),   # Interior
            (np.pi/4, 0.0),       # Interior
            (3*np.pi/4, np.pi/4), # Interior
            (np.pi/6, np.pi/2),   # Interior
            (np.pi/3, 0.0),       # Interior
        ]

        min_omega_5 = float('inf')
        max_omega_4 = 0.0

        for kx, ky in k_points:
            omegas = solve_bands_at_k(kx, ky, fft_vals)
            if len(omegas) >= 5:
                w4 = omegas[3]  # 4th band (0-indexed)
                w5 = omegas[4]  # 5th band (0-indexed)
                if w5 < min_omega_5:
                    min_omega_5 = w5
                if w4 > max_omega_4:
                    max_omega_4 = w4

        if min_omega_5 == float('inf') or max_omega_4 == 0.0:
            return -1.0

        gap = 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap

    # Initial design: air background with dielectric rods in a 2x2 supercell arrangement
    # This is a common structure for opening higher-order band gaps
    cell = [[0] * size for _ in range(size)]

    # Place 4 rods centered at (1/4, 1/4), (1/4, 3/4), (3/4, 1/4), (3/4, 3/4)
    rod_radius = 5  # Radius in pixels
    centers = [
        (size // 4, size // 4),
        (size // 4, 3 * size // 4),
        (3 * size // 4, size // 4),
        (3 * size // 4, 3 * size // 4),
    ]

    for i in range(size):
        for j in range(size):
            for cx, cy in centers:
                dist_sq = (i - cx)**2 + (j - cy)**2
                if dist_sq <= rod_radius**2:
                    cell[i][j] = 1
                    break

    cell = enforce_d4(cell)

    # Evaluate initial
    current_score = evaluate_cell(cell)
    best_cell = [row[:] for row in cell]
    best_score = current_score

    # Local search: flip orbits
    def flip_orbit(cell, i, j):
        n = len(cell)
        orbit = [
            (i, j),
            (n - 1 - i, j),
            (i, n - 1 - j),
            (n - 1 - i, n - 1 - j),
            (j, i),
            (n - 1 - j, i),
            (j, n - 1 - i),
            (n - 1 - j, n - 1 - i),
        ]
        unique_orbit = list(set(orbit))
        for p in unique_orbit:
            cell[p[0]][p[1]] = 1 - cell[p[0]][p[1]]
        return cell

    # Hill climbing with restarts
    max_evals = 150
    eval_count = 0

    while time.time() - t0 < deadline and eval_count < max_evals:
        # Try random flips
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)

        new_cell = [row[:] for row in cell]
        flip_orbit(new_cell, i, j)

        new_score = evaluate_cell(new_cell)
        eval_count += 1

        if new_score > current_score:
            cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = [row[:] for row in cell]
        else:
            # Occasionally restart from best
            if eval_count % 20 == 0 and best_score > -0.5:
                cell = [row[:] for row in best_cell]
                current_score = best_score

    return best_cell
# EVOLVE-BLOCK-END
