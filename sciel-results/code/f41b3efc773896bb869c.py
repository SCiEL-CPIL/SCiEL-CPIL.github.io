import numpy as np
import time

def design(problem):
    start_time = time.time()
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]  # 1
    eps_high = problem["eps_high"]  # 12
    deadline = 210  # seconds, safety margin before 230s kill

    def make_ring(r_inner, r_outer):
        """Create a ring-shaped dielectric region."""
        cell = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            y = (i + 0.5) / size - 0.5
            for j in range(size):
                x = (j + 0.5) / size - 0.5
                r2 = x * x + y * y
                if r_inner * r_inner < r2 < r_outer * r_outer:
                    cell[i, j] = 1
        return cell

    def make_disc(radius):
        """Create a solid disc of dielectric."""
        cell = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            y = (i + 0.5) / size - 0.5
            for j in range(size):
                x = (j + 0.5) / size - 0.5
                if x * x + y * y < radius * radius:
                    cell[i, j] = 1
        return cell

    def make_cross(warmth):
        """Create a cross/plus shape with variable arm width."""
        cell = np.zeros((size, size), dtype=np.int8)
        center = size // 2
        half_w = max(1, int(warmth * size / 2))
        for i in range(size):
            for j in range(size):
                if abs(i - center + 0.5) <= half_w or abs(j - center + 0.5) <= half_w:
                    cell[i, j] = 1
        return cell

    def make_concentric(r1, r2, r3):
        """Two concentric rings with different radii."""
        cell = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            y = (i + 0.5) / size - 0.5
            for j in range(size):
                x = (j + 0.5) / size - 0.5
                r2 = x * x + y * y
                # Inner disc
                if r2 < r1 * r1:
                    cell[i, j] = 1
                # Outer ring
                elif r2 * 2 * r2 < r3 * r3 * r3 and r2 > r2 * r2:
                    cell[i, j] = 1
        return cell

    def pwe_gap(cell, n_pw=8, k_points=None):
        """Compute gap-midgap ratio using PWE method. Returns ratio or 0 on failure."""
        try:
            # Build eps_inv grid
            eps_inv = np.ones((size, size))
            eps_inv[cell == 1] = 1.0 / eps_high
            fft_eps_inv = np.fft.fft2(eps_inv)

            # Plane wave grid
            n_max = n_pw
            m_vals = np.arange(-n_max, n_max + 1)
            n_vals = np.arange(-n_max, n_max + 1)
            M_grid, N_grid = np.meshgrid(m_vals, n_vals, indexing='ij')
            Gx = M_grid * 2 * np.pi / size
            Gy = N_grid * 2 * np.pi / size
            G_flat_x = Gx.ravel()
            G_flat_y = Gy.ravel()
            N_g = len(G_flat_x)

            # Precompute eps_inv for all difference vectors needed
            diff_range = 2 * n_max
            dm_vals = np.arange(-diff_range, diff_range + 1)
            dn_vals = np.arange(-diff_range, diff_range + 1)
            DM, DN = np.meshgrid(dm_vals, dn_vals, indexing='ij')
            eps_inv_diff = np.zeros((2 * diff_range + 1, 2 * diff_range + 1), dtype=complex)
            for idx_i, dm in enumerate(dm_vals):
                for idx_j, dn in enumerate(dn_vals):
                    u = dm % size
                    v = dn % size
                    eps_inv_diff[idx_i, idx_j] = fft_eps_inv[v, u]

            # K-points in irreducible BZ
            if k_points is None:
                k_points = []
                # Gamma to X: (k, 0) for k in [0, pi]
                for i in range(11):
                    k_points.append((np.pi * i / 10, 0))
                # X to M: (pi, k) for k in [0, pi]
                for i in range(1, 11):
                    k_points.append((np.pi, np.pi * i / 10))
                # M to Gamma: (k, k) for k in [0, pi]
                for i in range(1, 11):
                    k_points.append((np.pi * (1 - i / 10), np.pi * (1 - i / 10)))

            min_omega_5 = np.inf
            max_omega_4 = 0.0

            for kx, ky in k_points:
                k_vec = np.array([kx, ky])
                KG_x = k_vec[0] + G_flat_x
                KG_y = k_vec[1] + G_flat_y
                KG2 = KG_x ** 2 + KG_y ** 2

                # Build M matrix vectorized
                # M[i,j] = (k+G_i).(k+G_j) / eps_inv(G_i - G_j)
                KG_dot = np.outer(KG_x, KG_x) + np.outer(KG_y, KG_y)

                # Get differences
                M_idx = M_grid.ravel()
                N_idx = N_grid.ravel()
                diff_M = M_idx[:, None] - M_idx[None, :]
                diff_N = N_idx[:, None] - N_idx[None, :]

                # Look up eps_inv_diff
                denom = np.zeros((N_g, N_g), dtype=complex)
                for i in range(N_g):
                    for j in range(N_g):
                        dm = diff_M[i, j]
                        dn = diff_N[i, j]
                        denom[i, j] = eps_inv_diff[dm + diff_range, dn + diff_range]

                # Avoid division by zero
                denom[denom == 0] = 1e-10

                M_mat = KG_dot / denom

                # Solve eigenvalue problem
                w2 = np.linalg.eigvalsh(M_mat)
                w2 = np.sort(np.real(w2))

                if len(w2) >= 5:
                    omega_4 = np.sqrt(max(w2[3], 0))
                    omega_5 = np.sqrt(max(w2[4], 0))
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)

            if min_omega_5 == np.inf or max_omega_4 == 0:
                return 0.0
            gap = min_omega_5 - max_omega_4
            midgap = (min_omega_5 + max_omega_4) / 2
            if midgap <= 0:
                return 0.0
            return gap / midgap

        except Exception:
            return 0.0

    # Search over multiple shape families
    best_gap = -np.inf
    best_cell = None

    # Phase 1: Coarse search over different shape families
    # Try rings
    for r_in in [0.05, 0.1, 0.15, 0.2]:
        if time.time() > deadline:
            break
        for r_out in [0.2, 0.25, 0.3, 0.35, 0.4]:
            if time.time() > deadline:
                break
            if r_in >= r_out:
                continue
            cell = make_ring(r_in, r_out)
            gap = pwe_gap(cell, n_pw=6)
            if gap > best_gap:
                best_gap = gap
                best_cell = cell

    # Try discs
    for r in [0.15, 0.2, 0.25, 0.3, 0.35]:
        if time.time() > deadline:
            break
        cell = make_disc(r)
        gap = pwe_gap(cell, n_pw=6)
        if gap > best_gap:
            best_gap = gap
            best_cell = cell

    # Try crosses
    for w in [0.1, 0.15, 0.2, 0.25]:
        if time.time() > deadline:
            break
        cell = make_cross(w)
        gap = pwe_gap(cell, n_pw=6)
        if gap > best_gap:
            best_gap = gap
            best_cell = cell

    # Phase 2: Refine around best found
    if best_cell is not None and time.time() < deadline - 30:
        # Try variations around the best shape type
        # Determine what type of shape was best and refine
        # For now, do a local perturbation search
        for _ in range(5):
            if time.time() > deadline:
                break
            # Slightly perturb the cell
            cell = best_cell.copy()
            # Flip a few pixels randomly but maintain symmetry
            for _ in range(3):
                i = np.random.randint(0, size // 2)
                j = np.random.randint(0, size // 2)
                # Apply symmetrically
                cell[i, j] = 1 - cell[i, j]
                cell[size - 1 - i, j] = cell[i, j]
                cell[i, size - 1 - j] = cell[i, j]
                cell[size - 1 - i, size - 1 - j] = cell[i, j]
                # Also transpose symmetry
                cell[j, i] = cell[i, j]
                cell[j, size - 1 - i] = cell[i, j]
                cell[size - 1 - j, i] = cell[i, j]
                cell[size - 1 - j, size - 1 - i] = cell[i, j]
            gap = pwe_gap(cell, n_pw=7)
            if gap > best_gap:
                best_gap = gap
                best_cell = cell

    # Final evaluation with more accuracy if time permits
    if best_cell is not None and time.time() < deadline - 10:
        final_gap = pwe_gap(best_cell, n_pw=9)
        if final_gap > best_gap:
            best_gap = final_gap

    if best_cell is None:
        best_cell = make_disc(0.2)

    return best_cell.tolist()
