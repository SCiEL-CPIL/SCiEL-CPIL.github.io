# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.

    Uses separable 1D optimization: the union volume factorizes as
    integral over c of Lx(c) * Ly(c) dc, where Lx(c) is the 1D union length
    of x-intervals at height c. We optimize x-positions (one per i) and
    y-positions (one per j) independently using Nelder-Mead.
    """
    deadline = time.time() + 620  # 620s safety margin before 630s kill

    N = 8
    s = 1.0 / N  # square side length

    def union_length_1d(positions_1d, c_grid, indices, s_val):
        """
        Compute the 1D union length of intervals at heights c_grid.
        positions_1d: array of shape (N,), base positions for indices 0..N-1
        c_grid: array of heights
        indices: array of shape (N,), the grid index for each position
        Returns: array of union lengths at each c in c_grid
        """
        n_c = len(c_grid)
        # For each c, interval for index i is [pos[i] + c*indices[i]*s, pos[i] + c*indices[i]*s + s]
        # Shape: (N, n_c)
        starts = positions_1d[:, None] + c_grid[None, :] * (indices * s_val)[ :, None]
        ends = starts + s_val

        # Sort by start for each c
        order = np.argsort(starts, axis=0)
        s_sorted = np.take_along_axis(starts, order, axis=0)
        e_sorted = np.take_along_axis(ends, order, axis=0)

        # Compute union length via sweep
        # Union = sum of (min(end_i, max_end_so_far) - start_i) for each interval in sorted order
        max_end = s_sorted[0]
        for k in range(1, N):
            max_end = np.maximum(max_end, e_sorted[k])
        # Union length = max_end - min_start, but need to handle gaps
        # Better: union = sum over k of max(0, min(e_k, running_max) - s_k)
        # Actually: union length = sum_k max(0, min(e_k, max_{j<=k} e_j) - s_k)
        # Simpler: union = max_end - s_sorted[0] only if no gaps.
        # Correct sweep:
        running_max = s_sorted[0]
        union_len = np.zeros(n_c)
        for k in range(N):
            new_max = np.maximum(running_max, e_sorted[k])
            union_len += np.maximum(0.0, new_max - running_max)
            running_max = new_max
        return union_len

    def objective_x(positions_1d, c_grid, indices, s_val):
        """Objective: integral of Lx(c) dc (to be multiplied by Ly integral)."""
        Lx = union_length_1d(positions_1d, c_grid, indices, s_val)
        return np.trapz(Lx, c_grid)

    def objective_y(positions_1d, c_grid, indices, s_val):
        Ly = union_length_1d(positions_1d, c_grid, indices, s_val)
        return np.trapz(Ly, c_grid)

    # Use a fine grid for c
    c_grid = np.linspace(0.0, 1.0, 200)
    indices = np.arange(N, dtype=np.float64)

    # Initial guess: Keich construction
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(bin(idx)[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)

    # Optimize x-positions
    result_x = minimize(
        objective_x,
        keich.copy(),
        args=(c_grid, indices, s),
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    best_x = result_x.x
    best_x_val = result_x.fun

    if time.time() > deadline - 10:
        # Not enough time; use initial guess for y
        best_y = keich.copy()
    else:
        # Optimize y-positions
        result_y = minimize(
            objective_y,
            keich.copy(),
            args=(c_grid, indices, s),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        best_y = result_y.x

    # Polish with more iterations if time allows
    if time.time() < deadline - 20:
        result_x2 = minimize(
            objective_x,
            best_x,
            args=(c_grid, indices, s),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result_x2.fun < best_x_val:
            best_x = result_x2.x

        if time.time() < deadline - 10:
            result_y2 = minimize(
                objective_y,
                best_y,
                args=(c_grid, indices, s),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            best_y = result_y2.x

    # Build the 64x2 positions array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    return positions


# EVOLVE-BLOCK-END
