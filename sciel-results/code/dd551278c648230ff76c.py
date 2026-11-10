# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_volume(positions, n_samples=500000):
    """Compute the Monte Carlo volume estimate as per the task specification."""
    N = 8
    n = len(positions)
    i_idx = np.arange(n) // N
    j_idx = np.arange(n) % N

    x0 = positions[:, 0]
    y0 = positions[:, 1]

    X0 = np.min(x0)
    X1 = np.max(x0 + (i_idx + 1) / N)
    Y0 = np.min(y0)
    Y1 = np.max(y0 + (j_idx + 1) / N)

    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return float('inf')

    rng = np.random.default_rng(12345)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)

    inside = np.zeros(n_samples, dtype=bool)
    for k in range(n):
        i = i_idx[k]
        j = j_idx[k]
        x_lo = x0[k] + c * i / N
        x_hi = x_lo + 1.0 / N
        y_lo = y0[k] + c * j / N
        y_hi = y_lo + 1.0 / N
        inside |= (a >= x_lo) & (a <= x_hi) & (b >= y_lo) & (b <= y_hi)

    frac = np.mean(inside)
    return frac * (X1 - X0) * (Y1 - Y0)


def _compute_volume_fast(positions, n_samples=30000):
    """Faster volume estimate for optimization (fewer samples, different seed)."""
    N = 8
    n = len(positions)
    i_idx = np.arange(n) // N
    j_idx = np.arange(n) % N

    x0 = positions[:, 0]
    y0 = positions[:, 1]

    X0 = np.min(x0)
    X1 = np.max(x0 + (i_idx + 1) / N)
    Y0 = np.min(y0)
    Y1 = np.max(y0 + (j_idx + 1) / N)

    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return float('inf')

    rng = np.random.default_rng(42)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)

    inside = np.zeros(n_samples, dtype=bool)
    for k in range(n):
        i = i_idx[k]
        j = j_idx[k]
        x_lo = x0[k] + c * i / N
        x_hi = x_lo + 1.0 / N
        y_lo = y0[k] + c * j / N
        y_hi = y_lo + 1.0 / N
        inside |= (a >= x_lo) & (a <= x_hi) & (b >= y_lo) & (b <= y_hi)

    frac = np.mean(inside)
    return frac * (X1 - X0) * (Y1 - Y0)


def _keich_positions_1d(N):
    """Generate Keich binary construction positions for 1D."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _initial_positions():
    """Generate initial positions using Keich construction."""
    N = 8
    keich_1d = _keich_positions_1d(N)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = keich_1d[i_flat]
    base_y = keich_1d[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


def _local_search(positions, max_time=550, initial_step=0.02):
    """Local search to minimize volume."""
    best_positions = positions.copy()
    best_vol = _compute_volume_fast(best_positions)
    step_size = initial_step
    start_time = time.time()
    deadline = start_time + max_time

    # Coordinate descent with adaptive step size
    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline:
        improved = False

        for k in range(len(positions)):
            if time.time() > deadline:
                break

            for dim in [0, 1]:
                # Try multiple step sizes for this coordinate
                for delta in [step_size, -step_size]:
                    trial = best_positions.copy()
                    trial[k, dim] += delta

                    vol = _compute_volume_fast(trial)
                    if vol < best_vol:
                        best_positions = trial
                        best_vol = vol
                        improved = True
                        no_improve_count = 0

        if not improved:
            no_improve_count += 1
            step_size *= 0.7
            if step_size < 0.0005 or no_improve_count > max_no_improve:
                break
        else:
            no_improve_count = 0

    return best_positions, best_vol


def _random_restarts(positions, n_restarts=5, max_time=550):
    """Try multiple random restarts and keep the best."""
    best_positions = positions.copy()
    best_vol = _compute_volume_fast(best_positions)
    start_time = time.time()
    deadline = start_time + max_time

    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        # Create perturbed starting point
        perturbation_scale = 0.1 * (restart + 1)
        rng = np.random.default_rng(100 + restart)
        perturbed = positions + rng.normal(0, perturbation_scale, positions.shape)

        # Run local search from this point
        improved_positions, improved_vol = _local_search(
            perturbed,
            max_time=max(10, deadline - time.time() - 5)
        )

        if improved_vol < best_vol:
            best_positions = improved_positions
            best_vol = improved_vol

    return best_positions, best_vol


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses Keich binary construction as initialization, then local search with
    random restarts to minimize the 2D union volume.
    
    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit

    # Initial Keich construction
    positions = _initial_positions()
    init_vol = _compute_volume_fast(positions)

    elapsed = time.time() - start_time
    if elapsed > 580:
        return positions

    # Try random restarts with local search
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        improved_positions, improved_vol = _random_restarts(
            positions,
            n_restarts=8,
            max_time=remaining_time - 10
        )

        if np.all(np.isfinite(improved_positions)) and improved_vol > 0:
            positions = improved_positions

    # Final check
    if not np.all(np.isfinite(positions)):
        return _initial_positions()

    return positions


# EVOLVE-BLOCK-END
