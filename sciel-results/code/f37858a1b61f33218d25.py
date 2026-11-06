"""
Direct 100-coordinate optimization via L-BFGS-B on a smooth soft-min objective.

Strategy:
1. Initialize with a structured hexagonal lattice pattern (8-7-7-7-7-7-7-7 or similar).
2. Optimize all 100 coordinates (50 centers × 2) using scipy L-BFGS-B.
   The objective is a smooth approximation of the negative minimum radius,
   using log-sum-exp to approximate the min of pairwise distances and
   boundary distances. This provides gradients that guide the optimizer.
3. Polish with a fast coordinate-wise local search to fine-tune.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE_FRACTION = 0.82
TOTAL_BUDGET = 150.0


def _compute_radius(centers):
    """Compute the largest common radius for given centers (exact)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0

    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)

    if min_boundary <= 0:
        return 0.0

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0

    return min(min_boundary, min_pair)


def _fast_radius(centers):
    """Fast radius computation without validity checks."""
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_boundary, min_pair)


def _hex_lattice_init(pattern=None):
    """
    Generate a hexagonal lattice initialization with a specific row pattern.
    Returns an (N, 2) array of centers.
    """
    if pattern is None:
        # 8-7-7-7-7-7-7-7 = 50
        pattern = [8, 7, 7, 7, 7, 7, 7, 7]

    total = sum(pattern)
    assert total == N, f"Pattern {pattern} sums to {total}, not {N}"

    centers = []
    n_rows = len(pattern)
    # Vertical spacing: distribute rows evenly in [-0.5, 0.5]
    # Leave margin for circles
    margin = 0.07  # approximate radius
    y_spacing = (1.0 - 2 * margin) / max(n_rows - 1, 1)
    y_start = -0.5 + margin + 0.5 * y_spacing

    for row_idx, count in enumerate(pattern):
        y = y_start + row_idx * y_spacing
        # Horizontal spacing
        # Hex offset: alternate rows shift by half spacing
        # First estimate dx from the widest row
        max_count = max(pattern)
        dx = (1.0 - 2 * margin) / max(max_count - 1, 1)

        # Hex offset for alternating rows
        offset = 0.0
        if row_idx % 2 == 1:
            offset = dx / 2.0

        # Center the row
        x_start = -0.5 + margin + dx / 2.0 - offset
        for col in range(count):
            x = x_start + col * dx
            centers.append([x, y])

    centers = np.array(centers, dtype=np.float64)

    # Clip to ensure inside
    np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9, out=centers)

    return centers


def _hex_lattice_init_2():
    """Alternative hex lattice: 7-6-7-6-7-6-7-6-7-8 = 50"""
    pattern = [7, 6, 7, 6, 7, 6, 7, 6, 7, 8]
    return _hex_lattice_init(pattern)


def _hex_lattice_init_3():
    """Another pattern: 8-7-8-7-8-7-8-7 = 50"""
    pattern = [8, 7, 8, 7, 8, 7, 8, 7]
    return _hex_lattice_init(pattern)


def _soft_min(values, temperature=0.005):
    """
    Smooth approximation of min(values) using log-sum-exp.
    soft_min(v) = -log(sum(exp(-v_i / T))) * T
    As T -> 0, this approaches min(v_i).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    # Shift for numerical stability
    v_shifted = values / temperature
    v_max = np.max(v_shifted)
    # soft_min = -T * log(sum(exp(-(v_i - v_min)/T)))
    # But we want to approximate min, so:
    # soft_min ≈ v_min - T * log(sum(exp(-(v_i - v_min)/T)))
    v_min = np.min(v_shifted)
    exp_terms = np.exp(-(v_shifted - v_min))
    log_sum = np.log(np.sum(exp_terms))
    return temperature * (v_min - log_sum)


def _negative_soft_radius_flat(flat, temperature=0.005):
    """
    Objective: negative of soft-approximated minimum radius.
    flat: shape (100,) -> reshape to (50, 2)
    """
    centers = flat.reshape(N, 2)

    # Boundary distances
    boundary_dists = 0.5 - np.abs(centers)
    # We want min of boundary_dists
    # Soft min of boundary_dists
    soft_boundary = _soft_min(boundary_dists, temperature)

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    # We need min of dists/2, so min of dists_sq / 4
    # Take sqrt first for correctness
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    # Get only upper triangle to avoid duplicates
    iu = np.triu_indices(N, k=1)
    pair_dists_upper = pair_dists[iu]
    soft_pair = _soft_min(pair_dists_upper, temperature)

    # The minimum radius is min(soft_boundary, soft_pair)
    # Use soft min of the two
    min_r = _soft_min(np.array([soft_boundary, soft_pair]), temperature)

    return -min_r


def _negative_soft_radius_flat_grad(flat, temperature=0.005):
    """
    Objective with numerical gradient (L-BFGS-B can do finite differences,
    but providing gradient is faster).
    Actually, let's use scipy's numerical gradient for simplicity,
    but we can provide an analytical gradient later if needed.
    For now, use the function only and let L-BFGS-B compute gradients.
    """
    return _negative_soft_radius_flat(flat, temperature)


def _local_search(centers, n_iterations=3000, deadline=None):
    """Fast coordinate-wise local search to polish."""
    best_centers = centers.copy()
    best_radius = _fast_radius(best_centers)
    rng = np.random.RandomState(123)

    current_step = 0.01
    no_improve_count = 0

    for i in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break

        idx = rng.randint(N)
        coord = rng.randint(2)
        orig_val = best_centers[idx, coord]

        improved = False
        for attempt in range(4):
            s = current_step * (0.5 ** attempt)
            delta = rng.normal(0, s)
            candidate_val = orig_val + delta

            # Clip to bounds
            candidate_val = np.clip(candidate_val, -0.5 + 1e-9, 0.5 - 1e-9)

            best_centers[idx, coord] = candidate_val
            new_radius = _fast_radius(best_centers)

            if new_radius > best_radius:
                best_radius = new_radius
                improved = True
                break
            else:
                best_centers[idx, coord] = orig_val

        if improved:
            no_improve_count = 0
            current_step = min(0.03, current_step * 1.05)
        else:
            no_improve_count += 1
            if no_improve_count > 30:
                current_step = max(0.0005, current_step * 0.7)
                no_improve_count = 0

    return best_centers, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_BUDGET * DEADLINE_FRACTION

    best_centers = None
    best_radius = 0.0

    # Generate multiple initial configurations
    initial_configs = [
        _hex_lattice_init(),       # 8-7-7-7-7-7-7-7
        _hex_lattice_init_2(),     # 7-6-7-6-7-6-7-6-7-8
        _hex_lattice_init_3(),     # 8-7-8-7-8-7-8-7
    ]

    # Also add a simple grid as fallback
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initial_configs.append(grid[:N].astype(np.float64))

    for init_idx, centers_init in enumerate(initial_configs):
        if time.time() > deadline:
            break

        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        # Evaluate initial radius
        r_init = _fast_radius(centers_init)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers_init.copy()

        # Run L-BFGS-B optimization
        flat_init = centers_init.flatten().astype(np.float64)

        # Set bounds
        bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (2 * N)

        # Temperature for soft min: start relatively high, can anneal
        # But for a single run, use a moderate value
        temperature = 0.003

        def objective(flat):
            if time.time() > deadline:
                return 1e10
            return _negative_soft_radius_flat(flat, temperature)

        try:
            res = minimize(
                objective,
                flat_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50,
                }
            )

            opt_centers = res.x.reshape(N, 2)
            # Ensure validity
            if np.all(np.isfinite(opt_centers)) and np.all(np.abs(opt_centers) <= 0.5 + 1e-9):
                r_opt = _fast_radius(opt_centers)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = opt_centers.copy()

        except Exception as e:
            pass  # Keep best so far

        # Polish with local search if we have time
        if best_centers is not None and time.time() < deadline - 8.0:
            remaining_ls = deadline - time.time() - 8.0
            # Scale iterations
            n_iters = max(500, int(5000 * remaining_ls / 60.0))
            polished, r_polished = _local_search(
                best_centers, n_iterations=n_iters, deadline=deadline
            )
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished

    # Final safety check
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        m = 8
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N].astype(np.float64)

    final_radius = _compute_radius(best_centers)
    if final_radius <= 0 or not np.all(np.isfinite(best_centers)):
        m = 8
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N].astype(np.float64)

    return best_centers


# EVOLVE-BLOCK-END
