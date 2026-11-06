"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 115.0  # safety margin before 120s kill


def _compute_radius(c):
    """Compute the achievable common radius for centres c (N,3)."""
    n = c.shape[0]
    # Wall distances
    wall = 0.5 - np.max(np.abs(c), axis=1)
    min_wall = np.min(wall)
    # Pairwise distances - vectorized
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(min_dist / 2.0, min_wall)


def _relax_spheres(c, r, max_iter=3000, tol=1e-9, deadline=None):
    """
    Relax spheres to satisfy radius r using vectorized repulsion.
    Returns (c_new, converged)
    """
    c = c.copy()
    n = c.shape[0]
    limit = 0.5 - r
    # Adaptive learning rate
    lr = 0.4

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Pairwise differences and distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))

        # Avoid division by zero
        dist_safe = np.where(dist > 1e-8, dist, 1.0)

        # Overlap: positive if dist < 2r
        overlap = np.where(dist < 2 * r, 2 * r - dist, 0.0)

        # Force on each sphere from all others
        force = np.sum(diff * (overlap / dist_safe)[:, :, np.newaxis], axis=1)

        # Update positions
        c += lr * force

        # Project to walls
        c = np.clip(c, -limit, limit)

        # Check convergence
        max_force = np.max(np.abs(force))
        if max_force < tol:
            return c, True

        # Decay learning rate slightly
        lr *= 0.9999

    return c, False


def _make_grid(nx, ny, nz, jitter=0.0):
    """Create a grid of points in the unit cube centered at origin."""
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    if jitter > 0:
        grid = grid + np.random.uniform(-jitter, jitter, grid.shape)
    return grid


def _local_search(c, r, n_iterations=5000, deadline=None):
    """
    Local search: randomly perturb a sphere and accept if radius can be maintained.
    """
    c = c.copy()
    n = c.shape[0]
    limit = 0.5 - r
    best_c = c.copy()
    best_r = _compute_radius(c)

    for it in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break

        # Pick a random sphere to move
        idx = np.random.randint(n)
        old_pos = c[idx].copy()

        # Random perturbation - scale with current radius
        step = 0.003 * r * (1.0 - it / n_iterations) + 0.001 * r
        c[idx] += np.random.randn(3) * step
        c[idx] = np.clip(c[idx], -limit, limit)

        new_r = _compute_radius(c)
        if new_r > best_r:
            best_r = new_r
            best_c = c.copy()
        else:
            c[idx] = old_pos

    return best_c, best_r


def _soft_relax(c, r, max_iter=2000, deadline=None):
    """
    Smooth surrogate relaxation using soft-min formulation.
    Minimize sum of soft violations for radius r.
    """
    c = c.copy()
    n = c.shape[0]
    limit = 0.5 - r
    lr = 0.01

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Wall violations
        abs_c = np.abs(c)
        wall_viol = np.maximum(abs_c - limit, 0.0)
        wall_grad = np.sign(c) * wall_viol

        # Pairwise violations (soft)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)

        pair_viol = np.maximum(2 * r - dist, 0.0)
        dist_safe = np.where(dist > 1e-8, dist, 1.0)

        # Gradient for pairwise: for each i, sum over j of viol_ij * (c_i - c_j) / dist_ij
        pair_grad = np.sum(
            diff * (pair_viol / dist_safe)[:, :, np.newaxis], axis=1
        )

        total_grad = wall_grad + pair_grad

        c -= lr * total_grad
        c = np.clip(c, -limit, limit)

        # Decay learning rate
        lr *= 0.9995

    return c


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE

    best_c = None
    best_r = 0.0

    # Try multiple grid configurations
    grid_configs = [
        (5, 5, 4),
        (5, 4, 5),
        (4, 5, 5),
        (5, 5, 4),  # repeat with different jitter
    ]

    for gi, (nx, ny, nz) in enumerate(grid_configs):
        if time.time() > deadline:
            break

        jitter = 0.01 if gi > 0 else 0.0
        c0 = _make_grid(nx, ny, nz, jitter=jitter)
        r0 = _compute_radius(c0)

        # Phase 1: Relax to current radius
        c_relaxed, _ = _relax_spheres(c0, r0, max_iter=2000, deadline=deadline)
        r_relaxed = _compute_radius(c_relaxed)

        if r_relaxed > best_r:
            best_r = r_relaxed
            best_c = c_relaxed.copy()

        # Phase 2: Incrementally increase radius
        r_target = r_relaxed * 1.001
        max_r_target = 0.115  # known best is ~0.1114
        no_improve = 0

        while time.time() < deadline and r_target < max_r_target:
            c_test, converged = _relax_spheres(best_c, r_target, max_iter=1500, deadline=deadline)
            r_actual = _compute_radius(c_test)

            if r_actual >= r_target - 1e-9 and r_actual > best_r:
                best_r = r_actual
                best_c = c_test.copy()
                no_improve = 0
                r_target *= 1.002
            else:
                no_improve += 1
                r_target *= 1.0005
                if no_improve > 30:
                    break

        # Phase 3: Soft relaxation for smoothing
        if time.time() < deadline - 10:
            c_soft = _soft_relax(best_c, best_r, max_iter=1000, deadline=deadline)
            r_soft = _compute_radius(c_soft)
            if r_soft > best_r:
                best_r = r_soft
                best_c = c_soft.copy()

        # Phase 4: Local search
        if time.time() < deadline - 5:
            c_ls, r_ls = _local_search(best_c, best_r, n_iterations=3000, deadline=deadline)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()

    # Final polish: one more relaxation pass with the best radius
    if best_c is not None and time.time() < deadline - 2:
        c_final, _ = _relax_spheres(best_c, best_r, max_iter=2000, tol=1e-10, deadline=deadline)
        r_final = _compute_radius(c_final)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()

    # Ensure validity
    best_c = np.clip(best_c, -0.49999, 0.49999)

    return best_c
# EVOLVE-BLOCK-END
