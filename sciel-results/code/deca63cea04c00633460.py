"""
Combined packer: soft-min surrogate optimization with L-BFGS-B + active-set polish.
Structural idea from inspiration 1 (soft-min + L-BFGS-B with temperature annealing).
Refinement machinery from current program (multi-restart, polish, deadline guards).
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100


def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Wall clearance
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall = np.min(wall_dists)
    if min_wall < -1e-9:
        return 0.0

    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_pair = np.sqrt(dists_sq[iu]).min()

    return min(min_pair * 0.5, min_wall)


def _soft_min_obj_grad(x, temperature):
    """
    Smooth surrogate: minimize -soft_min(all distances).
    Distances include pairwise/2 and wall clearances.
    Uses log-sum-exp for numerical stability.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)

    # Pairwise distances / 2
    diff = centers[:, None, :] - centers[None, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    pair_dists = np.sqrt(dists_sq[iu] + 1e-15) * 0.5
    i_idx, j_idx = iu

    # Wall distances
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs

    # Combine all distances
    all_dists = np.concatenate([pair_dists, wall_dists])

    # Soft-min via log-sum-exp
    scaled = -all_dists / temperature
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp

    obj = temperature * (np.log(sum_exp) + max_scaled)

    # Gradient
    grad = np.zeros_like(centers)

    # Pairwise gradient
    for k in range(len(i_idx)):
        w = weights[k]
        if w < 1e-10:
            continue
        i = i_idx[k]
        j = j_idx[k]
        d = pair_dists[k] * 2.0
        if d < 1e-10:
            continue
        vec = centers[i] - centers[j]
        factor = -w / (2.0 * d)
        grad[i] += factor * vec
        grad[j] -= factor * vec

    # Wall gradient
    for i in range(n):
        w = weights[len(i_idx) + i]
        if w < 1e-10:
            continue
        max_val = max_abs[i]
        for k in range(3):
            if np.isclose(abs(centers[i, k]), max_val, atol=1e-10):
                grad[i, k] += w * np.sign(centers[i, k])
                break

    return obj, grad.flatten()


def _active_set_polish(centers, t0, deadline, max_iter=2000):
    """
    High-precision polish: minimize the minimum over active constraints.
    Active constraints are those within a small tolerance of the current minimum.
    Uses a simple projected gradient / coordinate descent approach.
    """
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    step = 0.002
    no_improve = 0

    for it in range(max_iter):
        if time.time() - t0 > deadline:
            break
        if no_improve > 500:
            break

        # Find active constraints
        diff = best_centers[:, None, :] - best_centers[None, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_dists = np.sqrt(dists_sq[iu] + 1e-15) * 0.5

        max_abs = np.max(np.abs(best_centers), axis=1)
        wall_dists = 0.5 - max_abs

        all_dists = np.concatenate([pair_dists, wall_dists])
        min_d = np.min(all_dists)
        tol = 1e-4

        # Build gradient from active constraints only
        grad = np.zeros_like(best_centers)

        # Active pairs
        for k in range(len(iu[0])):
            if pair_dists[k] < min_d + tol:
                i = iu[0][k]
                j = iu[1][k]
                d = pair_dists[k] * 2.0
                if d < 1e-10:
                    continue
                vec = best_centers[i] - best_centers[j]
                factor = -1.0 / (2.0 * d)
                grad[i] += factor * vec
                grad[j] -= factor * vec

        # Active walls
        for i in range(N):
            if wall_dists[i] < min_d + tol:
                max_val = max_abs[i]
                for k in range(3):
                    if np.isclose(abs(best_centers[i, k]), max_val, atol=1e-10):
                        grad[i, k] += np.sign(best_centers[i, k])
                        break

        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break

        g = grad / gnorm
        new_centers = best_centers + step * g
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)

        r = _compute_radius(new_centers)
        if r > best_r:
            best_r = r
            best_centers = new_centers.copy()
            no_improve = 0
            step = min(step * 1.05, 0.01)
        else:
            no_improve += 1
            step = max(step * 0.9, 1e-7)

    return best_centers, best_r


def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0  # 10s safety margin before 150s kill

    best_radius = 0.0
    best_centers = None

    # Grid initialization: 5x4x5 = 100
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5 / nx, 0.5 - 0.5 / nx, nx)
    ys = np.linspace(-0.5 + 0.5 / ny, 0.5 - 0.5 / ny, ny)
    zs = np.linspace(-0.5 + 0.5 / nz, 0.5 - 0.5 / nz, nz)
    grid = np.array([[x, y, z] for z in zs for y in ys for x in xs])

    rng = np.random.default_rng(123)

    # Build diverse initializations
    inits = [grid.copy()]
    for _ in range(4):
        pert = grid.copy() + rng.normal(0, 0.03, size=(N, 3))
        inits.append(np.clip(pert, -0.45, 0.45))
    for _ in range(3):
        inits.append(rng.uniform(-0.45, 0.45, size=(N, 3)))

    bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3)

    # Temperature schedule for soft-min annealing
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]

    for init in inits:
        if time.time() > DEADLINE - 25.0:
            break

        current = init.copy()

        # Phase 1: L-BFGS-B with decreasing temperature
        for temp in temperatures:
            if time.time() > DEADLINE - 20.0:
                break

            def obj_grad(x, t=temp):
                return _soft_min_obj_grad(x, t)

            try:
                res = minimize(
                    obj_grad,
                    current.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'disp': False, 'ftol': 1e-12,
                             'gtol': 1e-10, 'maxls': 40}
                )
                current = res.x.reshape(-1, 3)
            except Exception:
                pass

            current = np.clip(current, -0.5 + 1e-9, 0.5 - 1e-9)
            r = _compute_radius(current)
            if r > best_radius:
                best_radius = r
                best_centers = current.copy()

    # Phase 2: Polish the best result
    if best_centers is not None and time.time() < DEADLINE - 5.0:
        best_centers, best_radius = _active_set_polish(
            best_centers.copy(), t0, DEADLINE - 2.0
        )

    # Final safety
    if best_centers is None:
        best_centers = grid.copy()
    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    if not np.all(np.isfinite(best_centers)):
        best_centers = grid.copy()

    return best_centers
