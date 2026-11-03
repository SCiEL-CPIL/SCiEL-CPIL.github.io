"""
Optimization of 100 equal spheres in a unit cube.
Combines iterative radius-increase energy minimization (vectorized) with
soft-min polishing and a final high-precision polish pass.
"""
N = 100
HALF = 0.5

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Wall constraints
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = HALF - max_abs
    min_wall_dist = np.min(wall_dists)

    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)

    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _energy_objective_and_grad(flat_pos, r_target):
    """
    Smooth energy objective: penalize pair overlaps and wall violations.
    E = 0.5 * sum_{i<j} max(0, 2*r_target - d_ij)^2 + 0.5 * sum_{i,k} max(0, r_target - (HALF - |x_ik|))^2
    This is C^1 smooth.
    """
    pos = flat_pos.reshape(N, 3)

    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)

    # Wall distances
    d_walls = HALF - np.abs(pos)

    # Compute violations
    pair_viol = np.maximum(0.0, 2.0 * r_target - dists)
    wall_viol = np.maximum(0.0, r_target - d_walls)

    energy = 0.5 * np.sum(pair_viol**2) + 0.5 * np.sum(wall_viol**2)

    # Gradient - vectorized
    grad_pos = np.zeros_like(pos)

    # Pair gradient: for each violating pair, push apart
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_viol_vals = pair_viol[iu]

    mask = pair_viol_vals > 0
    if np.any(mask):
        i_idx = iu[0][mask]
        j_idx = iu[1][mask]
        d_vals = pair_dists[mask]
        v_vals = pair_viol_vals[mask]

        # Avoid division by zero
        valid = d_vals > 1e-14
        if np.any(valid):
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            d_v = d_vals[valid]
            v_v = v_vals[valid]

            vecs = pos[i_v] - pos[j_v]  # shape (n_valid, 3)
            coeffs = v_v / d_v  # shape (n_valid,)

            # Accumulate gradients
            np.add.at(grad_pos, i_v, coeffs[:, np.newaxis] * vecs)
            np.add.at(grad_pos, j_v, -coeffs[:, np.newaxis] * vecs)

    # Wall gradient
    wall_viol_mask = wall_viol > 0
    if np.any(wall_viol_mask):
        i_idx, k_idx = np.where(wall_viol_mask)
        v_vals = wall_viol[wall_viol_mask]

        signs = np.sign(pos[i_idx, k_idx])
        signs[signs == 0] = 1.0

        grad_pos[i_idx, k_idx] += v_vals * signs

    return energy, -grad_pos.flatten()

def _energy_minimize(pos, r_target, max_iter=200, ftol=1e-12, gtol=1e-10):
    """Minimize energy for a fixed target radius."""
    bounds = [(-HALF, HALF)] * (N * 3)
    try:
        res = minimize(
            lambda x: _energy_objective_and_grad(x, r_target)[0],
            pos.flatten(),
            jac=lambda x: _energy_objective_and_grad(x, r_target)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if res.success or res.fun < 1e-8:
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def _soft_min_objective_and_grad(flat_pos, beta=200.0):
    """
    Soft-min objective: maximize the soft-minimum of all effective distances.
    soft_min = -logsumexp(-beta * all_eff_dists) / beta
    """
    pos = flat_pos.reshape(N, 3)

    # Pairwise distances (upper triangle only)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)

    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]

    # Wall distances
    d_walls = HALF - np.abs(pos)
    wall_dists = d_walls.flatten()

    # All effective distances
    all_eff_dists = np.concatenate([pair_dists / 2.0, wall_dists])

    # Soft-min
    scaled = -beta * all_eff_dists
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum_exp / beta

    # Gradient
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)

    grad_pos = np.zeros_like(pos)

    # Pair gradient
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        w = weights[idx]
        d = pair_dists[idx]
        if d > 1e-14:
            vec = pos[i] - pos[j]
            coeff = w / (2.0 * d)
            grad_pos[i] += coeff * vec
            grad_pos[j] -= coeff * vec

    # Wall gradient
    for idx in range(N * 3):
        i = idx // 3
        k = idx % 3
        w = weights[len(pair_dists) + idx]
        s = np.sign(pos[i, k])
        if s == 0:
            s = 1.0
        grad_pos[i, k] -= w * s

    return -soft_min_val, -grad_pos.flatten()

def _soft_min_optimize(pos, beta=200.0, max_iter=500, ftol=1e-12, gtol=1e-10):
    """Optimize using soft-min objective."""
    bounds = [(-HALF, HALF)] * (N * 3)
    try:
        res = minimize(
            lambda x: _soft_min_objective_and_grad(x, beta)[0],
            pos.flatten(),
            jac=lambda x: _soft_min_objective_and_grad(x, beta)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if res.success or np.isfinite(res.fun):
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def _polish(pos, max_iter=500, ftol=1e-12, gtol=1e-10):
    """High-precision polish pass."""
    # First try soft-min optimization
    polished = _soft_min_optimize(pos, beta=200.0, max_iter=max_iter, ftol=ftol, gtol=gtol)
    r1 = _compute_radius(polished)

    # Then try energy-based polish
    r_target = r1 * 1.001
    polished2 = _energy_minimize(polished, r_target, max_iter=max_iter, ftol=ftol, gtol=gtol)
    r2 = _compute_radius(polished2)

    if r2 > r1:
        return polished2
    return polished

def _radius_increase_search(pos, r_start, r_end, n_steps=50, deadline=None):
    """
    Iteratively increase target radius and minimize energy at each step.
    """
    r_values = np.linspace(r_start, r_end, n_steps)
    current_pos = pos.copy()

    for r in r_values:
        if deadline is not None and time.time() > deadline:
            break
        current_pos = _energy_minimize(current_pos, r, max_iter=100, ftol=1e-12, gtol=1e-10)

    return current_pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_radius = 0.0
    best_centers = None

    # Strategy 1: Grid Initialization
    def make_grid(nx, ny, nz):
        xs = np.linspace(-HALF + HALF/nx, HALF - HALF/nx, nx)
        ys = np.linspace(-HALF + HALF/ny, HALF - HALF/ny, ny)
        zs = np.linspace(-HALF + HALF/nz, HALF - HALF/nz, nz)
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        return np.array(grid)

    grid_centers = make_grid(5, 4, 5)

    # Strategy 2: Random Initializations
    rng = np.random.RandomState(42)
    random_centers = rng.uniform(-0.4, 0.4, size=(N, 3))

    inits = [grid_centers, random_centers]
    for i in range(5):
        rc = rng.uniform(-0.4, 0.4, size=(N, 3))
        inits.append(rc)

    for init in inits:
        if time.time() > deadline - 10.0:
            break

        # 1. Calculate initial radius
        r_est = _compute_radius(init)

        # 2. Iterative radius increase search
        r_start = r_est * 1.01
        if r_start < 0.05:
            r_start = 0.05
        r_end = 0.12

        current_pos = _radius_increase_search(init, r_start, r_end, n_steps=50, deadline=deadline)

        # 3. Polish
        polished = _polish(current_pos, max_iter=300)
        r_final = _compute_radius(polished)

        if r_final > best_radius:
            best_radius = r_final
            best_centers = polished.copy()

    # Final safety check
    if best_centers is None:
        best_centers = grid_centers
    else:
        best_centers = np.clip(best_centers, -0.4999, 0.4999)

    return best_centers
# EVOLVE-BLOCK-END
