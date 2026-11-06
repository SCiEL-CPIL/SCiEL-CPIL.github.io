"""
Pack 200 equal circles in the unit square [-1/2, 1/2]^2 to maximize common radius.
Combines hexagonal grid initialization, a vectorized repulsion-based local optimizer,
and L-BFGS-B with a soft-min surrogate for global refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 200
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distances: min_i (0.5 - |c_i|)
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(dx, dy)
    min_wall = np.min(wall_dists)
    if min_wall < 0:
        return 0.0
    if N < 2:
        return float(min_wall)
    d = pdist(centers)
    min_pair = np.min(d)
    pair_term = min_pair / 2.0
    return min(float(min_wall), float(pair_term))


def _pdist_index_to_pair(k, n):
    """Convert pdist condensed index to (i, j) with i < j."""
    start = 0
    for i in range(n - 1):
        block_size = n - i - 1
        if k < start + block_size:
            j = i + 1 + (k - start)
            return i, j
        start += block_size
    return 0, 1


def _hexagonal_grid(n_circles):
    """
    Create a hexagonal packing arrangement for n_circles circles.
    Returns an array of shape (n_circles, 2).
    """
    best_centers = None
    best_r = 0.0

    for m in range(10, 25):
        base_per_row = N // m
        extra = N % m
        row_counts = []
        for i in range(m):
            if i < extra:
                row_counts.append(base_per_row + 1)
            else:
                row_counts.append(base_per_row)
        max_row_count = max(row_counts)

        r_width = 1.0 / (2.0 * max_row_count)
        r_height = 1.0 / ((m - 1) * np.sqrt(3) + 2)
        r_est = min(r_width, r_height)

        if r_est <= best_r and best_centers is not None:
            continue

        dx = 2.0 * r_est
        dy = np.sqrt(3) * r_est

        centers = []
        y_offset = -0.5 + r_est

        for row_idx, row_count in enumerate(row_counts):
            x_offset = 0.0
            if row_idx % 2 == 1:
                x_offset = dx / 2.0

            if row_count > 0:
                row_width = (row_count - 1) * dx
                start_x = -row_width / 2.0 + x_offset
            else:
                start_x = 0.0

            for col_idx in range(row_count):
                x = start_x + col_idx * dx
                y = y_offset + row_idx * dy
                centers.append([x, y])

        centers = np.array(centers)
        centers[:, 0] -= np.mean(centers[:, 0])
        centers[:, 1] -= np.mean(centers[:, 1])
        centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)

        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()

    return best_centers if best_centers is not None else np.random.uniform(-0.5, 0.5, (N, 2))


def _refine_repulsion(centers, max_iters=500000):
    """
    Refine the packing using a repulsion-based optimizer that directly
    targets the bottleneck (min pair distance or min wall distance).
    """
    centers = centers.copy()
    r = _compute_radius(centers)
    best_r = r
    best_centers = centers.copy()

    step_size = 0.02
    no_improve = 0
    stagnation_threshold = 100

    for it in range(max_iters):
        if _remaining_time() < 5.0:
            break

        # Compute pairwise distances
        d = pdist(centers)
        min_pair_idx = np.argmin(d)
        min_pair_dist = d[min_pair_idx]
        pair_term = min_pair_dist / 2.0

        # Wall distances
        dw = np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
        min_wall_idx = np.argmin(dw)
        min_wall_dist = dw[min_wall_idx]
        wall_term = min_wall_dist

        current_r = min(pair_term, wall_term)

        if pair_term <= wall_term:
            # Pair is the bottleneck: push the closest pair apart
            i, j = _pdist_index_to_pair(min_pair_idx, N)
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                vec = np.random.randn(2)
                norm = np.linalg.norm(vec)
            direction = vec / norm
            move = step_size * direction
            centers[i] += move
            centers[j] -= move
        else:
            # Wall is the bottleneck: push the closest-to-wall circle inward
            idx = min_wall_idx
            x, y = centers[idx]
            dl = x + 0.5
            dr = 0.5 - x
            db = y + 0.5
            dt = 0.5 - y
            min_d = min(dl, dr, db, dt)
            if min_d == dl:
                move = np.array([step_size, 0.0])
            elif min_d == dr:
                move = np.array([-step_size, 0.0])
            elif min_d == db:
                move = np.array([0.0, step_size])
            else:
                move = np.array([0.0, -step_size])
            centers[idx] += move

        # Clip to square
        centers = np.clip(centers, -0.5, 0.5)

        new_r = _compute_radius(centers)
        if new_r > best_r + 1e-12:
            best_r = new_r
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > stagnation_threshold * 2:
                # Strong perturbation: move a random circle
                idx = np.random.randint(N)
                centers[idx] += np.random.randn(2) * 0.02
                centers[idx] = np.clip(centers[idx], -0.5, 0.5)
                no_improve = 0
                step_size = max(0.003, step_size * 0.9)
            elif no_improve > stagnation_threshold:
                step_size = max(0.003, step_size * 0.97)

    return best_centers, best_r


def _soft_min_objective_and_grad(x, beta=100.0):
    """
    Soft-min objective for L-BFGS-B.
    We want to maximize r = min(min_pair_dist/2, min_wall_dist).
    Equivalent to minimizing -r.
    Uses log-sum-exp trick: -min(d_i) ≈ (1/beta) * log(sum(exp(-beta * d_i)))
    """
    centers = x.reshape(N, 2)

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)

    # Wall distances
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(dx, dy)

    pair_half_dists = dists / 2.0
    iu = np.triu_indices(N, k=1)
    pair_vals = pair_half_dists[iu]
    all_vals = np.concatenate([pair_vals, wall_dists])

    min_val = np.min(all_vals)
    if min_val < 1e-10:
        return (1.0, np.zeros_like(x))

    exp_terms = np.exp(-beta * (all_vals - min_val))
    soft_min_val = min_val + np.log(np.sum(exp_terms)) / beta

    weights = exp_terms / np.sum(exp_terms)

    grad_centers = np.zeros((N, 2))

    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        w = weights[idx]
        d = dists[i, j]
        if d < 1e-10:
            continue
        diff_ij = centers[i] - centers[j]
        grad_ij = diff_ij / (2.0 * d)
        grad_centers[i] += w * grad_ij
        grad_centers[j] -= w * grad_ij

    for idx in range(N):
        w = weights[len(pair_vals) + idx]
        x_i = centers[idx, 0]
        y_i = centers[idx, 1]
        dx_i = 0.5 - abs(x_i)
        dy_i = 0.5 - abs(y_i)

        if dx_i <= dy_i:
            if x_i > 0:
                grad_centers[idx, 0] -= w
            else:
                grad_centers[idx, 0] += w
        else:
            if y_i > 0:
                grad_centers[idx, 1] -= w
            else:
                grad_centers[idx, 1] += w

    obj = -soft_min_val
    grad = -grad_centers.flatten()

    return (obj, grad)


def _refine_lbfgs(centers, beta=100.0, max_time=10.0):
    """
    Refine the packing using L-BFGS-B on a soft-min objective.
    """
    x0 = centers.flatten()

    def obj_and_grad(x):
        if _remaining_time() < 3.0:
            return (0.0, np.zeros_like(x))
        return _soft_min_objective_and_grad(x, beta=beta)

    bounds = [(-0.5, 0.5)] * (2 * N)

    best_r = _compute_radius(centers)
    best_x = x0.copy()

    try:
        result = minimize(
            lambda x: obj_and_grad(x)[0],
            x0,
            jac=lambda x: obj_and_grad(x)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 150,
                'maxfun': 1500,
                'ftol': 1e-8,
                'gtol': 1e-6,
                'maxls': 50,
            }
        )

        if result.x is not None and np.all(np.isfinite(result.x)):
            new_centers = result.x.reshape(N, 2)
            new_r = _compute_radius(new_centers)
            if new_r > best_r:
                best_r = new_r
                best_x = result.x.copy()
    except Exception:
        pass

    return best_x.reshape(N, 2), best_r


def construct():
    """Construct optimal circle packing."""
    # Start with hexagonal grid
    centers = _hexagonal_grid(N)
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()

    # Phase 1: Fast repulsion-based refinement (direct bottleneck targeting)
    if _remaining_time() > 10.0:
        try:
            refined_centers, refined_radius = _refine_repulsion(centers)
            if refined_radius > best_radius:
                best_radius = refined_radius
                best_centers = refined_centers.copy()
        except Exception:
            pass

    # Phase 2: L-BFGS-B with soft-min for global optimization
    if _remaining_time() > 15.0:
        try:
            refined_centers, refined_radius = _refine_lbfgs(best_centers, beta=100.0)
            if refined_radius > best_radius:
                best_radius = refined_radius
                best_centers = refined_centers.copy()
        except Exception:
            pass

    # Phase 3: Multi-scale L-BFGS-B with different beta values
    if _remaining_time() > 20.0:
        for beta_val in [50.0, 200.0, 500.0]:
            if _remaining_time() < 10.0:
                break
            try:
                refined_centers, refined_radius = _refine_lbfgs(best_centers, beta=beta_val)
                if refined_radius > best_radius:
                    best_radius = refined_radius
                    best_centers = refined_centers.copy()
            except Exception:
                pass

    # Phase 4: Perturbed restarts with repulsion + L-BFGS-B
    if _remaining_time() > 25.0:
        n_restarts = 4
        for restart in range(n_restarts):
            if _remaining_time() < 10.0:
                break
            perturbed = best_centers + np.random.randn(N, 2) * 0.015
            perturbed = np.clip(perturbed, -0.5, 0.5)
            try:
                # Quick repulsion pass
                ref1, r1 = _refine_repulsion(perturbed, max_iters=50000)
                if r1 > best_radius:
                    best_radius = r1
                    best_centers = ref1.copy()
                # L-BFGS-B polish
                if _remaining_time() > 10.0:
                    ref2, r2 = _refine_lbfgs(best_centers, beta=100.0)
                    if r2 > best_radius:
                        best_radius = r2
                        best_centers = ref2.copy()
            except Exception:
                pass

    # Final safety clip
    best_centers = np.clip(best_centers, -0.5 + 1e-12, 0.5 - 1e-12)

    return best_centers
