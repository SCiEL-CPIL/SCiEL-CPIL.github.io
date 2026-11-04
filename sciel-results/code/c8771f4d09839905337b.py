"""Pack 100 equal spheres in a unit cube [-0.5, 0.5]^3 to maximize minimum radius.
Combines violation-energy landscape with contact-graph refinement and high-precision polish.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    wall_dist = HALF - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    if r_wall < 0:
        return 0.0
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair_dist = np.min(dists[:, 1])
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _energy_and_grad(x, target_r):
    """
    Energy for a fixed target radius: sum of squared violations of
    pairwise separation (2*r) and wall distance (r).
    Returns (energy, gradient_flat).
    """
    c = x.reshape(N, 3)

    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)

    active = dists < 2.0 * target_r
    active[np.diag_indices_from(active)] = False

    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    energy_pair = 0.5 * np.sum(violations ** 2)

    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]

    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)

    wall_dists = HALF - np.abs(c)
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    energy_wall = np.sum(wall_violations ** 2)

    signs = np.sign(c)
    grad_wall = 2.0 * wall_violations * signs

    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()

    return total_energy, total_grad

def _local_optimize_violation(init_pts, target_r, deadline, max_iter=3000):
    """
    L-BFGS-B minimization of the violation energy for a fixed target radius.
    """
    def objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)

    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (N * 3)

    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts
    except TimeoutError:
        return init_pts
    except Exception:
        return init_pts

def _generate_grid():
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _generate_bcc():
    points = []
    for ix in range(5):
        for iy in range(5):
            for iz in range(4):
                x = (ix + 0.5) / 5.0 - 0.5
                y = (iy + 0.5) / 5.0 - 0.5
                z = (iz + 0.5) / 4.0 - 0.5
                points.append([x, y, z])
                xc = (ix + 1.0) / 5.0 - 0.5
                yc = (iy + 1.0) / 5.0 - 0.5
                zc = (iz + 1.0) / 4.0 - 0.5
                points.append([xc, yc, zc])
    return np.array(points[:N])

def _generate_fcc():
    """Generate FCC-like structure."""
    points = []
    s = 0.5 / 4.0
    for ix in range(5):
        for iy in range(5):
            for iz in range(4):
                x = (ix + 0.5) * s - 0.5
                y = (iy + 0.5) * s - 0.5
                z = (iz + 0.5) * s - 0.5
                points.append([x, y, z])
                # Face centers
                if ix < 4:
                    points.append([(ix + 1.0) * s - 0.5, y, z])
                if iy < 4:
                    points.append([x, (iy + 1.0) * s - 0.5, z])
                if iz < 3:
                    points.append([x, y, (iz + 1.0) * s - 0.5])
    return np.array(points[:N])

def _contact_graph_refine(centers, radius, deadline):
    """
    Identify near-contact pairs and refine using gradient-based approach.
    Uses all contact pairs, not just the minimum one.
    """
    if time.time() > deadline - 5.0:
        return centers, _compute_radius(centers)

    centers = centers.copy()
    tree = cKDTree(centers)
    pairs = list(tree.query_pairs(2.2 * radius))

    if len(pairs) == 0:
        return centers, _compute_radius(centers)

    involved = set()
    for i, j in pairs:
        involved.add(i)
        involved.add(j)
    involved = sorted(involved)

    if len(involved) < 2:
        return centers, _compute_radius(centers)

    idx_array = np.array(involved)
    n_involved = len(involved)

    def sub_objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        c = centers.copy()
        c[idx_array] = x.reshape(n_involved, 3)
        
        # Compute minimum over all contact pairs
        min_d = np.inf
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_d:
                min_d = d
        
        # Wall distance for involved spheres
        wall = HALF - np.max(np.abs(c[idx_array]), axis=1)
        min_wall = np.min(wall)
        
        # Objective: minimize the negative of the minimum radius
        return -min(min_d / 2.0, min_wall)

    def sub_grad(x):
        if time.time() > deadline:
            raise TimeoutError()
        c = centers.copy()
        c[idx_array] = x.reshape(n_involved, 3)

        # Find the binding constraint (minimum distance pair or wall)
        min_d = np.inf
        min_pair = None
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_d:
                min_d = d
                min_pair = (i, j)

        wall = HALF - np.max(np.abs(c[idx_array]), axis=1)
        min_wall_idx = np.argmin(wall)
        min_wall = wall[min_wall_idx]

        grad_full = np.zeros((N, 3))

        # If pair distance is the bottleneck
        if min_d / 2.0 <= min_wall:
            if min_pair is not None:
                i, j = min_pair
                d_vec = c[i] - c[j]
                dist = min_d
                if dist > 1e-12:
                    grad_full[i] += d_vec / (2.0 * dist)
                    grad_full[j] -= d_vec / (2.0 * dist)
        # If wall is the bottleneck
        else:
            i_wall = idx_array[min_wall_idx]
            abs_c = np.abs(c[i_wall])
            k_max = np.argmax(abs_c)
            grad_full[i_wall, k_max] += -np.sign(c[i_wall, k_max])

        return grad_full[idx_array].flatten()

    x0 = centers[idx_array].flatten()
    bounds_sub = [(-HALF + 1e-9, HALF - 1e-9)] * (n_involved * 3)

    try:
        result = minimize(
            sub_objective,
            x0,
            jac=sub_grad,
            method='L-BFGS-B',
            bounds=bounds_sub,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        new_sub = result.x.reshape(n_involved, 3)
        centers[idx_array] = new_sub
    except TimeoutError:
        pass
    except Exception:
        pass

    centers = np.clip(centers, -HALF, HALF)
    return centers, _compute_radius(centers)

def _high_precision_polish(centers, radius, deadline):
    """
    Final polish: use very tight tolerances and more iterations.
    Try to squeeze out the last bit of radius.
    """
    if time.time() > deadline - 3.0:
        return centers, _compute_radius(centers)

    # Try a series of very small radius increases with tight optimization
    best_c = centers.copy()
    best_r = radius

    for attempt in range(5):
        if time.time() > deadline - 2.0:
            break

        target_r = best_r * (1.0 + 0.0002 * (attempt + 1))
        
        # Small perturbation to help escape local minima
        rng_local = np.random.default_rng(1000 + attempt)
        noise = rng_local.normal(0, 0.0005, size=(N, 3))
        init_pts = np.clip(best_c + noise, -0.499, 0.499)

        opt_pts = _local_optimize_violation(init_pts, target_r, deadline, max_iter=5000)

        if np.all(np.isfinite(opt_pts)):
            opt_pts = np.clip(opt_pts, -HALF + 1e-9, HALF - 1e-9)
            r_opt = _compute_radius(opt_pts)

            if r_opt > best_r:
                best_r = r_opt
                best_c = opt_pts.copy()

    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_centers = None
    best_radius = 0.0

    rng = np.random.default_rng(42)

    # Phase 1: Multi-start with violation-energy optimization
    initializations = []

    # Grid
    grid_centers = _generate_grid()
    initializations.append(grid_centers)

    # BCC
    bcc_centers = _generate_bcc()
    initializations.append(bcc_centers)

    # FCC-like
    fcc_centers = _generate_fcc()
    initializations.append(fcc_centers)

    # Perturbed grids
    for i in range(6):
        if time.time() > deadline - 60.0:
            break
        base = _generate_grid()
        noise_scale = 0.025 + 0.01 * i
        perturbed = base + rng.normal(0, noise_scale, size=(N, 3))
        perturbed = np.clip(perturbed, -0.48, 0.48)
        initializations.append(perturbed)

    # Random
    for i in range(4):
        if time.time() > deadline - 50.0:
            break
        random_c = rng.uniform(-0.45, 0.45, size=(N, 3))
        initializations.append(random_c)

    # Target radii to try
    target_radii = [0.105, 0.110, 0.112, 0.114, 0.116, 0.118]

    for init in initializations:
        if time.time() > deadline - 35.0:
            break

        for target_r in target_radii:
            if time.time() > deadline - 30.0:
                break

            # Add small noise to break symmetry
            noise = rng.normal(0, 0.002, size=(N, 3))
            init_pts = np.clip(init + noise, -0.49, 0.49)

            opt_pts = _local_optimize_violation(init_pts, target_r, deadline)

            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -HALF + 1e-9, HALF - 1e-9)
                r_opt = _compute_radius(opt_pts)

                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = opt_pts.copy()

    # Phase 2: Iterative radius increase (squeeze)
    if best_centers is not None and time.time() < deadline - 25.0:
        num_squeezes = 20
        for i in range(num_squeezes):
            if time.time() > deadline - 15.0:
                break

            success_improved = False
            increments = [0.003, 0.002, 0.001, 0.0005, 0.0002]

            for inc in increments:
                if time.time() > deadline - 12.0:
                    break

                target_r = best_radius * (1.0 + inc)

                noise = rng.normal(0, 0.0015, size=(N, 3))
                init_pts = np.clip(best_centers + noise, -0.49, 0.49)

                opt_pts = _local_optimize_violation(init_pts, target_r, deadline)

                if np.all(np.isfinite(opt_pts)):
                    opt_pts = np.clip(opt_pts, -HALF + 1e-9, HALF - 1e-9)
                    r_opt = _compute_radius(opt_pts)

                    if r_opt > best_radius:
                        best_radius = r_opt
                        best_centers = opt_pts.copy()
                        success_improved = True
                        break

            if not success_improved:
                # Try with more noise
                noise = rng.normal(0, 0.004, size=(N, 3))
                init_pts = np.clip(best_centers + noise, -0.49, 0.49)
                target_r = best_radius * 1.004

                opt_pts = _local_optimize_violation(init_pts, target_r, deadline)

                if np.all(np.isfinite(opt_pts)):
                    opt_pts = np.clip(opt_pts, -HALF + 1e-9, HALF - 1e-9)
                    r_opt = _compute_radius(opt_pts)

                    if r_opt > best_radius:
                        best_radius = r_opt
                        best_centers = opt_pts.copy()

    # Phase 3: Contact-graph refinement
    if best_centers is not None and time.time() < deadline - 10.0:
        for refine_iter in range(5):
            if time.time() > deadline - 8.0:
                break
            best_centers, r_refined = _contact_graph_refine(
                best_centers, best_radius, deadline
            )
            if r_refined > best_radius:
                best_radius = r_refined
            else:
                break

    # Phase 4: High-precision polish
    if best_centers is not None and time.time() < deadline - 5.0:
        best_centers, best_radius = _high_precision_polish(
            best_centers, best_radius, deadline
        )

    # Ensure validity
    if best_centers is None:
        best_centers = _generate_grid()

    best_centers = np.clip(best_centers, -HALF, HALF)
    final_radius = _compute_radius(best_centers)
    if final_radius > 0:
        return best_centers

    return _generate_grid()
