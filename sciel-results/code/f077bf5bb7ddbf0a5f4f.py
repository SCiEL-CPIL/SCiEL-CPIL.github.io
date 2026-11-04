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
    if np.any(centers < -HALF) or np.any(centers > HALF):
        return 0.0
    
    # Wall distance
    wall_dist = HALF - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    if r_wall < 0:
        return 0.0
    
    # Pairwise distance
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

def _local_optimize_violation(init_pts, target_r, t_start, deadline, max_iter=2000):
    """
    L-BFGS-B minimization of the violation energy for a fixed target radius.
    Uses the global deadline to bound runtime.
    """
    def objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)[0]

    def gradient(x):
        if time.time() > deadline:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)[1]

    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (N * 3)

    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
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

def _contact_graph_refine(centers, radius, deadline):
    """
    Identify near-contact pairs and refine only the involved spheres
    to maximize the minimum distance among those pairs (and wall).
    """
    if time.time() > deadline - 3.0:
        return centers, _compute_radius(centers)

    centers = centers.copy()
    tree = cKDTree(centers)
    pairs = tree.query_pairs(2.1 * radius)

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

    def sub_objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        c = centers.copy()
        c[idx_array] = x.reshape(len(involved), 3)
        min_d = np.inf
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_d:
                min_d = d
        wall = HALF - np.max(np.abs(c[idx_array]), axis=1)
        # We want to maximize min(min_d/2, min_wall)
        # So minimize -min(min_d/2, min_wall)
        return -(min(min_d / 2.0, np.min(wall)))

    def sub_grad(x):
        if time.time() > deadline:
            raise TimeoutError()
        c = centers.copy()
        c[idx_array] = x.reshape(len(involved), 3)

        min_d = np.inf
        min_pair = None
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_d:
                min_d = d
                min_pair = (i, j)

        wall = HALF - np.max(np.abs(c[idx_array]), axis=1)
        min_wall_idx = np.argmin(wall)
        min_wall_val = wall[min_wall_idx]
        
        # Determine which constraint is active
        if min_d / 2.0 < min_wall_val:
            # Pair constraint is active
            if min_pair is None:
                return np.zeros_like(x)
            i, j = min_pair
            d = c[i] - c[j]
            dist = min_d
            if dist > 1e-12:
                grad_full = np.zeros((N, 3))
                grad_full[i] += d / (2.0 * dist)
                grad_full[j] -= d / (2.0 * dist)
                return -grad_full[idx_array].flatten()
            else:
                return np.zeros_like(x)
        else:
            # Wall constraint is active
            i_wall = idx_array[min_wall_idx]
            abs_c = np.abs(c[i_wall])
            k_max = np.argmax(abs_c)
            grad_full = np.zeros((N, 3))
            grad_full[i_wall, k_max] += -np.sign(c[i_wall, k_max])
            return -grad_full[idx_array].flatten()

    x0 = centers[idx_array].flatten()
    bounds_sub = [(-HALF + 1e-9, HALF - 1e-9)] * (len(involved) * 3)

    try:
        result = minimize(
            sub_objective,
            x0,
            jac=sub_grad,
            method='L-BFGS-B',
            bounds=bounds_sub,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        new_sub = result.x.reshape(len(involved), 3)
        centers[idx_array] = new_sub
    except TimeoutError:
        pass
    except Exception:
        pass

    centers = np.clip(centers, -HALF, HALF)
    return centers, _compute_radius(centers)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

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

    # Perturbed grids
    for i in range(5):
        if time.time() > deadline - 50.0:
            break
        base = _generate_grid()
        noise_scale = 0.03 + 0.015 * i
        perturbed = base + rng.normal(0, noise_scale, size=(N, 3))
        perturbed = np.clip(perturbed, -0.48, 0.48)
        initializations.append(perturbed)

    # Random
    for i in range(3):
        if time.time() > deadline - 40.0:
            break
        random_c = rng.uniform(-0.45, 0.45, size=(N, 3))
        initializations.append(random_c)

    # Target radii to try
    target_radii = [0.105, 0.110, 0.112, 0.114, 0.116]

    for init in initializations:
        if time.time() > deadline - 30.0:
            break

        # Try each target radius from this initialization
        for target_r in target_radii:
            if time.time() > deadline - 25.0:
                break

            # Add small noise to break symmetry
            noise = rng.normal(0, 0.003, size=(N, 3))
            init_pts = np.clip(init + noise, -0.49, 0.49)

            opt_pts = _local_optimize_violation(init_pts, target_r, start_time, deadline)

            if np.all(np.isfinite(opt_pts)):
                r_achieved = _compute_radius(opt_pts)
                if r_achieved > best_radius:
                    best_radius = r_achieved
                    best_centers = opt_pts.copy()

    # Phase 2: Refinement loop on the best solution found
    if best_centers is not None and time.time() < deadline - 10.0:
        current_centers = best_centers.copy()
        current_radius = best_radius
        
        # Iterative squeeze
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
                
            # Try to increase radius slightly
            new_target = current_radius + 0.001
            opt_pts = _local_optimize_violation(current_centers, new_target, start_time, deadline)
            new_r = _compute_radius(opt_pts)
            
            if new_r > current_radius:
                current_centers = opt_pts
                current_radius = new_r
                best_centers = current_centers.copy()
                best_radius = current_radius
            else:
                # Try contact graph refinement
                refined_centers, refined_r = _contact_graph_refine(current_centers, current_radius, deadline)
                if refined_r > current_radius:
                    current_centers = refined_centers
                    current_radius = refined_r
                    best_centers = current_centers.copy()
                    best_radius = current_radius
                else:
                    break

    # Final check
    if best_centers is None:
        return _generate_grid()
    
    return best_centers
