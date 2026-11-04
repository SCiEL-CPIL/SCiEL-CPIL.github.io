"""
Pack 100 equal spheres in a unit cube [-0.5, 0.5]^3 to maximize minimum radius.
Combines a violation-energy landscape with iterative radius-increase, multi-start,
FCC lattice initialization, and contact-graph refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

N = 100
HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distance
    wall_dist = HALF - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    if r_wall < 0:
        return 0.0
    # Pairwise distance via cKDTree
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
        return _energy_and_grad(x, target_r)

    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (N * 3)

    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts
    except TimeoutError:
        return init_pts
    except Exception:
        return init_pts

def _high_precision_polish(centers, deadline):
    """
    Final high-precision polish: use a smooth log-sum-exp based energy to
    maximize the minimum distance more precisely.
    """
    if time.time() > deadline - 5.0:
        return centers, _compute_radius(centers)

    c = centers.copy()
    current_r = _compute_radius(c)
    if current_r <= 0:
        return centers, current_r

    # Use a smooth surrogate: minimize sum of exp(-(d - 2r)/beta) for pairs
    # plus wall violations
    beta = 0.005

    def smooth_objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        cc = x.reshape(N, 3)
        diff = cc[:, np.newaxis, :] - cc[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1e10)

        # Pair energy: sum of soft violations
        pair_viol = 2.0 * current_r - dists
        mask = ~np.eye(N, dtype=bool)
        pair_viol = np.where(mask, pair_viol, 0.0)
        # Soft max via log-sum-exp
        max_v = np.max(pair_viol[mask])
        if max_v < -50.0:
            pair_energy = 0.0
        else:
            shifted = pair_viol - max_v
            pair_energy = beta * (max_v + np.log(np.sum(np.exp(shifted[mask]))))

        # Wall energy
        wall_dists = HALF - np.max(np.abs(cc), axis=1)
        wall_viol = np.maximum(0.0, current_r - wall_dists)
        wall_energy = np.sum(wall_viol ** 2) * 10.0

        return pair_energy + wall_energy

    def smooth_gradient(x):
        if time.time() > deadline:
            raise TimeoutError()
        cc = x.reshape(N, 3)
        diff = cc[:, np.newaxis, :] - cc[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1e10)

        pair_viol = 2.0 * current_r - dists
        mask = ~np.eye(N, dtype=bool)
        pair_viol = np.where(mask, pair_viol, 0.0)

        max_v = np.max(pair_viol[mask])
        if max_v < -50.0:
            grad = np.zeros((N, 3))
        else:
            shifted = pair_viol - max_v
            weights = np.exp(shifted)
            weights[~mask] = 0.0
            weights /= np.sum(weights)

            safe_dists = np.where(dists > 1e-12, dists, 1.0)
            unit_vecs = diff / safe_dists[:, :, np.newaxis]
            # d/dx_i sum_j w_ij * (2r - d_ij) where w_ij depends on all
            # Simplified: gradient of soft-max energy
            grad_pair = np.zeros((N, 3))
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    w = weights[i, j]
                    if w < 1e-15:
                        continue
                    d = diff[i, j]
                    dist = dists[i, j]
                    if dist > 1e-12:
                        grad_pair[i] += w * (-d / dist)

            grad = grad_pair

        # Wall gradient
        wall_dists = HALF - np.max(np.abs(cc), axis=1)
        wall_viol = np.maximum(0.0, current_r - wall_dists)
        signs = np.sign(cc)
        grad_wall = 20.0 * wall_viol[:, np.newaxis] * signs
        grad += grad_wall

        return grad.flatten()

    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (N * 3)
    try:
        result = minimize(
            smooth_objective,
            c.flatten(),
            jac=smooth_gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
        )
        new_c = result.x.reshape(N, 3)
        new_c = np.clip(new_c, -HALF + 1e-9, HALF - 1e-9)
        new_r = _compute_radius(new_c)
        if new_r > current_r:
            return new_c, new_r
        else:
            return c, current_r
    except TimeoutError:
        return c, current_r
    except Exception:
        return c, current_r

def _generate_fcc():
    """
    Generate FCC lattice points in the cube.
    FCC has points at corners and face centers of each cubic cell.
    We use a 5x5x5 grid of cubic cells, but only keep points inside [-0.5, 0.5].
    Actually, let's use a simpler approach: generate FCC points in a larger region
    and select the first N that fit inside the cube.
    """
    points = []
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    # in units of the cubic lattice constant a
    # We want the total span to be about 1.0, so a ~ 1/5 = 0.2 for 5 cells
    a = 0.2
    basis = np.array([[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]])
    
    # Generate points in a 6x6x6 grid to ensure we have enough
    for ix in range(6):
        for iy in range(6):
            for iz in range(6):
                for b in basis:
                    pt = a * np.array([ix, iy, iz]) + a * b
                    # Center the whole thing so it's roughly in [-0.5, 0.5]
                    # The span is 6*a = 1.2, so center at 0.6*a = 0.12
                    # Actually, let's just generate and filter
                    points.append(pt)
    
    points = np.array(points)
    # Center the cloud
    points -= np.mean(points, axis=0)
    # Scale to fit in [-0.5, 0.5] roughly
    # The max extent is now about 0.6
    points *= 0.5 / 0.6
    
    # Filter points inside [-0.5, 0.5]
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    # If we have more than N, randomly select N
    if len(points) > N:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), N, replace=False)
        points = points[idx]
    
    # If we have fewer, pad with random points
    if len(points) < N:
        rng = np.random.default_rng(43)
        extra = rng.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
        
    return points

def _generate_grid():
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _contact_graph_refine(centers, radius, deadline):
    """
    Identify near-contact pairs and refine only the involved spheres
    to maximize the minimum distance among those pairs (and wall).
    Uses a smooth log-sum-exp surrogate for better convergence.
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
    n_inv = len(involved)
    beta = 0.01

    def sub_objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        c = centers.copy()
        c[idx_array] = x.reshape(n_inv, 3)
        # Compute distances for contact pairs
        dists = []
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            dists.append(d)
        dists = np.array(dists)
        # Soft min via log-sum-exp (negative)
        min_d = np.min(dists)
        if min_d < 2 * radius - 0.1:
            # Smooth approximation of min
            shifted = dists - min_d
            soft_min = min_d - beta * np.log(np.sum(np.exp(shifted / beta)))
        else:
            soft_min = min_d
        wall = HALF - np.max(np.abs(c[idx_array]), axis=1)
        min_wall = np.min(wall)
        # Combine: maximize min(soft_min/2, min_wall)
        return -(min(soft_min / 2.0, min_wall))

    def sub_grad(x):
        if time.time() > deadline:
            raise TimeoutError()
        c = centers.copy()
        c[idx_array] = x.reshape(n_inv, 3)

        dists = []
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            dists.append(d)
        dists = np.array(dists)
        min_d = np.min(dists)

        wall = HALF - np.max(np.abs(c[idx_array]), axis=1)
        min_wall = np.min(wall)

        # Determine which constraint is active
        if min_d / 2.0 < min_wall:
            # Pair constraint active
            # Use soft-min gradient
            shifted = dists - min_d
            weights = np.exp(shifted / beta)
            weights /= np.sum(weights)

            grad_full = np.zeros((N, 3))
            for k, (i, j) in enumerate(pairs):
                w = weights[k]
                if w < 1e-15:
                    continue
                d = c[i] - c[j]
                dist = dists[k]
                if dist > 1e-12:
                    grad_full[i] += w * (-d / dist)
                    grad_full[j] += w * (d / dist)
            # Scale by -0.5 (since we're minimizing -min_d/2)
            grad_full *= -0.5
        else:
            # Wall constraint active
            min_wall_idx = np.argmin(wall)
            i_wall = idx_array[min_wall_idx]
            abs_c = np.abs(c[i_wall])
            k_max = np.argmax(abs_c)
            grad_full = np.zeros((N, 3))
            grad_full[i_wall, k_max] = np.sign(c[i_wall, k_max])

        return grad_full[idx_array].flatten()

    x0 = centers[idx_array].flatten()
    bounds_sub = [(-HALF + 1e-9, HALF - 1e-9)] * (n_inv * 3)

    try:
        result = minimize(
            sub_objective,
            x0,
            jac=sub_grad,
            method='L-BFGS-B',
            bounds=bounds_sub,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        new_sub = result.x.reshape(n_inv, 3)
        centers[idx_array] = new_sub
    except TimeoutError:
        pass
    except Exception:
        pass

    centers = np.clip(centers, -HALF, HALF)
    return centers, _compute_radius(centers)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    best_centers = None
    best_radius = 0.0

    rng = np.random.default_rng(42)

    # Phase 1: Multi-start with violation-energy optimization
    initializations = []

    # FCC lattice
    fcc_centers = _generate_fcc()
    initializations.append(fcc_centers)

    # Grid
    grid_centers = _generate_grid()
    initializations.append(grid_centers)

    # Perturbed FCC
    for i in range(5):
        if time.time() > deadline - 50.0:
            break
        base = _generate_fcc()
        noise_scale = 0.03 + 0.015 * i
        perturbed = base + rng.normal(0, noise_scale, size=(N, 3))
        perturbed = np.clip(perturbed, -0.48, 0.48)
        initializations.append(perturbed)

    # Perturbed Grid
    for i in range(3):
        if time.time() > deadline - 40.0:
            break
        base = _generate_grid()
        noise_scale = 0.03 + 0.015 * i
        perturbed = base + rng.normal(0, noise_scale, size=(N, 3))
        perturbed = np.clip(perturbed, -0.48, 0.48)
        initializations.append(perturbed)

    # Random
    for i in range(3):
        if time.time() > deadline - 30.0:
            break
        random_c = rng.uniform(-0.45, 0.45, size=(N, 3))
        initializations.append(random_c)

    # Target radii to try - start lower, increase progressively
    # Known best is ~0.11138
    target_radii = [0.100, 0.105, 0.108, 0.110, 0.111, 0.112, 0.113, 0.114]

    for init in initializations:
        if time.time() > deadline - 25.0:
            break

        # Try each target radius from this initialization
        for target_r in target_radii:
            if time.time() > deadline - 20.0:
                break

            # Add small noise to break symmetry
            noise = rng.normal(0, 0.003, size=(N, 3))
            init_pts = np.clip(init + noise, -0.49, 0.49)

            opt_pts = _local_optimize_violation(init_pts, target_r, start_time, deadline)

            if np.all(np.isfinite(opt_pts)):
                r_opt = _compute_radius(opt_pts)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = opt_pts.copy()

    # Phase 2: Refinement using contact graph
    if best_centers is not None and time.time() < deadline - 10.0:
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            best_centers, best_radius = _contact_graph_refine(best_centers, best_radius, deadline)

    # Phase 3: High-precision polish
    if best_centers is not None and time.time() < deadline - 3.0:
        best_centers, best_radius = _high_precision_polish(best_centers, deadline)

    # Final validation
    if best_centers is None:
        best_centers = _generate_grid()
    
    best_centers = np.clip(best_centers, -HALF + 1e-9, HALF - 1e-9)
    
    return best_centers
