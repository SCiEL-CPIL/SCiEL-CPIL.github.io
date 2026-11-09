"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _score(points):
    """Compute the packing score: min of pairwise distances/2 and (1 - |c_i|)."""
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    return min(min_pair, min_boundary)


def _compute_score(points):
    """Alias for _score."""
    return _score(points)


def _hex_lattice_init(n=50, spacing_factor=1.0, rotation=0.0, jitter=0.0):
    """Generate a hexagonal lattice initial configuration."""
    area_per_circle = np.pi / n
    hex_area_factor = np.sqrt(3) / 2
    density = np.pi / np.sqrt(12)
    s = np.sqrt(area_per_circle / (density * hex_area_factor))
    s *= spacing_factor

    k = int(2.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2.0
            points.append([x, y])
    points = np.array(points)

    if rotation != 0:
        theta = np.deg2rad(rotation)
        c, s_ = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, -s_], [s_, c]])
        points = points @ rot_mat.T

    dists = np.linalg.norm(points, axis=1)
    valid = dists < 1.0
    points = points[valid]
    dists = dists[valid]

    if len(points) >= n:
        indices = np.argsort(dists)[:n]
        points = points[indices]

    if jitter > 0:
        points = points + np.random.normal(0, jitter, points.shape)

    return points


def _violation(pos, r):
    """Calculate the sum of squared violations for a given radius r."""
    dists_center = np.linalg.norm(pos, axis=1)
    boundary_viol = np.maximum(0, dists_center - (1.0 - r))
    boundary_viol = np.sum(boundary_viol ** 2)

    dists = cdist(pos, pos)
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu]
    pair_viol = np.maximum(0, 2.0 * r - pair_dists)
    pair_viol = np.sum(pair_viol ** 2)

    return boundary_viol + pair_viol


def _optimize_for_radius(pos_init, r_target, maxiter=200):
    """Minimize the violation for a fixed radius r_target."""
    def objective(x):
        pos = x.reshape(N, 2)
        return _violation(pos, r_target)

    x0 = pos_init.flatten()
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-12})
    pos_opt = res.x.reshape(N, 2)
    viol = res.fun
    return pos_opt, viol


def _find_max_radius(pos_init, r_low=0.05, r_high=0.15, tol=1e-7, max_iters=40, deadline=None):
    """Binary search for the maximum feasible radius."""
    pos_current = pos_init.copy()

    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        if r_high - r_low < tol:
            break

        r_mid = (r_low + r_high) / 2.0
        pos_opt, viol = _optimize_for_radius(pos_current, r_mid)

        if viol < 1e-8:
            r_low = r_mid
            pos_current = pos_opt
        else:
            r_high = r_mid

    pos_final, _ = _optimize_for_radius(pos_current, r_low)

    dists_center = np.linalg.norm(pos_final, axis=1)
    r_boundary = np.min(1.0 - dists_center)
    dists = cdist(pos_final, pos_final)
    iu = np.triu_indices(N, 1)
    r_pair = np.min(dists[iu]) / 2.0
    r_actual = min(r_boundary, r_pair)

    return r_actual, pos_final


def _simulated_annealing(points, deadline=None):
    """Simulated annealing to escape local minima."""
    N = points.shape[0]
    points = points.copy()
    best_points = points.copy()
    best_score = _score(points)

    T = 0.05
    T_min = 1e-5
    cooling = 0.995
    step_scale = 0.02

    while T > T_min:
        if deadline is not None and time.time() > deadline - 10.0:
            break

        new_points = points.copy()
        # Random perturbation
        noise = np.random.normal(0, step_scale, new_points.shape)
        new_points += noise

        # Project into unit disk
        norms = np.linalg.norm(new_points, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            scale = 0.99 / norms[mask]
            new_points[mask] *= scale[:, np.newaxis]

        new_score = _score(new_points)
        delta = new_score - best_score

        # Accept if better, or with probability exp(delta/T)
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            points = new_points
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()

        T *= cooling

    return best_points


def _local_polish(points, n_iter=500, deadline=None):
    """Gradient-based local optimization with adaptive step size."""
    N = points.shape[0]
    points = points.copy()
    best_score = _score(points)
    step_size = 0.01

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline - 2.0:
            break

        forces = np.zeros_like(points)

        # Pairwise repulsion - vectorized
        dists = cdist(points, points)
        np.fill_diagonal(dists, 1.0)

        close_mask = dists < 0.35
        for i in range(N):
            for j in range(i + 1, N):
                if close_mask[i, j]:
                    d = dists[i, j]
                    force_mag = 0.005 / max(d, 0.001)
                    direction = (points[i] - points[j]) / max(d, 0.001)
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction

        # Boundary attraction
        norms = np.linalg.norm(points, axis=1)
        for i in range(N):
            if norms[i] > 0.85:
                force_mag = 0.005 * (norms[i] - 0.85)
                direction = -points[i] / max(norms[i], 0.001)
                forces[i] += force_mag * direction

        new_points = points + forces * step_size

        # Project into unit disk
        norms = np.linalg.norm(new_points, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            scale = 0.99 / norms[mask]
            new_points[mask] *= scale[:, np.newaxis]

        new_score = _score(new_points)
        if new_score > best_score:
            points = new_points
            best_score = new_score
            step_size = min(step_size * 1.1, 0.05)
        else:
            step_size *= 0.9

    return points


def _final_polish(points, deadline=None):
    """High-precision final polish with very small steps."""
    N = points.shape[0]
    points = points.copy()
    best_score = _score(points)
    step_size = 0.001

    for it in range(2000):
        if deadline is not None and time.time() > deadline - 1.0:
            break

        new_points = points.copy()
        forces = np.zeros_like(points)
        dists = cdist(points, points)
        np.fill_diagonal(dists, 1.0)

        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 0.28:
                    force_mag = 0.001 / max(d, 0.001)
                    direction = (points[i] - points[j]) / max(d, 0.001)
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction

        norms = np.linalg.norm(points, axis=1)
        for i in range(N):
            if norms[i] > 0.95:
                force_mag = 0.001 * (norms[i] - 0.95)
                direction = -points[i] / max(norms[i], 0.001)
                forces[i] += force_mag * direction

        new_points = points + forces * step_size

        norms = np.linalg.norm(new_points, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            scale = 0.999 / norms[mask]
            new_points[mask] *= scale[:, np.newaxis]

        new_score = _score(new_points)
        if new_score > best_score:
            points = new_points
            best_score = new_score
        else:
            step_size *= 0.95

    return points


def construct():
    """Construct optimal packing of 50 circles in unit circle."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_points = None
    best_score = -np.inf

    # Generate diverse initial configurations
    inits = []
    inits.append(_hex_lattice_init(N, jitter=0.01))
    inits.append(_hex_lattice_init(N, jitter=0.01, rotation=15))
    inits.append(_hex_lattice_init(N, jitter=0.01, rotation=30))
    inits.append(_hex_lattice_init(N, jitter=0.02, spacing_factor=0.95))
    inits.append(_hex_lattice_init(N, jitter=0.02, spacing_factor=1.05))

    # Random inits
    for _ in range(2):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        inits.append(np.column_stack((x, y)))

    # Iterate through inits with binary search + SA + polish
    for i, init_pos in enumerate(inits):
        if time.time() > deadline - 30.0:
            break

        # Binary search for max radius
        try:
            r, pos = _find_max_radius(init_pos, deadline=deadline)
            if r > best_score:
                best_score = r
                best_points = pos
        except Exception:
            pass

        # Simulated annealing to escape local minima
        if time.time() < deadline - 20.0:
            try:
                sa_points = _simulated_annealing(best_points, deadline=deadline)
                sa_score = _score(sa_points)
                if sa_score > best_score:
                    best_score = sa_score
                    best_points = sa_points
            except Exception:
                pass

        # Local polish
        if time.time() < deadline - 10.0:
            try:
                polished_points = _local_polish(best_points, n_iter=300, deadline=deadline)
                polished_score = _score(polished_points)
                if polished_score > best_score:
                    best_score = polished_score
                    best_points = polished_points
            except Exception:
                pass

    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 5.0:
        try:
            final_points = _final_polish(best_points, deadline=deadline)
            final_score = _score(final_points)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points
        except Exception:
            pass

    if best_points is None:
        best_points = _hex_lattice_init(N, jitter=0.0)

    # Final check: ensure all points are strictly inside
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms >= 0.999
    if np.any(mask):
        scale = 0.999 / norms[mask]
        best_points[mask] *= scale[:, np.newaxis]

    return best_points
# EVOLVE-BLOCK-END
