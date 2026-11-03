"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145.0  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0) using log-sum-exp."""
    return np.logaddexp(0, z / eps) * eps


def _sigmoid(z, eps=1e-4):
    """Smooth derivative of softplus."""
    return 1.0 / (1.0 + np.exp(-z / eps))


def _validate(circles, tol=1e-7):
    """Check circles are inside unit square and non-overlapping."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    n = len(circles)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < r[idx_i] + r[idx_j] - tol):
        return False
    return True


def _fix_boundaries(circles):
    """Project circles inside unit square and ensure non-overlap by shrinking if necessary."""
    circles = circles.copy()
    for i in range(N):
        r = max(0.0, circles[i, 2])
        r = min(r, 0.5)
        circles[i, 2] = r
        circles[i, 0] = np.clip(circles[i, 0], r, 1.0 - r)
        circles[i, 1] = np.clip(circles[i, 1], r, 1.0 - r)

    for _iter in range(10):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                if dist < min_dist:
                    overlap = min_dist - dist
                    shrink = overlap / 2.0 + 1e-5
                    circles[i, 2] = max(0.0, circles[i, 2] - shrink)
                    circles[j, 2] = max(0.0, circles[j, 2] - shrink)
                    circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1.0 - circles[i, 2])
                    circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1.0 - circles[i, 2])
                    circles[j, 0] = np.clip(circles[j, 0], circles[j, 2], 1.0 - circles[j, 2])
                    circles[j, 1] = np.clip(circles[j, 1], circles[j, 2], 1.0 - circles[j, 2])
                    changed = True
        if not changed:
            break

    return circles


def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective and analytic gradient for soft-plus penalty method.
    Returns (value, gradient).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum r)/dr

    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    # Left x boundary
    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _sigmoid(lx, smooth_eps)
    grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
    grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

    # Right x boundary
    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _sigmoid(ux, smooth_eps)
    grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
    grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

    # Left y boundary
    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _sigmoid(ly, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
    grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

    # Right y boundary
    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _sigmoid(uy, smooth_eps)
    grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
    grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

    # Negative radius
    sp_nr = _softplus(-r, smooth_eps)
    sg_nr = _sigmoid(-r, smooth_eps)
    grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

    val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) +
                             np.sum(sp_ly**2) + np.sum(sp_uy**2) +
                             np.sum(sp_nr**2))

    # Pairwise overlaps (vectorized)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlap, smooth_eps)
    sg_ov = _sigmoid(overlap, smooth_eps)

    factor = 2.0 * penalty_weight * sp_ov * sg_ov
    inv_dist = 1.0 / dist

    np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
    np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
    np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
    np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
    np.add.at(grad[2::3], idx_i, factor)
    np.add.at(grad[2::3], idx_j, factor)

    val += penalty_weight * np.sum(sp_ov**2)

    return val, grad


def _get_contact_edges(circles):
    """Get contact edges from Delaunay triangulation of centers."""
    centers = circles[:, :2]
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                a, b = simplex[k], simplex[(k + 1) % 3]
                edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        return None


def _objective_delaunay_grad(params, penalty_weight, smooth_eps=1e-4, contact_edges=None):
    """
    Objective and gradient using Delaunay-based contact graph for overlap constraints.
    Faster than full pairwise when contact graph is sparse.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0

    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _sigmoid(lx, smooth_eps)
    grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
    grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _sigmoid(ux, smooth_eps)
    grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
    grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _sigmoid(ly, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
    grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _sigmoid(uy, smooth_eps)
    grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
    grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

    sp_nr = _softplus(-r, smooth_eps)
    sg_nr = _sigmoid(-r, smooth_eps)
    grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

    val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) +
                             np.sum(sp_ly**2) + np.sum(sp_uy**2) +
                             np.sum(sp_nr**2))

    # Overlaps on contact edges only
    if contact_edges is not None and len(contact_edges) > 0:
        for (i, j) in contact_edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            sp_ov = _softplus(overlap, smooth_eps)
            sg_ov = _sigmoid(overlap, smooth_eps)
            factor = 2.0 * penalty_weight * sp_ov * sg_ov
            inv_dist = 1.0 / max(dist, 1e-12)
            grad[i * 3] += -factor * dx * inv_dist
            grad[j * 3] += factor * dx * inv_dist
            grad[i * 3 + 1] += -factor * dy * inv_dist
            grad[j * 3 + 1] += factor * dy * inv_dist
            grad[i * 3 + 2] += factor
            grad[j * 3 + 2] += factor
            val += penalty_weight * sp_ov**2
    else:
        # Fallback to full pairwise
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        sp_ov = _softplus(overlap, smooth_eps)
        sg_ov = _sigmoid(overlap, smooth_eps)

        factor = 2.0 * penalty_weight * sp_ov * sg_ov
        inv_dist = 1.0 / dist

        np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
        np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
        np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
        np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
        np.add.at(grad[2::3], idx_i, factor)
        np.add.at(grad[2::3], idx_j, factor)

        val += penalty_weight * np.sum(sp_ov**2)

    return val, grad


def _compute_overlap_penalty(xyz):
    """Compute hard overlap penalty for a configuration."""
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    penalty = 0.0
    for i in range(N):
        penalty += max(0, r[i] - x[i])**2
        penalty += max(0, x[i] + r[i] - 1.0)**2
        penalty += max(0, r[i] - y[i])**2
        penalty += max(0, y[i] + r[i] - 1.0)**2
        penalty += max(0, -r[i])**2
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap**2
    return penalty


def _parallel_tempering(xyz0, deadline, n_replicas=6, seed=42):
    """
    Parallel tempering (replica exchange) Monte Carlo for circle packing.
    Simultaneously optimizes positions and radii.
    """
    rng = np.random.default_rng(seed)
    temps = np.array([50.0, 20.0, 8.0, 3.0, 1.0, 0.3])[:n_replicas]

    replicas = []
    for rep in range(n_replicas):
        if rep == 0:
            replicas.append(xyz0.copy())
        else:
            xyz_rep = xyz0.copy()
            xyz_rep[:, 0] += rng.normal(0, 0.03, N)
            xyz_rep[:, 1] += rng.normal(0, 0.03, N)
            xyz_rep[:, 2] *= rng.uniform(0.8, 1.2, N)
            xyz_rep = _fix_boundaries(xyz_rep)
            replicas.append(xyz_rep)

    best_xyz = replicas[0].copy()
    best_score = np.sum(replicas[0][:, 2])

    n_steps = 5000
    swap_interval = 20

    for step in range(n_steps):
        if time.time() > deadline - 2.0:
            break

        cooling_factor = 1.0 - 0.3 * (step / n_steps)
        current_temps = temps * cooling_factor

        for rep in range(n_replicas):
            xyz = replicas[rep]
            T = current_temps[rep]

            move_type = rng.random()

            if move_type < 0.4:
                i = rng.integers(N)
                dx = rng.normal(0, 0.02)
                dy = rng.normal(0, 0.02)
                new_x = np.clip(xyz[i, 0] + dx, 0, 1)
                new_y = np.clip(xyz[i, 1] + dy, 0, 1)

                old_energy = -np.sum(xyz[:, 2]) + 100.0 * _compute_overlap_penalty(xyz)
                xyz_new = xyz.copy()
                xyz_new[i, 0] = new_x
                xyz_new[i, 1] = new_y
                new_energy = -np.sum(xyz_new[:, 2]) + 100.0 * _compute_overlap_penalty(xyz_new)

                delta_e = new_energy - old_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-6)):
                    replicas[rep] = xyz_new

            elif move_type < 0.7:
                i = rng.integers(N)
                dr = rng.normal(0, 0.01)
                new_r = np.clip(xyz[i, 2] + dr, 0.01, 0.45)

                old_energy = -np.sum(xyz[:, 2]) + 100.0 * _compute_overlap_penalty(xyz)
                xyz_new = xyz.copy()
                xyz_new[i, 2] = new_r
                new_energy = -np.sum(xyz_new[:, 2]) + 100.0 * _compute_overlap_penalty(xyz_new)

                delta_e = new_energy - old_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-6)):
                    replicas[rep] = xyz_new

            else:
                i, j = rng.choice(N, size=2, replace=False)
                xyz_new = xyz.copy()
                xyz_new[[i, j], 0] = xyz[[j, i], 0] + rng.normal(0, 0.01, 2)
                xyz_new[[i, j], 1] = xyz[[j, i], 1] + rng.normal(0, 0.01, 2)
                xyz_new[:, 0] = np.clip(xyz_new[:, 0], 0, 1)
                xyz_new[:, 1] = np.clip(xyz_new[:, 1], 0, 1)

                old_energy = -np.sum(xyz[:, 2]) + 100.0 * _compute_overlap_penalty(xyz)
                new_energy = -np.sum(xyz_new[:, 2]) + 100.0 * _compute_overlap_penalty(xyz_new)

                delta_e = new_energy - old_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-6)):
                    replicas[rep] = xyz_new

        if step % swap_interval == 0:
            rep_a = rng.integers(n_replicas - 1)
            rep_b = rep_a + 1

            xyz_a = replicas[rep_a]
            xyz_b = replicas[rep_b]
            T_a = current_temps[rep_a]
            T_b = current_temps[rep_b]

            E_a = -np.sum(xyz_a[:, 2]) + 100.0 * _compute_overlap_penalty(xyz_a)
            E_b = -np.sum(xyz_b[:, 2]) + 100.0 * _compute_overlap_penalty(xyz_b)

            delta = (1.0 / T_b - 1.0 / T_a) * (E_b - E_a)
            if delta < 0 or rng.random() < np.exp(-delta):
                replicas[rep_a], replicas[rep_b] = replicas[rep_b], replicas[rep_a]

        cold_score = np.sum(replicas[0][:, 2])
        if cold_score > best_score and _validate(replicas[0]):
            best_score = cold_score
            best_xyz = replicas[0].copy()

        for rep in range(n_replicas):
            rep_score = np.sum(replicas[rep][:, 2])
            if rep_score > best_score and _validate(replicas[rep]):
                best_score = rep_score
                best_xyz = replicas[rep].copy()

    return best_xyz, best_score


def _initial_hex():
    """Hexagonal packing initial config."""
    cols = 5
    rows = 6
    positions = []
    r = 0.09
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5 / cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])


def _initial_grid():
    """Grid initial config."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5 / side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])


def _initial_corner_weighted(seed=0):
    """Random config with corner preference."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
               [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.08])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])


def _initial_ring(seed=0):
    """Ring-shaped initial config."""
    rng = np.random.default_rng(seed)
    positions = [[0.5, 0.5, 0.08]]
    for ring, n_circles, radius in [(1, 6, 0.10), (2, 10, 0.08), (3, 9, 0.07)]:
        for k in range(n_circles):
            angle = 2 * np.pi * k / n_circles + rng.uniform(-0.1, 0.1)
            dist = ring * 0.14
            x = 0.5 + dist * np.cos(angle)
            y = 0.5 + dist * np.sin(angle)
            positions.append([x, y, radius])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.06])
    return np.array(positions[:N])


def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config: PT-MC then local polish with analytic gradients."""
    best_params = xyz0.copy()
    best_score = np.sum(xyz0[:, 2])

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))

    # Phase 0: Parallel tempering MC (global search)
    remaining = deadline - time.time()
    if remaining > 20:
        pt_budget = deadline - 8.0
        try:
            xyz_pt, score_pt = _parallel_tempering(xyz0, pt_budget, n_replicas=5,
                                                    seed=int(hash(tuple(xyz0[:3].round(4)))) % 10000)
            if score_pt > best_score:
                best_params = xyz_pt.copy()
                best_score = score_pt
        except Exception:
            pass

    # Phase 1: Delaunay-based optimization with analytic gradient
    params = best_params.flatten().copy()
    remaining = deadline - time.time()
    if remaining > 5:
        penalty = 50.0
        contact_edges = _get_contact_edges(best_params)
        try:
            def obj(p):
                v, _ = _objective_delaunay_grad(p, penalty, 1e-4, contact_edges)
                return v

            def grad(p):
                _, g = _objective_delaunay_grad(p, penalty, 1e-4, contact_edges)
                return g

            res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < obj(params):
                params = res.x
                if np.sum(params[2::3]) > best_score:
                    best_params = params.reshape(N, 3).copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 2: Full pairwise with analytic gradient, higher penalty
    remaining = deadline - time.time()
    if remaining > 4:
        penalty = 200.0
        try:
            def obj(p):
                v, _ = _objective_grad(p, penalty, 1e-5)
                return v

            def grad(p):
                _, g = _objective_grad(p, penalty, 1e-5)
                return g

            res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < obj(params):
                params = res.x
                if np.sum(params[2::3]) > best_score:
                    best_params = params.reshape(N, 3).copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 3: Even higher penalty
    remaining = deadline - time.time()
    if remaining > 3:
        penalty = 1000.0
        try:
            def obj(p):
                v, _ = _objective_grad(p, penalty, 1e-6)
                return v

            def grad(p):
                _, g = _objective_grad(p, penalty, 1e-6)
                return g

            res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < obj(params):
                params = res.x
                if np.sum(params[2::3]) > best_score:
                    best_params = params.reshape(N, 3).copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 4: Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 2:
        penalty = 5000.0
        try:
            def obj(p):
                v, _ = _objective_grad(p, penalty, 1e-7)
                return v

            def grad(p):
                _, g = _objective_grad(p, penalty, 1e-7)
                return g

            res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 20})
            if res.fun < obj(params):
                params = res.x
                if np.sum(params[2::3]) > best_score:
                    best_params = params.reshape(N, 3).copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 5: Ultra-high precision final polish
    remaining = deadline - time.time()
    if remaining > 1.5:
        penalty = 20000.0
        try:
            def obj(p):
                v, _ = _objective_grad(p, penalty, 1e-8)
                return v

            def grad(p):
                _, g = _objective_grad(p, penalty, 1e-8)
                return g

            res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 15})
            if res.fun < obj(params):
                params = res.x
                if np.sum(params[2::3]) > best_score:
                    best_params = params.reshape(N, 3).copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass

    return best_params, best_score


def construct():
    global_start = time.time()
    deadline = global_start + 143.0  # 7s safety margin before 150s kill

    # Generate multiple diverse initial configurations
    initial_configs = [
        _initial_hex(),
        _initial_grid(),
        _initial_corner_weighted(seed=42),
        _initial_ring(seed=42),
        _initial_corner_weighted(seed=123),
        _initial_ring(seed=999)
    ]

    # Optimize each initial config with a portion of the time budget
    best_xyz = None
    best_score = -np.inf

    n_configs = len(initial_configs)
    time_per_config = (deadline - global_start) / (n_configs + 1)  # +1 for final polish

    for i, xyz0 in enumerate(initial_configs):
        if time.time() > deadline - 15.0:
            break

        # Allocate time for this config
        config_deadline = min(deadline - 10.0, global_start + (i + 1) * time_per_config)

        try:
            xyz_opt, score = _optimize_single(xyz0, config_deadline)

            if not _validate(xyz_opt):
                xyz_fix = xyz_opt.copy()
                for _ in range(100):
                    if _validate(xyz_fix):
                        break
                    xyz_fix[:, 2] *= 0.95
                    xyz_fix = _fix_boundaries(xyz_fix)
                xyz_opt = xyz_fix

            if _validate(xyz_opt):
                score = np.sum(xyz_opt[:, 2])
                if score > best_score:
                    best_score = score
                    best_xyz = xyz_opt.copy()
        except Exception:
            pass

    # Final polish on the best result
    if best_xyz is not None and _remaining() > 5.0:
        try:
            # Do one more round of optimization on the best result
            final_deadline = deadline - 2.0
            xyz_final, score_final = _optimize_single(best_xyz, final_deadline)

            if not _validate(xyz_final):
                xyz_fix = xyz_final.copy()
                for _ in range(100):
                    if _validate(xyz_fix):
                        break
                    xyz_fix[:, 2] *= 0.95
                    xyz_fix = _fix_boundaries(xyz_fix)
                xyz_final = xyz_fix

            if _validate(xyz_final):
                score_final = np.sum(xyz_final[:, 2])
                if score_final > best_score:
                    best_score = score_final
                    best_xyz = xyz_final.copy()
        except Exception:
            pass

    # Fallback if no valid result
    if best_xyz is None or not _validate(best_xyz):
        best_xyz = _initial_grid()

    return best_xyz

# EVOLVE-BLOCK-END
