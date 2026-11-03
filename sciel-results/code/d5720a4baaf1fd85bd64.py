"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Combined approach: robust penalty-based optimization with Delaunay contact graphs,
analytic gradients, multi-start, plus a local "rigid body" refinement pass that
exploits the contact graph to perform coordinated moves (structural change)."""
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 142  # safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _softplus(z, eps=1e-4):
    return np.logaddexp(0, z / eps) * eps


def _validate(circles, tol=1e-7):
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r < -tol) or np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _fix_boundaries(circles):
    circles = circles.copy()
    for i in range(N):
        r = max(0.0, circles[i, 2])
        r = min(r, 0.5)
        circles[i, 2] = r
        circles[i, 0] = np.clip(circles[i, 0], r, 1.0 - r)
        circles[i, 1] = np.clip(circles[i, 1], r, 1.0 - r)
    for _iter in range(20):
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


def _objective_penalty(params, penalty_weight, smooth_eps=1e-4):
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    penalty = 0.0
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    penalty += np.sum(_softplus(lx, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ux, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ly, smooth_eps) ** 2)
    penalty += np.sum(_softplus(uy, smooth_eps) ** 2)
    penalty += np.sum(_softplus(-r, smooth_eps) ** 2)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)
    return -np.sum(r) + penalty_weight * penalty


def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0

    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    def _sigmoid(z, eps):
        return 1.0 / (1.0 + np.exp(-z / eps))

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

    val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) + np.sum(sp_ly**2) +
                              np.sum(sp_uy**2) + np.sum(sp_nr**2) + np.sum(sp_ov**2))
    return val, grad


def _get_contact_edges(circles):
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


def _objective_contact(params, penalty_weight, smooth_eps=1e-4, contact_edges=None):
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    penalty = 0.0
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    penalty += np.sum(_softplus(lx, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ux, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ly, smooth_eps) ** 2)
    penalty += np.sum(_softplus(uy, smooth_eps) ** 2)
    penalty += np.sum(_softplus(-r, smooth_eps) ** 2)

    if contact_edges is not None and len(contact_edges) > 0:
        for (i, j) in contact_edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            penalty += _softplus(overlap, smooth_eps) ** 2
    else:
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)

    return -np.sum(r) + penalty_weight * penalty


def _rigid_body_refinement(circles, deadline, n_trials=50):
    """
    Structural refinement: exploit contact graph rigidity.
    For each contact edge, try coordinated moves that preserve the contact distance
    while pushing other circles outward to create room for radius growth.
    This is a discrete angular placement search over the contact graph.
    """
    best = circles.copy()
    best_score = np.sum(best[:, 2])

    for trial in range(n_trials):
        if time.time() > deadline - 3:
            break

        current = best.copy()
        edges = _get_contact_edges(current)
        if edges is None or len(edges) == 0:
            break

        # Pick a random contact edge
        i, j = edges[np.random.randint(len(edges))]

        # Direction from i to j
        dx = current[j, 0] - current[i, 0]
        dy = current[j, 1] - current[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < 1e-10:
            continue
        ux, uy = dx / dist, dy / dist

        # Try rotating the pair slightly and expanding radii
        for angle in [-0.05, -0.02, 0.02, 0.05]:
            # Rotate the direction
            ca, sa = np.cos(angle), np.sin(angle)
            nux = ux * ca - uy * sa
            nuy = ux * sa + uy * ca

            # Try moving i and j apart along the new direction
            step = 0.003
            new_i = current[i].copy()
            new_j = current[j].copy()
            new_i[0] -= nux * step
            new_i[1] -= nuy * step
            new_j[0] += nux * step
            new_j[1] += nuy * step

            # Try to increase both radii slightly
            dr = 0.002
            new_i[2] = min(0.5, new_i[2] + dr)
            new_j[2] = min(0.5, new_j[2] + dr)

            # Clip to boundaries
            for idx in [i, j]:
                r = max(0.0, current[idx, 2])
                current[idx, 0] = np.clip(current[idx, 0], r, 1.0 - r)
                current[idx, 1] = np.clip(current[idx, 1], r, 1.0 - r)

            # Apply the move
            current[i] = new_i
            current[j] = new_j

            # Fix boundaries and overlaps
            current = _fix_boundaries(current)

            if _validate(current):
                score = np.sum(current[:, 2])
                if score > best_score + 1e-8:
                    best_score = score
                    best = current.copy()

        # Also try pure radius growth with contact graph aware repair
        current = best.copy()
        # Identify circles that are "free" (not touching boundary or other circles tightly)
        for k in range(N):
            if time.time() > deadline - 3:
                break
            r_old = current[k, 2]
            current[k, 2] = min(0.5, r_old + 0.003)
            current = _fix_boundaries(current)
            if _validate(current):
                score = np.sum(current[:, 2])
                if score > best_score + 1e-8:
                    best_score = score
                    best = current.copy()
                else:
                    current[k, 2] = r_old  # revert

    return best, best_score


def _initial_hex():
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


def _initial_voronoi(seed=0):
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    radii = np.zeros(N)
    for i in range(N):
        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([points[i, 0], 1 - points[i, 0], points[i, 1], 1 - points[i, 1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.95
    return np.column_stack([points[:, 0], points[:, 1], radii])


def _optimize_single(xyz0, deadline, use_grad=True):
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])

    bounds = [(0, 1), (0, 1), (0, 0.5)] * N

    # Phase 1: Low penalty
    remaining = deadline - time.time()
    if remaining > 5:
        penalty = 5.0
        try:
            if use_grad:
                res = minimize(lambda p: _objective_grad(p, penalty, 1e-3)[0], params,
                              jac=lambda p: _objective_grad(p, penalty, 1e-3)[1],
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
            else:
                res = minimize(_objective_penalty, params, method='L-BFGS-B',
                              bounds=bounds, args=(penalty, 1e-3),
                              options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_penalty(params, penalty, 1e-3):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 2: Medium penalty
    remaining = deadline - time.time()
    if remaining > 5:
        penalty = 50.0
        try:
            if use_grad:
                res = minimize(lambda p: _objective_grad(p, penalty, 1e-4)[0], params,
                              jac=lambda p: _objective_grad(p, penalty, 1e-4)[1],
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
            else:
                res = minimize(_objective_penalty, params, method='L-BFGS-B',
                              bounds=bounds, args=(penalty, 1e-4),
                              options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_penalty(params, penalty, 1e-4):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 3: High penalty with Delaunay contact graph
    remaining = deadline - time.time()
    if remaining > 5:
        penalty = 500.0
        circles = np.column_stack([best_params[0::3], best_params[1::3], best_params[2::3]])
        contact_edges = _get_contact_edges(circles)
        try:
            res = minimize(_objective_contact, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4, contact_edges),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_contact(params, penalty, 1e-4, contact_edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 4: Rigid body refinement (structural change)
    remaining = deadline - time.time()
    if remaining > 8:
        try:
            circles = np.column_stack([best_params[0::3], best_params[1::3], best_params[2::3]])
            refined, refined_score = _rigid_body_refinement(circles, deadline, n_trials=80)
            if refined_score > best_score:
                best_params = refined.flatten()
                best_score = refined_score
        except Exception:
            pass

    # Phase 5: Final polish
    remaining = deadline - time.time()
    if remaining > 3:
        penalty = 1000.0
        try:
            res = minimize(_objective_penalty, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5),
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_penalty(best_params, penalty, 1e-5):
                best_params = res.x
                best_score = np.sum(best_params[2::3])
        except Exception:
            pass

    return best_params, best_score


def construct():
    start_time = time.time()
    deadline = _start_time + _TIME_LIMIT

    inits = [
        _initial_hex(),
        _initial_grid(),
        _initial_corner_weighted(seed=42),
        _initial_corner_weighted(seed=123),
        _initial_voronoi(seed=7),
        _initial_voronoi(seed=1234),
    ]

    best_overall = None
    best_score_overall = -np.inf

    for i, xyz0 in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 8:
            break

        alloc = min(remaining - 5, 22)
        local_deadline = time.time() + alloc
        use_grad = (i % 2 == 0)

        params, score = _optimize_single(xyz0, local_deadline, use_grad=use_grad)

        circles = np.column_stack([params[0::3], params[1::3], params[2::3]])
        if _validate(circles):
            if score > best_score_overall:
                best_score_overall = score
                best_overall = circles.copy()
        else:
            circles_fixed = _fix_boundaries(circles)
            if _validate(circles_fixed):
                score_fixed = np.sum(circles_fixed[:, 2])
                if score_fixed > best_score_overall:
                    best_score_overall = score_fixed
                    best_overall = circles_fixed.copy()

    if best_overall is not None:
        repaired = _fix_boundaries(best_overall)
        if _validate(repaired):
            best_overall = repaired
        else:
            best_overall = None

    if best_overall is None:
        result = np.zeros((N, 3))
        side = 5
        for idx in range(N):
            i = idx % side
            j = idx // side
            result[idx, 0] = (i + 0.5) / side
            result[idx, 1] = (j + 0.5) / side
            result[idx, 2] = 0.5 / side * 0.8
        return result

    return best_overall

# EVOLVE-BLOCK-END
