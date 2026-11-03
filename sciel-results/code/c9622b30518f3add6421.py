"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
import time
from scipy.spatial import Delaunay, Voronoi
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0) using log-sum-exp."""
    return np.logaddexp(0, z / eps) * eps


def _validate(circles, tol=1e-7):
    """Check circles are inside unit square and non-overlapping."""
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
    """Project circles inside unit square."""
    circles = circles.copy()
    for i in range(N):
        r = max(0.0, circles[i, 2])
        lo = r if r < 0.5 else 0.5
        hi = 1 - r if r < 0.5 else 0.5
        circles[i, 0] = np.clip(circles[i, 0], lo, hi)
        circles[i, 1] = np.clip(circles[i, 1], lo, hi)
        circles[i, 2] = r
    return circles


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
        if not edges:
            return None
        return list(edges)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Analytic-gradient objective (STRUCTURAL: reduces L-BFGS-B line-search noise)
# ---------------------------------------------------------------------------

def _objective_grad(params, penalty_weight, smooth_eps=1e-4, contact_edges=None):
    """Return (f, g) for the softplus penalty objective.

    params: flat [x0,y0,r0, x1,y1,r1, ...]
    """
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)

    f = -np.sum(r)

    # Boundary violations (smooth)
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    for term in (lx, ux, ly, uy):
        sp = _softplus(term, smooth_eps)
        f += penalty_weight * np.sum(sp * sp)
    # d/dx of lx = -1 ; d/dx of ux = +1
    gx += penalty_weight * (2 * _softplus(ux, smooth_eps) - 2 * _softplus(lx, smooth_eps))
    gy += penalty_weight * (2 * _softplus(uy, smooth_eps) - 2 * _softplus(ly, smooth_eps))
    gr += penalty_weight * (2 * _softplus(lx, smooth_eps) + 2 * _softplus(ux, smooth_eps)
                            + 2 * _softplus(ly, smooth_eps) + 2 * _softplus(uy, smooth_eps))

    # Negative radius
    sp = _softplus(-r, smooth_eps)
    f += penalty_weight * np.sum(sp * sp)
    gr += penalty_weight * (-2 * sp)

    # Pairwise overlaps
    if contact_edges is not None and len(contact_edges) > 0:
        for (i, j) in contact_edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(max(dx * dx + dy * dy, 1e-12))
            overlap = r[i] + r[j] - dist
            sp = _softplus(overlap, smooth_eps)
            f += penalty_weight * sp * sp
            s = 2 * penalty_weight * sp
            if dist > 1e-9:
                gx[i] += s * dx / dist
                gx[j] -= s * dx / dist
                gy[i] += s * dy / dist
                gy[j] -= s * dy / dist
            gr[i] += s
            gr[j] += s
    else:
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        sp = _softplus(overlap, smooth_eps)
        f += penalty_weight * np.sum(sp * sp)
        s = 2 * penalty_weight * sp
        inv = np.where(dist > 1e-9, 1.0 / dist, 0.0)
        np.add.at(gx, idx_i, s * dx * inv)
        np.add.at(gx, idx_j, -s * dx * inv)
        np.add.at(gy, idx_i, s * dy * inv)
        np.add.at(gy, idx_j, -s * dy * inv)
        np.add.at(gr, idx_i, s)
        np.add.at(gr, idx_j, s)

    g = np.empty(3 * n)
    g[0::3] = gx
    g[1::3] = gy
    g[2::3] = gr
    return f, g


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------

def _voronoi_radii(centers):
    """Radius upper bound per circle from Voronoi cell distance to square edges."""
    x, y = centers[:, 0], centers[:, 1]
    radii = np.zeros(N)
    try:
        vor = Voronoi(centers)
        for i in range(N):
            dists = []
            for region_idx in vor.point_region[i]:
                region = vor.regions[region_idx]
                if len(region) == 0:
                    continue
                for edge in region:
                    a, b = vor.vertices[edge[0]], vor.vertices[edge[1]]
                    ax, ay = a
                    bx, by = b
                    ex, ey = bx - ax, by - ay
                    length2 = ex * ex + ey * ey
                    if length2 < 1e-14:
                        continue
                    t = ((x[i] - ax) * ex + (y[i] - ay) * ey) / length2
                    t = np.clip(t, 0, 1)
                    px, py = ax + t * ex, ay + t * ey
                    dists.append((x[i] - px) ** 2 + (y[i] - py) ** 2)
            if dists:
                radii[i] = np.sqrt(min(dists))
    except Exception:
        pass
    # Always respect the square
    sq = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    radii = np.minimum(radii, sq)
    radii = np.clip(radii, 0.02, 0.4)
    return radii


def _initial_hex():
    """Hexagonal packing initial config with Voronoi-informed radii."""
    cols = 5
    rows = 6
    positions = []
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5 / cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y, 0.09])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.045])
    positions = np.array(positions[:N])
    positions[:, 2] = _voronoi_radii(positions[:, :2])
    return positions


def _initial_grid():
    """Grid initial config with Voronoi-informed radii."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y, 0.09])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.045])
    positions = np.array(positions[:N])
    positions[:, 2] = _voronoi_radii(positions[:, :2])
    return positions


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
    positions = np.array(positions[:N])
    positions[:, 2] = _voronoi_radii(positions[:, :2])
    return positions


def _force_relax(circles, time_limit=8.0):
    """Physics-based relaxation: push overlapping circles apart and grow radii.

    Produces a valid, well-separated starting point for the L-BFGS refinement.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = N
    i_idx, j_idx = np.triu_indices(n, k=1)

    start = time.time()
    it = 0
    while time.time() - start < time_limit:
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
        overlap = r[i_idx] + r[j_idx] - dist
        if not np.any(overlap > 1e-9):
            # Grow radii a little while valid
            grow = 0.002
            nr = r + grow
            nr = np.minimum(nr, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
            if np.any(nr > r):
                r = nr
                continue
            break
        for k in range(len(i_idx)):
            if overlap[k] > 1e-9:
                i = i_idx[k]
                j = j_idx[k]
                if dist[k] < 1e-9:
                    dxk, dyk = 1e-6, 0.0
                else:
                    dxk, dyk = dx[k] / dist[k], dy[k] / dist[k]
                push = overlap[k] * 0.5
                x[i] -= push * dxk
                y[i] -= push * dyk
                x[j] += push * dxk
                y[j] += push * dyk
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
        it += 1
        if it > 5000:
            break

    # Gentle radius growth sweep while staying valid
    for _ in range(200):
        if time.time() - start > time_limit:
            break
        grow = 0.0015
        nr = r + grow
        nr = np.minimum(nr, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
        # check overlaps with grown radii
        ok = True
        for k in range(len(i_idx)):
            d = np.sqrt((x[i_idx[k]] - x[j_idx[k]]) ** 2 + (y[i_idx[k]] - y[j_idx[k]]) ** 2)
            if d < nr[i_idx[k]] + nr[j_idx[k]] - 1e-9:
                ok = False
                break
        if ok:
            r = nr
        else:
            break

    return np.column_stack([x, y, r])


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

_BOUNDS = [(0, 1), (0, 1), (0, 0.5)] * N


def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config with adaptive penalty + gradient."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])

    def _try(res, fval, cur):
        nonlocal best_params, best_score
        if res is not None and res.x is not None:
            cand = res.x
            cand_score = np.sum(cand[2::3])
            if cand_score > best_score:
                best_score = cand_score
                best_params = cand.copy()

    # Phase 1: Low penalty, full pairwise, analytic gradient
    remaining = deadline - time.time()
    if remaining > 4:
        try:
            res = minimize(_objective_grad, params, method='L-BFGS-B',
                          jac=True, bounds=_BOUNDS, args=(5.0, 1e-3, None),
                          options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-10})
            _try(res, None, None)
            params = best_params.copy()
        except Exception:
            pass

    # Phase 2: Medium penalty
    remaining = deadline - time.time()
    if remaining > 4:
        try:
            res = minimize(_objective_grad, params, method='L-BFGS-B',
                          jac=True, bounds=_BOUNDS, args=(50.0, 1e-4, None),
                          options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-10})
            _try(res, None, None)
            params = best_params.copy()
        except Exception:
            pass

    # Phase 3: High penalty with DELAUNAY contact graph that REBUILDS as geometry moves
    remaining = deadline - time.time()
    if remaining > 6:
        circles = best_params.reshape(N, 3)
        contact_edges = _get_contact_edges(circles)
        for sub in range(4):
            if time.time() > deadline - 2:
                break
            if contact_edges is None:
                contact_edges = _get_contact_edges(best_params.reshape(N, 3))
            try:
                res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                              jac=True, bounds=_BOUNDS,
                              args=(200.0, 1e-5, contact_edges),
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                _try(res, None, None)
            except Exception:
                break
            # Rebuild contact graph from the (possibly) improved geometry
            new_edges = _get_contact_edges(best_params.reshape(N, 3))
            if new_edges is not None:
                contact_edges = new_edges
            else:
                break

    # Final polish with very high penalty
    remaining = deadline - time.time()
    if remaining > 3:
        contact_edges = _get_contact_edges(best_params.reshape(N, 3))
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, bounds=_BOUNDS,
                          args=(1000.0, 1e-6, contact_edges),
                          options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-14})
            _try(res, None, None)
        except Exception:
            pass

    # Validate and fix
    best_circles = best_params.reshape(N, 3)
    best_circles = _fix_boundaries(best_circles)
    if not _validate(best_circles):
        best_circles[:, 2] *= 0.99
        best_circles = _fix_boundaries(best_circles)
        if not _validate(best_circles):
            best_circles = xyz0.copy()

    return best_circles


def construct():
    global _start_time
    _start_time = time.time()

    deadline = _start_time + _TIME_LIMIT - 2  # 2s safety margin

    inits = [
        _initial_hex(),
        _initial_grid(),
        _initial_corner_weighted(seed=42),
        _initial_corner_weighted(seed=7),
    ]

    best_circles = None
    best_val = -np.inf

    for idx, init in enumerate(inits):
        if time.time() > deadline - 10:
            break

        # Force-relax to get a valid, well-separated start (bounded time)
        relax_budget = 6.0 if idx < 2 else 4.0
        if time.time() < deadline - 12:
            try:
                init = _force_relax(init, time_limit=relax_budget)
            except Exception:
                pass

        remaining = deadline - time.time()
        if remaining < 8:
            break

        try:
            result = _optimize_single(init, deadline)
            val = np.sum(result[:, 2])
            if val > best_val:
                best_val = val
                best_circles = result
        except Exception:
            continue

    if best_circles is None or not _validate(best_circles):
        best_circles = _initial_grid()
        best_circles = _fix_boundaries(best_circles)

    return best_circles

# EVOLVE-BLOCK-END
