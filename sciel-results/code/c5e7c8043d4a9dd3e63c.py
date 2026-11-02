"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Structural change: use a repulsive potential field / force-directed relaxation
to drive circle centers to equilibrium, then refine with LP radii and a
gradient-based polish. The potential field approach avoids explicit pairwise
constraint handling during the main search phase, allowing smoother navigation
of the non-convex configuration space.

Combines:
- Force-directed relaxation with repulsive potentials (structural idea)
- Delaunay-based LP radii solver for exact feasibility (refinement machinery)
- Multi-start with diverse initial configurations (hex, grid, random)
- Gradient-based polish with softplus penalties for final optimization
"""
import numpy as np
import time
import random
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21


def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def _delaunay_edges(xs, ys):
    """Return list of (i, j) edges from Delaunay triangulation of centers."""
    points = np.column_stack((xs, ys))
    try:
        tri = Delaunay(points)
    except Exception:
        return None
    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted((simplex[0], simplex[1]))))
        edges.add(tuple(sorted((simplex[1], simplex[2]))))
        edges.add(tuple(sorted((simplex[0], simplex[2]))))
    return list(edges)


def _solve_lp_radii(xs, ys, w, h):
    """
    Solve LP for per-circle radii given fixed centers and rectangle.
    Maximize sum(r) s.t.  r_i + r_j <= d_ij  (Delaunay edges)
                          r_i <= min(x_i, w-x_i, y_i, h-y_i)
                          r_i >= 0
    Returns array of radii (zeros on failure).
    """
    limits = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
    if np.any(limits < 0):
        return np.zeros(N)

    A_list = []
    b_list = []
    edges = _delaunay_edges(xs, ys)
    if edges is not None:
        for i, j in edges:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(N)
            row[i] = 1
            row[j] = 1
            A_list.append(row)
            b_list.append(dist)

    A_bound = np.eye(N)
    if A_list:
        A_ub = np.vstack([A_list, A_bound])
        b_ub = np.concatenate([b_list, limits])
    else:
        A_ub = A_bound
        b_ub = limits

    c = -np.ones(N)
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success:
            return np.maximum(0.0, res.x)
    except Exception:
        pass
    return np.maximum(0.0, limits * 0.9)


def _make_hex(rows, w, h):
    """Hexagonal layout of given per-row counts, scaled to fit w x h."""
    pts = []
    for k, n in enumerate(rows):
        y = k * np.sqrt(3.0)
        off = 1.0 if k % 2 == 1 else 0.0
        for i in range(n):
            pts.append((i * 2.0 + off, y))
    pts = np.array(pts[:N])
    xs = pts[:, 0]
    ys = pts[:, 1]
    cx = 0.5 * (xs.max() + xs.min())
    cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx
    ys = ys - cy
    wc = xs.max() - xs.min()
    hc = ys.max() - ys.min()
    wt = wc + 2.0
    ht = hc + 2.0
    s = 2.0 / (wt + ht)
    xs_f = xs * s + 0.5 * (s * wt)
    ys_f = ys * s + 0.5 * (s * ht)
    return xs_f, ys_f, s * wt, s * ht


def _make_grid(rows, cols, w, h):
    pts = []
    for j in range(rows):
        for i in range(cols):
            if len(pts) >= N:
                break
            pts.append((i * 2.0, j * 2.0))
    pts = np.array(pts[:N])
    xs = pts[:, 0]
    ys = pts[:, 1]
    cx = 0.5 * (xs.max() + xs.min())
    cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx
    ys = ys - cy
    wt = (xs.max() - xs.min()) + 2.0
    ht = (ys.max() - ys.min()) + 2.0
    s = 2.0 / (wt + ht)
    xs_f = xs * s + 0.5 * (s * wt)
    ys_f = ys * s + 0.5 * (s * ht)
    return xs_f, ys_f, s * wt, s * ht


def _make_random(w, h, seed):
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0.1, 0.9, N) * w
    ys = rng.uniform(0.1, 0.9, N) * h
    return xs, ys, w, h


def _force_directed_relaxation(xs, ys, w, h, deadline, max_iter=500, lr=0.01):
    """
    Force-directed relaxation using repulsive potential field.
    Circles repel each other and are attracted to stay within bounds.
    Returns refined (xs, ys).
    """
    xs = xs.copy()
    ys = ys.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Repulsive forces between circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                # Repulsive force ~ 1/dist^2 (Coulomb-like)
                f = 1.0 / (dist_sq + 1e-6)
                fx[i] += f * dx / dist
                fy[i] += f * dy / dist
                fx[j] -= f * dx / dist
                fy[j] -= f * dy / dist
        
        # Attractive forces toward center (to keep circles in bounds)
        for i in range(N):
            # Boundary repulsion
            margin = 0.05
            if xs[i] < margin:
                fx[i] += (margin - xs[i]) * 10.0
            if xs[i] > w - margin:
                fx[i] -= (xs[i] - (w - margin)) * 10.0
            if ys[i] < margin:
                fy[i] += (margin - ys[i]) * 10.0
            if ys[i] > h - margin:
                fy[i] -= (ys[i] - (h - margin)) * 10.0
        
        # Update positions with adaptive learning rate
        step = lr * (1.0 / (1.0 + 0.01 * it))
        xs += step * fx
        ys += step * fy
        
        # Clamp to bounds
        xs = np.clip(xs, 0.01, w - 0.01)
        ys = np.clip(ys, 0.01, h - 0.01)
    
    return xs, ys


def _polish_gradient(w, h, xs, ys, rs, deadline, max_iter=300):
    """
    Smooth gradient-based polish using L-BFGS-B with softplus penalty.
    Variables: [w, x0..x20, y0..y20, r0..r20]. h = 2 - w.
    Maximizes sum(r_i) subject to boundary and non-overlap via softplus penalties.
    Returns (w, h, xs, ys, rs) or input on failure.
    """
    if time.time() > deadline - 5.0:
        return w, h, xs, ys, rs

    v0 = np.zeros(1 + 3 * N)
    v0[0] = w
    v0[1:1 + N] = xs
    v0[1 + N:1 + 2 * N] = ys
    v0[1 + 2 * N:1 + 3 * N] = rs

    idx_i, idx_j = np.triu_indices(N, k=1)

    def objective_and_grad(v):
        w_val = v[0]
        h_val = 2.0 - w_val
        x = v[1:1 + N]
        y = v[1 + N:1 + 2 * N]
        r = v[1 + 2 * N:1 + 3 * N]

        if w_val < 0.05 or h_val < 0.05:
            return 1e6, np.zeros_like(v)

        obj = -np.sum(r)
        grad = np.zeros_like(v)
        grad[1 + 2 * N:1 + 3 * N] = -1.0

        eps_b = 1e-3
        eps_o = 1e-3
        K_b = 500.0
        K_o = 500.0

        # Boundary penalties
        v_l = r - x
        v_r = r - (w_val - x)
        v_b = r - y
        v_t = r - (h_val - y)

        for v_pen, idx in [(v_l, slice(1, 1 + N)), (v_r, slice(1, 1 + N)),
                            (v_b, slice(1 + N, 1 + 2 * N)), (v_t, slice(1 + N, 1 + 2 * N))]:
            sp = eps_b * np.log1p(np.exp(np.clip(v_pen / eps_b, -20, 20)))
            obj += K_b * np.sum(sp)
            sig = np.where(v_pen / eps_b > 0, 1.0 / (1.0 + np.exp(-v_pen / eps_b)),
                           np.exp(v_pen / eps_b) / (1.0 + np.exp(-v_pen / eps_b)))
            grad[idx] += K_b * sig

        # Overlap penalties
        for k in range(len(idx_i)):
            i, j = idx_i[k], idx_j[k]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
            overlap = r[i] + r[j] - dist
            sp = eps_o * np.log1p(np.exp(np.clip(overlap / eps_o, -20, 20)))
            obj += K_o * sp
            sig = np.where(overlap / eps_o > 0, 1.0 / (1.0 + np.exp(-overlap / eps_o)),
                           np.exp(overlap / eps_o) / (1.0 + np.exp(-overlap / eps_o)))
            g = K_o * sig
            grad[1 + 2 * N + i] += g
            grad[1 + 2 * N + j] += g
            if dist > 1e-12:
                grad[1 + i] -= g * dx / dist
                grad[1 + N + i] -= g * dy / dist
                grad[1 + j] += g * dx / dist
                grad[1 + N + j] += g * dy / dist

        return obj, grad

    def objective_only(v):
        return objective_and_grad(v)[0]

    bounds = [(0.1, 1.9)] + [(0.0, 2.0)] * (2 * N) + [(0.001, 1.0)] * N

    try:
        res = minimize(objective_only, v0, method='L-BFGS-B', jac=None,
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-8})
    except Exception:
        return w, h, xs, ys, rs

    if res.success:
        w_r = res.x[0]
        h_r = 2.0 - w_r
        xs_r = res.x[1:1 + N]
        ys_r = res.x[1 + N:1 + 2 * N]
        rs_r = res.x[1 + 2 * N:1 + 3 * N]
        if w_r > 0.05 and h_r > 0.05:
            return w_r, h_r, xs_r, ys_r, rs_r
    return w, h, xs, ys, rs


def _validate_and_score(xs, ys, rs, w, h):
    """Validate packing and return score (sum of radii) or -inf if invalid."""
    # Boundary check
    for i in range(N):
        if xs[i] - rs[i] < -1e-6 or w - xs[i] - rs[i] < -1e-6:
            return -np.inf
        if ys[i] - rs[i] < -1e-6 or h - ys[i] - rs[i] < -1e-6:
            return -np.inf
    # Overlap check
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-6:
                return -np.inf
    return np.sum(rs)


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    best_val = -np.inf
    best_result = None

    # Try multiple aspect ratios and initial configurations
    aspect_ratios = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0]
    
    for a in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
        
        w = 2.0 / (1.0 + a)
        h = 2.0 * a / (1.0 + a)
        
        # Try different initial configurations
        configs = []
        
        # Hexagonal
        for rows in [[7, 7, 7], [6, 7, 8], [5, 8, 8], [4, 8, 9]]:
            if sum(rows) >= N:
                configs.append(('hex', rows))
        
        # Grid
        for r, c in [(3, 7), (4, 6), (5, 5), (7, 3)]:
            if r * c >= N:
                configs.append(('grid', (r, c)))
        
        # Random
        configs.append(('random', 42))
        configs.append(('random', 123))
        
        for cfg_type, cfg_data in configs:
            if time.time() > deadline - 5.0:
                break
            
            # Generate initial configuration
            if cfg_type == 'hex':
                xs, ys, w_c, h_c = _make_hex(cfg_data, w, h)
            elif cfg_type == 'grid':
                xs, ys, w_c, h_c = _make_grid(cfg_data[0], cfg_data[1], w, h)
            else:
                xs, ys, w_c, h_c = _make_random(w, h, cfg_data)
            
            # Force-directed relaxation
            xs_rel, ys_rel = _force_directed_relaxation(xs, ys, w, h, deadline)
            
            # Solve LP for radii
            rs = _solve_lp_radii(xs_rel, ys_rel, w, h)
            
            # Validate
            val = _validate_and_score(xs_rel, ys_rel, rs, w, h)
            if val > best_val:
                best_val = val
                best_result = (w, h, xs_rel.copy(), ys_rel.copy(), rs.copy())
            
            # Gradient polish
            w_p, h_p, xs_p, ys_p, rs_p = _polish_gradient(w, h, xs_rel, ys_rel, rs, deadline)
            val_p = _validate_and_score(xs_p, ys_p, rs_p, w_p, h_p)
            if val_p > best_val:
                best_val = val_p
                best_result = (w_p, h_p, xs_p.copy(), ys_p.copy(), rs_p.copy())
    
    # If no valid solution found, return a simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append((x, y, r))
        return np.array(cells)
    
    w, h, xs, ys, rs = best_result
    return np.column_stack([xs, ys, rs])
