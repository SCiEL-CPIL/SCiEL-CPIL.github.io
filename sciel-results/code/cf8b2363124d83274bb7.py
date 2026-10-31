"""
Circle packing in perimeter-4 rectangle.

Structural change: replace the penalty-based objective with a proper
constrained NLP using scipy SLSQP with explicit constraint functions.
Add a physics-based force-directed pre-conditioner with adaptive step
size, and a multi-start strategy with diverse aspect ratios.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 145.0  # safety margin before 150 s kill


def _time_left():
    return DEADLINE - time.time()


# ---------------------------------------------------------------------------
# Evaluation / clamping helpers
# ---------------------------------------------------------------------------

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-8)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-8)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-9 < r[i] + r[j]:
                return None
    return float(np.sum(r))


def _clamp_valid(arr, W, H):
    """Clamp a (N,3) array to be valid in rectangle W×H."""
    cx = np.clip(arr[:, 0], 0.0, W)
    cy = np.clip(arr[:, 1], 0.0, H)
    r = np.maximum(arr[:, 2], 1e-8)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-8)
    return np.column_stack([cx, cy, r])


def _grid_layout(kx, ky, W, H, shrink=0.999):
    cx = np.linspace(0.0, W, kx + 2)[1:-1]
    cy = np.linspace(0.0, H, ky + 2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            r = min(xx, yy, W - xx, H - yy) * shrink
            pts.append([xx, yy, max(r, 1e-5)])
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]


# ---------------------------------------------------------------------------
# Force-directed pre-conditioner (physics-based, adaptive)
# ---------------------------------------------------------------------------

def _force_relax(x0, W, H, max_iter=300, dt_init=0.002):
    """
    Force-directed relaxation: push overlapping circles apart and
    pull boundary-violating circles inward. Adaptive dt with cooling.
    """
    cx = x0[:, 0].copy().astype(float)
    cy = x0[:, 1].copy().astype(float)
    r = x0[:, 2].copy().astype(float)
    n = len(cx)
    dt = dt_init

    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Pairwise overlap forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                req = r[i] + r[j]
                if d < req:
                    if d < 1e-12:
                        angle = random.uniform(0, 2 * np.pi)
                        fx[i] += np.cos(angle)
                        fy[i] += np.sin(angle)
                        fx[j] -= np.cos(angle)
                        fy[j] -= np.sin(angle)
                    else:
                        ov = (req - d) / d
                        fx[i] += ov * dx
                        fy[i] += ov * dy
                        fx[j] -= ov * dx
                        fy[j] -= ov * dy

        # Boundary forces
        for i in range(n):
            if cx[i] < r[i]:
                fx[i] += (r[i] - cx[i]) * 5.0
            if cx[i] > W - r[i]:
                fx[i] -= (cx[i] - (W - r[i])) * 5.0
            if cy[i] < r[i]:
                fy[i] += (r[i] - cy[i]) * 5.0
            if cy[i] > H - r[i]:
                fy[i] -= (cy[i] - (H - r[i])) * 5.0

        # Apply forces
        cx += dt * fx
        cy += dt * fy

        # Clamp to valid region
        cx = np.clip(cx, 0.0, W)
        cy = np.clip(cy, 0.0, H)
        lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
        r = np.minimum(r, lo)
        r = np.maximum(r, 1e-8)

        # Cool down
        dt *= 0.995

        # Check convergence
        max_force = np.max(np.sqrt(fx * fx + fy * fy))
        if max_force < 1e-8:
            break

    return np.column_stack([cx, cy, r])


# ---------------------------------------------------------------------------
# Constrained NLP with explicit constraints (structural change)
# ---------------------------------------------------------------------------

def _make_constraints(W, H):
    """
    Build constraint functions for SLSQP.
    Variables: [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    """
    n = N

    # Objective: maximize sum(r) => minimize -sum(r)
    def objective(v):
        return -np.sum(v[2::3])

    def objective_grad(v):
        g = np.zeros(3 * n)
        g[2::3] = -1.0
        return g

    # Inequality constraints: g(v) >= 0
    # Circle i inside rectangle:
    #   x_i >= r_i  =>  x_i - r_i >= 0
    #   x_i <= W - r_i  =>  W - x_i - r_i >= 0
    #   y_i >= r_i  =>  y_i - r_i >= 0
    #   y_i <= H - r_i  =>  H - y_i - r_i >= 0
    #   r_i >= 0  =>  r_i >= 0
    # Circle i-j non-overlap:
    #   (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
    #   => (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0

    constraints = []

    for i in range(n):
        bi = 3 * i
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i] - v[3*i+2],
            'jac': lambda v, i=i: _grad_boundary(v, i, 'x_lo')
        })
        # W - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: W - v[3*i] - v[3*i+2],
            'jac': lambda v, i=i: _grad_boundary(v, i, 'x_hi')
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i+1] - v[3*i+2],
            'jac': lambda v, i=i: _grad_boundary(v, i, 'y_lo')
        })
        # H - y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: H - v[3*i+1] - v[3*i+2],
            'jac': lambda v, i=i: _grad_boundary(v, i, 'y_hi')
        })
        # r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i+2],
            'jac': lambda v, i=i: _grad_radius(v, i)
        })

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: _overlap_constr(v, i, j),
                'jac': lambda v, i=i, j=j: _overlap_grad(v, i, j)
            })

    return objective, objective_grad, constraints


def _grad_boundary(v, i, side):
    g = np.zeros(3 * N)
    if side == 'x_lo':
        g[3*i] = 1.0
        g[3*i+2] = -1.0
    elif side == 'x_hi':
        g[3*i] = -1.0
        g[3*i+2] = -1.0
    elif side == 'y_lo':
        g[3*i+1] = 1.0
        g[3*i+2] = -1.0
    elif side == 'y_hi':
        g[3*i+1] = -1.0
        g[3*i+2] = -1.0
    return g


def _grad_radius(v, i):
    g = np.zeros(3 * N)
    g[3*i+2] = 1.0
    return g


def _overlap_constr(v, i, j):
    dx = v[3*i] - v[3*j]
    dy = v[3*i+1] - v[3*j+1]
    rs = v[3*i+2] + v[3*j+2]
    return dx*dx + dy*dy - rs*rs


def _overlap_grad(v, i, j):
    g = np.zeros(3 * N)
    dx = v[3*i] - v[3*j]
    dy = v[3*i+1] - v[3*j+1]
    rs = v[3*i+2] + v[3*j+2]
    g[3*i] = 2.0 * dx
    g[3*i+1] = 2.0 * dy
    g[3*j] = -2.0 * dx
    g[3*j+1] = -2.0 * dy
    g[3*i+2] = -2.0 * rs
    g[3*j+2] = -2.0 * rs
    return g


def _optimize_constrained(x0, W, H, max_iter=500):
    """Run SLSQP with explicit constraints."""
    v0 = x0.ravel()
    bounds = [(0.0, W)] * N + [(0.0, H)] * N + [(1e-8, min(W, H))] * N

    obj, obj_grad, cons = _make_constraints(W, H)

    try:
        res = minimize(obj, v0, jac=obj_grad, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': max_iter, 'ftol': 1e-12,
                                'maxls': 100, 'disp': 0})
        if res.success or res.x is not None:
            v_opt = res.x
            x_opt = v_opt.reshape(N, 3)
            s = _evaluate(x_opt, W, H)
            if s is not None:
                return x_opt, s
    except Exception:
        pass

    # Fallback: just return the initial point
    s = _evaluate(x0, W, H)
    return x0, (s if s is not None else 0.0)


# ---------------------------------------------------------------------------
# Multi-start with diverse aspect ratios
# ---------------------------------------------------------------------------

def _make_initial_guess(W, H, layout_type='grid', seed=0):
    """Create an initial guess for the optimizer."""
    rng = random.Random(seed)
    side = int(np.ceil(np.sqrt(N)))

    if layout_type == 'grid':
        cx = np.linspace(0.0, W, side + 2)[1:-1]
        cy = np.linspace(0.0, H, side + 2)[1:-1]
        pts = []
        for yy in cy:
            for xx in cx:
                r = min(xx, yy, W - xx, H - yy) * 0.9
                pts.append([xx, yy, max(r, 1e-5)])
        arr = np.array(pts)
        if len(arr) < N:
            while len(arr) < N:
                arr = np.vstack([arr, [W/2, H/2, 0.01]])
        return arr[:N]
    elif layout_type == 'random':
        pts = []
        for _ in range(N):
            cx = rng.uniform(0.1 * W, 0.9 * W)
            cy = rng.uniform(0.1 * H, 0.9 * H)
            r = min(cx, cy, W - cx, H - cy) * 0.3
            r = max(r, 1e-5)
            pts.append([cx, cy, r])
        return np.array(pts)
    else:  # 'ring' - circles around center
        pts = []
        # Center circle
        r_c = min(W, H) * 0.15
        pts.append([W/2, H/2, r_c])
        # Ring of circles
        n_ring = N - 1
        angle_step = 2 * np.pi / n_ring
        ring_r = min(W, H) * 0.2
        ring_radius = min(W, H) * 0.35
        for k in range(n_ring):
            angle = k * angle_step
            cx = W/2 + ring_radius * np.cos(angle)
            cy = H/2 + ring_radius * np.sin(angle)
            cx = np.clip(cx, ring_r, W - ring_r)
            cy = np.clip(cy, ring_r, H - ring_r)
            r = min(cx, cy, W - cx, H - cy) * 0.8
            r = max(r, 1e-5)
            pts.append([cx, cy, r])
        return np.array(pts)


def _optimize_aspect_ratio(W, H, x_init, max_iter=300):
    """
    Optimize both circle positions/radii AND aspect ratio.
    Variables: [x1, y1, r1, ..., xN, yN, rN, w] where h = 2-w
    """
    n = N
    h = 2.0 - W

    # Build initial vector
    v0 = np.concatenate([x_init.ravel(), [W]])

    # Objective: -sum(r)
    def objective(v):
        r = v[2::3][:n]
        return -np.sum(r)

    def objective_grad(v):
        g = np.zeros(3 * n + 1)
        g[2::3][:n] = -1.0
        return g

    # Constraints
    constraints = []

    # Perimeter constraint is built into h = 2 - w
    # Circle i inside rectangle [0,w] x [0,h]:
    for i in range(n):
        bi = 3 * i
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i] - v[3*i+2],
            'jac': lambda v, i=i: _grad_boundary_free(v, i, 'x_lo')
        })
        # w - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[-1] - v[3*i] - v[3*i+2],
            'jac': lambda v, i=i: _grad_boundary_free(v, i, 'x_hi')
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i+1] - v[3*i+2],
            'jac': lambda v, i=i: _grad_boundary_free(v, i, 'y_lo')
        })
        # h - y_i - r_i >= 0  =>  (2 - w) - y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: (2.0 - v[-1]) - v[3*i+1] - v[3*i+2],
            'jac': lambda v, i=i: _grad_boundary_free(v, i, 'y_hi')
        })
        # r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i+2],
            'jac': lambda v, i=i: _grad_radius_free(v, i)
        })

    # Non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: _overlap_constr_free(v, i, j),
                'jac': lambda v, i=i, j=j: _overlap_grad_free(v, i, j)
            })

    # Bounds
    bounds = [(0.0, 2.0)] * N + [(0.0, 2.0)] * N + [(1e-8, 1.0)] * N + [(0.4, 1.6)]

    try:
        res = minimize(objective, v0, jac=objective_grad, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': max_iter, 'ftol': 1e-12,
                                'maxls': 100, 'disp': 0})
        if res.x is not None:
            v_opt = res.x
            w_opt = v_opt[-1]
            h_opt = 2.0 - w_opt
            if w_opt > 0.3 and h_opt > 0.3:
                x_opt = v_opt[:3*n].reshape(n, 3)
                s = _evaluate(x_opt, w_opt, h_opt)
                if s is not None:
                    return x_opt, w_opt, h_opt, s
    except Exception:
        pass

    return x_init, W, H, _evaluate(x_init, W, H) or 0.0


def _grad_boundary_free(v, i, side):
    g = np.zeros(3 * N + 1)
    if side == 'x_lo':
        g[3*i] = 1.0
        g[3*i+2] = -1.0
    elif side == 'x_hi':
        g[3*i] = -1.0
        g[3*i+2] = -1.0
        g[-1] = 1.0
    elif side == 'y_lo':
        g[3*i+1] = 1.0
        g[3*i+2] = -1.0
    elif side == 'y_hi':
        g[3*i+1] = -1.0
        g[3*i+2] = -1.0
        g[-1] = -1.0
    return g


def _grad_radius_free(v, i):
    g = np.zeros(3 * N + 1)
    g[3*i+2] = 1.0
    return g


def _overlap_constr_free(v, i, j):
    dx = v[3*i] - v[3*j]
    dy = v[3*i+1] - v[3*j+1]
    rs = v[3*i+2] + v[3*j+2]
    return dx*dx + dy*dy - rs*rs


def _overlap_grad_free(v, i, j):
    g = np.zeros(3 * N + 1)
    dx = v[3*i] - v[3*j]
    dy = v[3*i+1] - v[3*j+1]
    rs = v[3*i+2] + v[3*j+2]
    g[3*i] = 2.0 * dx
    g[3*i+1] = 2.0 * dy
    g[3*j] = -2.0 * dx
    g[3*j+1] = -2.0 * dy
    g[3*i+2] = -2.0 * rs
    g[3*j+2] = -2.0 * rs
    return g


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0

    random.seed(42)
    np.random.seed(42)

    # Phase 1: Multi-start with diverse aspect ratios and layouts
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.2, 0.7, 0.9, 1.1]
    layout_types = ['grid', 'random', 'ring']

    candidates = []
    for idx, w0 in enumerate(aspect_ratios):
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        lt = layout_types[idx % len(layout_types)]
        x_init = _make_initial_guess(w0, h0, lt, seed=42 + idx)
        candidates.append((w0, h0, x_init))

    # Add some random aspect ratios
    rng = np.random.default_rng(42)
    for _ in range(6):
        w0 = float(rng.uniform(0.55, 1.45))
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        x_init = _make_initial_guess(w0, h0, 'random', seed=int(rng.integers(1000)))
        candidates.append((w0, h0, x_init))

    for idx, (W, H, x_init) in enumerate(candidates):
        if _time_left() < 10:
            break

        # Force-directed pre-condition
        x_relaxed = _force_relax(x_init, W, H, max_iter=200, dt_init=0.002)
        s_relaxed = _evaluate(x_relaxed, W, H)
        if s_relaxed is None:
            continue

        # Constrained NLP optimization
        x_opt, s_opt = _optimize_constrained(x_relaxed, W, H, max_iter=400)
        if s_opt > best_sum:
            best_sum = s_opt
            best_arr = x_opt
            best_W = W
            best_H = H

    # Phase 2: Free aspect ratio optimization from best result
    if best_arr is not None and _time_left() > 20:
        x_opt, W_opt, H_opt, s_opt = _optimize_aspect_ratio(
            best_W, best_H, best_arr, max_iter=300)
        if s_opt > best_sum:
            best_sum = s_opt
            best_arr = x_opt
            best_W = W_opt
            best_H = H_opt

    # Phase 3: Additional local refinements from best
    if best_arr is not None and _time_left() > 15:
        # Perturb and re-optimize
        for attempt in range(3):
            if _time_left() < 8:
                break
            noise = np.random.randn(*best_arr.shape) * 0.01
            x_pert = best_arr + noise
            x_pert = _clamp_valid(x_pert, best_W, best_H)
            x_relaxed = _force_relax(x_pert, best_W, best_H, max_iter=100, dt_init=0.001)
            x_opt, s_opt = _optimize_constrained(x_relaxed, best_W, best_H, max_iter=200)
            if s_opt > best_sum:
                best_sum = s_opt
                best_arr = x_opt

    # Phase 4: Final polish with force-directed radius growth
    if best_arr is not None and _time_left() > 5:
        x_polished = _force_relax(best_arr, best_W, best_H, max_iter=100, dt_init=0.0005)
        s_polished = _evaluate(x_polished, best_W, best_H)
        if s_polished is not None and s_polished > best_sum:
            best_sum = s_polished
            best_arr = x_polished

    # Final clamping & validation
    if best_arr is None:
        W, H = 1.0, 1.0
        best_arr = _grid_layout(5, 5, W, H, shrink=0.98)
        best_W, best_H = W, H

    best_arr = _clamp_valid(best_arr, best_W, best_H)

    s_final = _evaluate(best_arr, best_W, best_H)
    if s_final is None:
        # Fallback grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Safety shrink
    r = best_arr[:, 2] * (1 - 1e-7)
    best_arr = np.column_stack([best_arr[:, 0], best_arr[:, 1], r])
    best_arr = _clamp_valid(best_arr, best_W, best_H)

    return best_arr
