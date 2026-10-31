"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x):
    """Objective: minimize negative sum of radii."""
    n = N
    radii = x[2*n:]
    radii = np.maximum(radii, 1e-4)
    return -np.sum(radii)


def _gradient_objective(x):
    """Gradient of objective."""
    n = N
    radii = x[2*n:]
    radii = np.maximum(radii, 1e-4)
    grad = np.zeros_like(x)
    grad[2*n:] = -1.0
    return grad


def _pairwise_constraints(x, i, j):
    """Constraint: distance between i and j >= r_i + r_j. Returns (g, J)."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    dx = pos[i, 0] - pos[j, 0]
    dy = pos[i, 1] - pos[j, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[i] + radii[j]
    min_dist_sq = min_dist * min_dist
    
    # g = dist_sq - min_dist_sq >= 0
    g = dist_sq - min_dist_sq
    
    # Jacobian w.r.t. x
    J = np.zeros(2 * n + n)
    if dist_sq > 1e-12:
        dist = np.sqrt(dist_sq)
        # d(dist_sq)/d(pos_i) = 2*dx, d(dist_sq)/d(pos_j) = -2*dx
        J[2*i] = 2 * dx
        J[2*i + 1] = 2 * dy
        J[2*j] = -2 * dx
        J[2*j + 1] = -2 * dy
    # d(min_dist_sq)/d(r_i) = 2*r_i, d(min_dist_sq)/d(r_j) = 2*r_j
    # But we want d(dist_sq - min_dist_sq)/d(r_i) = -2*r_i
    J[2*n + i] = -2 * radii[i]
    J[2*n + j] = -2 * radii[j]
    
    return g, J


def _boundary_constraints(x, i):
    """Boundary constraints for circle i. Returns list of (g, J)."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    r = radii[i]
    px = pos[i, 0]
    py = pos[i, 1]
    
    constraints = []
    
    # x - r >= 0
    g = px - r
    J = np.zeros(3 * n)
    J[2*i] = 1.0
    J[2*n + i] = -1.0
    constraints.append((g, J))
    
    # 1 - x - r >= 0
    g = 1.0 - px - r
    J = np.zeros(3 * n)
    J[2*i] = -1.0
    J[2*n + i] = -1.0
    constraints.append((g, J))
    
    # y - r >= 0
    g = py - r
    J = np.zeros(3 * n)
    J[2*i + 1] = 1.0
    J[2*n + i] = -1.0
    constraints.append((g, J))
    
    # 1 - y - r >= 0
    g = 1.0 - py - r
    J = np.zeros(3 * n)
    J[2*i + 1] = -1.0
    J[2*n + i] = -1.0
    constraints.append((g, J))
    
    # r >= 0 (handled by bounds)
    
    return constraints


def _all_constraints(x):
    """Return all constraints as a flat array and Jacobian."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    g_list = []
    J_rows = []
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            g, J = _pairwise_constraints(x, i, j)
            g_list.append(g)
            J_rows.append(J)
    
    # Boundary constraints
    for i in range(n):
        for g, J in _boundary_constraints(x, i):
            g_list.append(g)
            J_rows.append(J)
    
    g = np.array(g_list)
    J = np.array(J_rows) if J_rows else np.zeros((0, 3*n))
    
    return g, J


def _initial_hexagonal():
    """Create a hexagonal packing pattern."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.092
    x_offset = 0.1
    y_spacing = r * np.sqrt(3)
    
    # Row 0: 5 circles
    for i in range(5):
        idx = i
        pos[idx, 0] = x_offset + i * 2 * r
        pos[idx, 1] = x_offset
        radii[idx] = r
    
    # Row 1: 4 circles
    for i in range(4):
        idx = 5 + i
        pos[idx, 0] = x_offset + r + i * 2 * r
        pos[idx, 1] = x_offset + y_spacing
        radii[idx] = r
    
    # Row 2: 5 circles
    for i in range(5):
        idx = 9 + i
        pos[idx, 0] = x_offset + i * 2 * r
        pos[idx, 1] = x_offset + 2 * y_spacing
        radii[idx] = r
    
    # Row 3: 4 circles
    for i in range(4):
        idx = 14 + i
        pos[idx, 0] = x_offset + r + i * 2 * r
        pos[idx, 1] = x_offset + 3 * y_spacing
        radii[idx] = r
    
    # Row 4: 5 circles
    for i in range(5):
        idx = 18 + i
        pos[idx, 0] = x_offset + i * 2 * r
        pos[idx, 1] = x_offset + 4 * y_spacing
        radii[idx] = r
    
    # 26th circle
    pos[25, 0] = 0.5
    pos[25, 1] = 0.96
    radii[25] = 0.045
    
    return pos, radii


def _initial_corner_heavy():
    """Corner-heavy initialization: larger circles in corners."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 4 corner circles
    corner_r = 0.11
    corners = [(corner_r, corner_r), (1-corner_r, corner_r),
               (corner_r, 1-corner_r), (1-corner_r, 1-corner_r)]
    for i, (cx, cy) in enumerate(corners):
        pos[i, 0] = cx
        pos[i, 1] = cy
        radii[i] = corner_r
    
    # Edge circles
    edge_r = 0.095
    edges = [
        (0.5, corner_r), (corner_r, 0.5),
        (1-corner_r, 0.5), (0.5, 1-corner_r),
        (0.3, corner_r), (0.7, corner_r),
        (corner_r, 0.3), (corner_r, 0.7),
        (1-corner_r, 0.3), (1-corner_r, 0.7),
        (0.3, 1-corner_r), (0.7, 1-corner_r)
    ]
    for i, (ex, ey) in enumerate(edges):
        idx = 4 + i
        pos[idx, 0] = ex
        pos[idx, 1] = ey
        radii[idx] = edge_r
    
    # Center area
    center_r = 0.085
    center_pts = [
        (0.5, 0.5), (0.35, 0.35), (0.65, 0.35),
        (0.35, 0.65), (0.65, 0.65), (0.5, 0.3),
        (0.5, 0.7), (0.3, 0.5), (0.7, 0.5),
        (0.4, 0.4), (0.6, 0.4), (0.4, 0.6),
        (0.6, 0.6)
    ]
    for i, (cx, cy) in enumerate(center_pts):
        idx = 16 + i
        pos[idx, 0] = cx
        pos[idx, 1] = cy
        radii[idx] = center_r
    
    return pos, radii


def _initial_grid():
    """5x5 grid initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.095
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = r
    
    pos[25, 0] = 0.92
    pos[25, 1] = 0.92
    radii[25] = 0.07
    
    return pos, radii


def _initial_random(seed=42):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, size=(N, 2))
    radii = rng.uniform(0.05, 0.12, size=N)
    return pos, radii


def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        valid = True
        
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(pos[i, 0] * 0.99, 1e-4)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 0]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], pos[i, 1] * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 1]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] = max(radii[i] * scale, 1e-4)
                    radii[j] = max(radii[j] * scale, 1e-4)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _trust_constr_solve(pos, radii, deadline):
    """Solve using trust-constr with explicit constraints."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.3))
    
    # Define constraint functions for trust-constr
    # trust-constr uses scipy.optimize.NonlinearConstraint
    from scipy.optimize import NonlinearConstraint
    
    # Pairwise constraints
    def pairwise_func(x):
        n = N
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                min_dist = radii[i] + radii[j]
                vals.append(dist_sq - min_dist * min_dist)
        return np.array(vals)
    
    def pairwise_jac(x):
        n = N
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        J = np.zeros((n*(n-1)//2, 3*n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                if dist_sq > 1e-12:
                    J[idx, 2*i] = 2 * dx
                    J[idx, 2*i + 1] = 2 * dy
                    J[idx, 2*j] = -2 * dx
                    J[idx, 2*j + 1] = -2 * dy
                J[idx, 2*n + i] = -2 * radii[i]
                J[idx, 2*n + j] = -2 * radii[j]
                idx += 1
        return J
    
    # Boundary constraints
    def boundary_func(x):
        n = N
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        vals = []
        for i in range(n):
            vals.append(pos[i, 0] - radii[i])
            vals.append(1.0 - pos[i, 0] - radii[i])
            vals.append(pos[i, 1] - radii[i])
            vals.append(1.0 - pos[i, 1] - radii[i])
        return np.array(vals)
    
    def boundary_jac(x):
        n = N
        J = np.zeros((4*n, 3*n))
        for i in range(n):
            J[4*i, 2*i] = 1.0
            J[4*i, 2*n + i] = -1.0
            J[4*i + 1, 2*i] = -1.0
            J[4*i + 1, 2*n + i] = -1.0
            J[4*i + 2, 2*i + 1] = 1.0
            J[4*i + 2, 2*n + i] = -1.0
            J[4*i + 3, 2*i + 1] = -1.0
            J[4*i + 3, 2*n + i] = -1.0
        return J
    
    n_pairwise = N * (N - 1) // 2
    n_boundary = 4 * N
    
    cons = [
        NonlinearConstraint(pairwise_func, 0, np.inf, jac=pairwise_jac),
        NonlinearConstraint(boundary_func, 0, np.inf, jac=boundary_jac)
    ]
    
    try:
        result = minimize(
            _objective, x0,
            jac=_gradient_objective,
            method='trust-constr',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'gtol': 1e-10, 'xtol': 1e-12,
                     'barrier_tol': 1e-8, 'hessiar_tol': 1e-8,
                     'verbose': 0}
        )
        
        if result.success or result.fun < _objective(x0):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-5)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp, np.sum(radii_pp)
    except Exception:
        pass
    
    return None, None, -np.inf


def _slsqp_polish(pos, radii, deadline):
    """SLSQP polish with tight bounds."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    # Tight bounds around current positions
    bounds = []
    for i in range(N):
        x_lo = max(0.0, pos[i, 0] - 0.03)
        x_hi = min(1.0, pos[i, 0] + 0.03)
        y_lo = max(0.0, pos[i, 1] - 0.03)
        y_hi = min(1.0, pos[i, 1] + 0.03)
        r_lo = max(1e-5, radii[i] * 0.9)
        r_hi = min(0.3, radii[i] * 1.1)
        bounds.append((x_lo, x_hi))
        bounds.append((y_lo, y_hi))
    for i in range(N):
        bounds.append((r_lo, r_hi))
    
    # Use penalty-based objective for SLSQP (simpler constraint handling)
    def obj_slsqp(x):
        n = N
        p = x[:2*n].reshape(n, 2)
        r = np.maximum(x[2*n:], 1e-4)
        pen = 0.0
        for i in range(n):
            pen += max(0, r[i] - p[i,0])**2
            pen += max(0, p[i,0] - (1-r[i]))**2
            pen += max(0, r[i] - p[i,1])**2
            pen += max(0, p[i,1] - (1-r[i]))**2
        for i in range(n):
            for j in range(i+1, n):
                d2 = (p[i,0]-p[j,0])**2 + (p[i,1]-p[j,1])**2
                md = r[i] + r[j]
                if d2 < md*md:
                    pen += (md*md - d2)**2 / (4*md*md)
        return -np.sum(r) + 500.0 * pen
    
    try:
        result = minimize(obj_slsqp, x0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-14, 'eps': 1e-10})
        
        if result.fun < obj_slsqp(x0) - 1e-12:
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-5)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp, np.sum(radii_pp)
    except Exception:
        pass
    
    return None, None, -np.inf


def _lbfgs_refine(pos, radii, deadline):
    """L-BFGS-B refinement with penalty objective."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.3))
    
    def obj_pen(x):
        n = N
        p = x[:2*n].reshape(n, 2)
        r = np.maximum(x[2*n:], 1e-4)
        pen = 0.0
        for i in range(n):
            pen += max(0, r[i] - p[i,0])**2
            pen += max(0, p[i,0] - (1-r[i]))**2
            pen += max(0, r[i] - p[i,1])**2
            pen += max(0, p[i,1] - (1-r[i]))**2
        for i in range(n):
            for j in range(i+1, n):
                d2 = (p[i,0]-p[j,0])**2 + (p[i,1]-p[j,1])**2
                md = r[i] + r[j]
                if d2 < md*md:
                    pen += (md*md - d2)**2 / (4*md*md)
        return -np.sum(r) + 1000.0 * pen
    
    try:
        result = minimize(obj_pen, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
        
        if result.fun < obj_pen(x0) - 1e-12:
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-5)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp, np.sum(radii_pp)
    except Exception:
        pass
    
    return None, None, -np.inf


def _differential_evolution_run(deadline):
    """Run differential evolution for global search."""
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.2))
    
    def obj_de(x):
        n = N
        p = x[:2*n].reshape(n, 2)
        r = np.maximum(x[2*n:], 1e-4)
        pen = 0.0
        for i in range(n):
            pen += max(0, r[i] - p[i,0])**2
            pen += max(0, p[i,0] - (1-r[i]))**2
            pen += max(0, r[i] - p[i,1])**2
            pen += max(0, p[i,1] - (1-r[i]))**2
        for i in range(n):
            for j in range(i+1, n):
                d2 = (p[i,0]-p[j,0])**2 + (p[i,1]-p[j,1])**2
                md = r[i] + r[j]
                if d2 < md*md:
                    pen += (md*md - d2)**2 / (4*md*md)
        return -np.sum(r) + 500.0 * pen
    
    de_deadline = deadline - 30
    
    try:
        result = differential_evolution(
            obj_de, bounds,
            maxiter=50, popsize=10,
            tol=1e-6, mutation=(0.5, 1.5), recombination=0.7,
            seed=42, polish=False,
            updating='deferred',
            callback=lambda xk, conv: (time.time() > de_deadline)
        )
        
        if result.fun < 0:
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp, np.sum(radii_pp)
    except Exception:
        pass
    
    return None, None, -np.inf


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    def _try_update(pos, radii, score):
        nonlocal best_pos, best_radii, best_score
        if score > best_score + 1e-12:
            best_score = score
            best_pos = pos.copy()
            best_radii = radii.copy()
    
    # Build initial configurations
    init_configs = []
    
    pos, radii = _initial_corner_heavy()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_grid()
    init_configs.append((pos, radii))
    
    for seed in [42, 123, 456]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii))
    
    # Perturbed versions
    rng = np.random.RandomState(99)
    pos, radii = _initial_corner_heavy()
    pos_pert = pos + rng.normal(0, 0.015, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    pos, radii = _initial_hexagonal()
    pos_pert = pos + rng.normal(0, 0.015, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Try each initial config with trust-constr
    config_order = [0, 5, 1, 6, 2, 3, 4]
    for idx in config_order:
        if time.time() > deadline - 40:
            break
        if idx >= len(init_configs):
            continue
        
        pos, radii = init_configs[idx]
        
        # Post-process initial to ensure validity
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            _try_update(pos_pp, radii_pp, np.sum(radii_pp))
        
        # Trust-constr solve
        if time.time() < deadline - 35:
            tc_pos, tc_radii, tc_score = _trust_constr_solve(pos_pp, radii_pp, deadline)
            if tc_pos is not None:
                _try_update(tc_pos, tc_radii, tc_score)
        
        # L-BFGS-B refine
        if time.time() < deadline - 25:
            lb_pos, lb_radii, lb_score = _lbfgs_refine(pos_pp, radii_pp, deadline)
            if lb_pos is not None:
                _try_update(lb_pos, lb_radii, lb_score)
        
        # SLSQP polish
        if time.time() < deadline - 15:
            sl_pos, sl_radii, sl_score = _slsqp_polish(pos_pp, radii_pp, deadline)
            if sl_pos is not None:
                _try_update(sl_pos, sl_radii, sl_score)
    
    # Differential evolution if time permits
    if time.time() < deadline - 35:
        de_pos, de_radii, de_score = _differential_evolution_run(deadline)
        if de_pos is not None:
            _try_update(de_pos, de_radii, de_score)
            # Polish DE result
            if time.time() < deadline - 15:
                tc_pos, tc_radii, tc_score = _trust_constr_solve(de_pos, de_radii, deadline)
                if tc_pos is not None:
                    _try_update(tc_pos, tc_radii, tc_score)
    
    # Fallback
    if best_pos is None or best_score < 0:
        pos, radii = _initial_grid()
        pos_pp, radii_pp = _post_process(pos, radii)
        best_pos = pos_pp
        best_radii = radii_pp
        best_score = np.sum(radii_pp)
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        # Trust-constr on best
        tc_pos, tc_radii, tc_score = _trust_constr_solve(best_pos, best_radii, deadline)
        if tc_pos is not None:
            _try_update(tc_pos, tc_radii, tc_score)
        
        # SLSQP final
        if time.time() < deadline - 3:
            sl_pos, sl_radii, sl_score = _slsqp_polish(best_pos, best_radii, deadline)
            if sl_pos is not None:
                _try_update(sl_pos, sl_radii, sl_score)
    
    # Final post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
