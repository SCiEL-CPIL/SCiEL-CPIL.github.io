# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 620.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _base_inside(u, v):
    """Vectorized base corridor check."""
    horiz = (u >= -1e-9) & (u <= 1.0 + 1e-9) & (v >= -3.0 - 1e-9) & (v <= 1.0 + 1e-9)
    vert = (u >= -3.0 - 1e-9) & (u <= 1.0 + 1e-9) & (v >= -1e-9) & (v <= 1.0 + 1e-9)
    return horiz | vert

def _build_grid(subsample=1):
    """Build sofa grid. subsample=1 for full, >1 for subsampled."""
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    if subsample > 1:
        xs = xs[::subsample]
        ys = ys[::subsample]
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return X, Y

def _densify_path(p):
    """Densify path according to evaluator rules."""
    dense = [p[0]]
    for i in range(len(p) - 1):
        a1 = p[i, 2]
        a2 = p[i + 1, 2]
        m = max(1, int(np.ceil(abs(a2 - a1) - 1e-9)))
        if m > 1:
            ts = np.linspace(0.0, 1.0, m + 1)
            for t in ts[1:-1]:
                row = p[i] * (1.0 - t) + p[i + 1] * t
                dense.append(row)
        dense.append(p[i + 1])
    return np.array(dense)

def _eval_path_score_fast(path, X, Y):
    """Faster evaluation using only surviving points."""
    p = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
    dense = _densify_path(p)
    
    if len(dense) == 0:
        return 0.0
    
    # Track surviving indices
    alive = np.ones(X.shape, dtype=bool)
    
    for (cx, cy, a_deg) in dense:
        if not alive.any():
            break
        a = np.deg2rad(a_deg)
        c, s = np.cos(a), np.sin(a)
        dx = X - cx
        dy = Y - cy
        u = c * dx + s * dy
        v = -s * dx + c * dy
        mask = _base_inside(u, v)
        alive &= mask
        
    return np.sum(alive) * 0.0001

def _generate_arc_path(h, n_pts=10):
    """
    Generate a path based on a circular arc.
    Center is at (-1, h). Radius R = sqrt(1 + h^2).
    Start (0,0), End (-2,0).
    We interpolate the angle from start to end.
    """
    cx = -1.0
    cy = h
    R = np.sqrt(1.0 + h*h)
    
    # Angle of start point (0,0) relative to center (-1, h)
    # Vector: (1, -h)
    theta_start = np.arctan2(-h, 1.0)
    
    # Angle of end point (-2,0) relative to center (-1, h)
    # Vector: (-1, -h)
    theta_end = np.arctan2(-h, -1.0)
    
    # Ensure we take the correct arc (the one that doesn't wrap around the long way)
    # If h > 0, the arc should bulge upwards (y > 0).
    # theta_start is in (-pi/2, 0). theta_end is in (-pi, -pi/2).
    # Difference is negative. We want to go from start to end.
    # If h > 0, the center is above. The arc connecting (0,0) and (-2,0) passing through y>0
    # corresponds to angles going from theta_start to theta_end?
    # Let's check a point in the middle.
    # If h=1, R=sqrt(2). Center (-1,1).
    # Start (0,0): vec (1,-1), angle -45 deg.
    # End (-2,0): vec (-1,-1), angle -135 deg.
    # Midpoint of arc should be at angle -90 deg.
    # Point: (-1 + R*cos(-90), 1 + R*sin(-90)) = (-1, 1-sqrt(2)) = (-1, -0.414).
    # This is below the x-axis. But we want the sofa to move "up" into the corridor?
    # The corridor is fixed. The sofa moves.
    # The inner corner of the corridor moves.
    # If the corner moves down, the corridor covers more of the top area?
    # Let's just optimize h over a range, both positive and negative.
    
    # Interpolate theta
    # We want the "short" arc.
    # If theta_end < theta_start, we can just linearly interpolate.
    # But we must be careful with the branch cut.
    # Let's just linearly interpolate the parameter t from 0 to 1 and map to angle.
    # To ensure we don't jump 360 degrees, we can adjust theta_end.
    diff = theta_end - theta_start
    if diff > np.pi:
        theta_end -= 2 * np.pi
    elif diff < -np.pi:
        theta_end += 2 * np.pi
        
    thetas = np.linspace(theta_start, theta_end, n_pts)
    
    x = cx + R * np.cos(thetas)
    y = cy + R * np.sin(thetas)
    a = np.linspace(0.0, 90.0, n_pts)
    
    return np.column_stack([x, y, a])

def _initial_paths():
    """Generate multiple initial path candidates."""
    paths = []
    
    # Arc-based paths with different h values
    for h in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
        paths.append(_generate_arc_path(h, n_pts=8))
        
    # Manual paths
    paths.append(np.array([
        [-0.4, 0.4, 20.0],
        [-0.9, 0.5, 45.0],
        [-1.4, 0.3, 65.0],
        [-1.8, 0.1, 80.0],
    ], dtype=np.float64))
    
    paths.append(np.array([
        [-0.3, 0.3, 15.0],
        [-0.7, 0.4, 30.0],
        [-1.1, 0.4, 50.0],
        [-1.5, 0.2, 70.0],
        [-1.8, 0.0, 85.0],
    ], dtype=np.float64))
    
    return paths

def _nelder_mead(f, x0, max_iter=2000, step=0.1, tol=1e-5, deadline=_DEADLINE - 30):
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        pt = x0.copy()
        pt[i] += step
        simplex[i + 1] = pt
    vals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    for it in range(max_iter):
        if _elapsed() > deadline:
            break
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]
        if abs(vals[0] - vals[-1]) < tol:
            break
            
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                vals[-1] = fe
            else:
                simplex[-1] = xr
                vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr
            vals[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1] = xc
                vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])
                    
    return simplex[int(np.argmin(vals))], np.min(vals)

def _refine_path(path, X, Y, max_iter=300, step=0.03, deadline=_DEADLINE - 30):
    """Refine a path using Nelder-Mead."""
    x0 = path.flatten()
    
    def f(flat):
        p = flat.reshape(-1, 3)
        return -_eval_path_score_fast(p, X, Y)
    
    xopt, _ = _nelder_mead(f, x0, max_iter=max_iter, step=step, deadline=deadline)
    return xopt.reshape(-1, 3)

def _polish_path(path, X, Y, n_iter=50, step_init=0.005, deadline=_DEADLINE - 5):
    """
    High-precision polish pass using smaller steps and tighter tolerance.
    """
    best = path.copy()
    best_score = _eval_path_score_fast(best, X, Y)
    
    step = step_init
    for it in range(n_iter):
        if _elapsed() > deadline:
            break
        if step < 1e-6:
            break
            
        improved = False
        for i in range(len(best)):
            for j in range(3):
                for delta in [step, -step]:
                    candidate = best.copy()
                    candidate[i, j] += delta
                    candidate[:, 2] = np.clip(candidate[:, 2], -5, 95)
                    
                    score = _eval_path_score_fast(candidate, X, Y)
                    if score > best_score + 1e-10:
                        best = candidate
                        best_score = score
                        improved = True
        
        if not improved:
            step *= 0.5
            
    return best, best_score

def moving_sofa_2d() -> np.ndarray:
    # Use subsampled grid for optimization
    X_opt, Y_opt = _build_grid(subsample=4)
    X_full, Y_full = _build_grid(subsample=1)
    
    best_path = None
    best_score = -1.0
    
    # Phase 1: Global search on arc parameters
    # We optimize h (center y-coordinate)
    def f_h(h_flat):
        h = h_flat[0]
        path = _generate_arc_path(h, n_pts=8)
        return -_eval_path_score_fast(path, X_opt, Y_opt)
        
    # Try a range of h values with coarse Nelder-Mead
    # Start with a few initial guesses
    for h_init in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        if _elapsed() > _DEADLINE - 100:
            break
        xopt, _ = _nelder_mead(f_h, np.array([h_init]), max_iter=200, step=0.5, deadline=_DEADLINE - 80)
        h_opt = xopt[0]
        path = _generate_arc_path(h_opt, n_pts=8)
        score = _eval_path_score_fast(path, X_opt, Y_opt)
        if score > best_score:
            best_score = score
            best_path = path

    # Phase 2: Local refinement of the best arc path and manual paths
    init_paths = _initial_paths()
    if best_path is not None:
        init_paths.append(best_path)
        
    for path in init_paths:
        if _elapsed() > _DEADLINE - 60:
            break
        refined = _refine_path(path, X_opt, Y_opt, max_iter=400, step=0.05, deadline=_DEADLINE - 40)
        score = _eval_path_score_fast(refined, X_opt, Y_opt)
        if score > best_score:
            best_score = score
            best_path = refined

    # Phase 3: Final refinement on full grid
    if best_path is not None and _elapsed() < _DEADLINE - 20:
        refined_full = _refine_path(best_path, X_full, Y_full, max_iter=100, step=0.015, deadline=_DEADLINE - 10)
        score_full = _eval_path_score_fast(refined_full, X_full, Y_full)
        score_best_full = _eval_path_score_fast(best_path, X_full, Y_full)
        if score_full > score_best_full:
            best_path = refined_full

    # Phase 4: Polish
    if best_path is not None and _elapsed() < _DEADLINE - 5:
        polished, polish_score = _polish_path(best_path, X_full, Y_full, n_iter=20, step_init=0.002, deadline=_DEADLINE - 2)
        if polish_score > _eval_path_score_fast(best_path, X_full, Y_full):
            best_path = polished

    # Ensure valid output
    if best_path is None:
        best_path = init_paths[0]
        
    best_path = np.asarray(best_path, dtype=np.float64)
    if best_path.ndim != 2 or best_path.shape[1] != 3 or best_path.shape[0] < 1:
        best_path = init_paths[0]
    if not np.all(np.isfinite(best_path)):
        best_path = init_paths[0]
        
    return best_path
# EVOLVE-BLOCK-END
