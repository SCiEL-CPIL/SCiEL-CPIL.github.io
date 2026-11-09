# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 610.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _base_inside(u, v):
    """Boolean: is (u,v) in the base L-shaped corridor?"""
    horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    return horiz | vert


def _build_sofa_grid():
    """Return (X, Y) for the sofa grid (450 x 300)."""
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return X, Y


def _densify_path(path):
    """Densify path per evaluator rules: add linear interpolations between consecutive rows."""
    # path is (n, 3)
    # Evaluator adds (0,0,0) before and (-2,0,90) after
    p = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
    
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


def _eval_path_vectorized(path, X, Y):
    """Vectorized evaluation: returns number of sofa grid points inside EVERY placement."""
    dense = _densify_path(path)
    n_placements = len(dense)
    
    if n_placements == 0:
        return 0

    # Precompute grid points
    Xf = X.ravel()
    Yf = Y.ravel()
    n_pts = Xf.size

    # We check if a point is inside ALL placements.
    # inside = np.ones(n_pts, dtype=bool)
    # This is memory heavy if we do it all at once.
    # Let's chunk the placements.
    
    inside = np.ones(n_pts, dtype=bool)
    
    chunk_size = 100  # Adjust based on memory. 135k points * 100 placements * 8 bytes * 4 vars ~ 432MB. Too much.
    # Let's use smaller chunks.
    # 135,000 points.
    # dx, dy, u, v are (chunk, 135000).
    # 10 * 135000 * 4 * 8 = 43 MB. Safe.
    chunk_size = 20

    for start in range(0, n_placements, chunk_size):
        if not np.any(inside):
            break
        end = min(start + chunk_size, n_placements)
        chunk = dense[start:end]
        n_chunk = end - start
        
        cx = chunk[:, 0]  # (n_chunk,)
        cy = chunk[:, 1]
        a_rad = np.deg2rad(chunk[:, 2])
        c = np.cos(a_rad)  # (n_chunk,)
        s = np.sin(a_rad)  # (n_chunk,)
        
        # dx = Xf - cx, dy = Yf - cy
        # We need to broadcast cx (n_chunk, 1) against Xf (1, n_pts)
        dx = Xf[None, :] - cx[:, None]  # (n_chunk, n_pts)
        dy = Yf[None, :] - cy[:, None]  # (n_chunk, n_pts)
        
        u = c[:, None] * dx + s[:, None] * dy
        v = -s[:, None] * dx + c[:, None] * dy
        
        horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        mask = horiz | vert  # (n_chunk, n_pts)
        
        # A point is inside if it's inside ALL placements in this chunk
        chunk_inside = mask.all(axis=0)  # (n_pts,)
        inside &= chunk_inside
        
    return int(inside.sum())


def _score(path, X, Y):
    return _eval_path_vectorized(path, X, Y) * 0.0001


def _generate_arc_path(x_c, y_c, R, n_points):
    """
    Generate a path of n_points along a circular arc.
    The arc connects (0,0) and (-2,0) approximately.
    We define the arc by the center (x_c, y_c) and radius R.
    We find the angles theta_start and theta_end corresponding to (0,0) and (-2,0).
    """
    # Start point (0,0)
    dx0 = 0.0 - x_c
    dy0 = 0.0 - y_c
    theta0 = np.arctan2(dy0, dx0)
    
    # End point (-2,0)
    dx1 = -2.0 - x_c
    dy1 = 0.0 - y_c
    theta1 = np.arctan2(dy1, dx1)
    
    # We want the arc from theta0 to theta1.
    # Determine the direction. We want the "shorter" arc or the one that keeps y positive?
    # Let's check the delta.
    delta = theta1 - theta0
    # Normalize delta to (-pi, pi]
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    
    # If delta is negative, we go clockwise.
    # We want to ensure the path stays in a reasonable region.
    # For the standard sofa problem, the path usually bulges into y > 0.
    # If center is (x_c, y_c) with y_c > 0, the arc bulges down (towards center) or up?
    # If we go from theta0 to theta1 with positive delta (CCW), the arc is on the "left" of the vector.
    
    # Let's just sample linearly in theta.
    # If the delta is large, we might wrap.
    # We assume the optimizer will find a good circle.
    
    thetas = np.linspace(theta0, theta0 + delta, n_points)
    
    xs = x_c + R * np.cos(thetas)
    ys = y_c + R * np.sin(thetas)
    
    # Calculate tangent angles for the corridor orientation 'a'
    # Tangent angle is theta + pi/2 (for CCW) or theta - pi/2 (for CW)
    # We want 'a' to go from 0 to 90.
    # Linear interpolation of a is safer for the evaluator's interpolation logic.
    # The evaluator interpolates (x,y,a) linearly.
    # If we provide points with a linearly increasing a, the evaluator's densification
    # will also have linearly increasing a.
    # So let's just set a = linspace(0, 90, n_points).
    # Wait, the evaluator adds (0,0,0) and (-2,0,90).
    # My path points are in between.
    # So a should be strictly between 0 and 90.
    a = np.linspace(0.0, 90.0, n_points + 2)[1:-1]
    
    path = np.column_stack([xs, ys, a])
    return path


def _nelder_mead(f, x0, max_iter=2000, step=0.1, tol=1e-6):
    """Nelder-Mead minimization."""
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        pt = x0.copy()
        pt[i] += step
        simplex[i + 1] = pt
    vals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 20:
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
    
    best_i = int(np.argmin(vals))
    return simplex[best_i], vals[best_i]


def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path (n,3) array of corridor inner-corner placements (x,y,a_deg).
    """
    X, Y = _build_sofa_grid()
    
    best_path = None
    best_score = -1.0
    
    # Strategy: Optimize circular arc parameters (x_c, y_c, R)
    # n_points is the number of intermediate points to return.
    # The evaluator adds 2 more.
    # Let's try a few n_points.
    
    n_points_options = [10, 20, 30]
    
    # Initial guesses for circle center and radius
    # The chord is length 2.
    # If it's a semi-circle, R=1, center=(-1, 0).
    # If it's a quarter circle, R=sqrt(2), center=(-1, 1) or (-1, -1).
    # Let's try center (-1, 1), R=sqrt(2) and center (-1, -1), R=sqrt(2).
    
    inits = [
        np.array([-1.0, 1.0, np.sqrt(2)]),
        np.array([-1.0, -1.0, np.sqrt(2)]),
        np.array([-1.0, 0.5, 1.5]),
        np.array([-0.5, 0.5, 1.0]),
        np.array([-1.5, 0.5, 1.0]),
    ]
    
    for n_pts in n_points_options:
        if _elapsed() > _DEADLINE - 100:
            break
            
        for init in inits:
            if _elapsed() > _DEADLINE - 50:
                break
                
            x_c0, y_c0, R0 = init
            
            def f(params):
                x_c, y_c, R = params
                # Ensure R is positive
                R = abs(R) + 1e-6
                path = _generate_arc_path(x_c, y_c, R, n_pts)
                # Check if path is valid (finite)
                if not np.all(np.isfinite(path)):
                    return 1e6
                s = _score(path, X, Y)
                return -s
            
            xopt, fopt = _nelder_mead(f, init, max_iter=500, step=0.2)
            x_c, y_c, R = xopt
            R = abs(R)
            
            opt_path = _generate_arc_path(x_c, y_c, R, n_pts)
            opt_score = _score(opt_path, X, Y)
            
            if opt_score > best_score:
                best_score = opt_score
                best_path = opt_path
    
    # If we found a good path, try to polish it by treating the points as free variables
    # This allows the path to deviate from the perfect circle if it improves the score.
    if best_path is not None and _elapsed() < _DEADLINE - 30:
        n = len(best_path)
        # Flatten x, y, a
        x0 = best_path[:, 0].copy()
        y0 = best_path[:, 1].copy()
        a0 = best_path[:, 2].copy()
        
        # We only optimize x and y, keeping a linear?
        # Or optimize all?
        # Optimizing all 3*n params might be slow but Nelder-Mead is local.
        # Let's stick to optimizing x and y for the fixed angles, as angles are already smooth.
        
        def f_xy(flat):
            x = flat[0:n]
            y = flat[n:2*n]
            path = np.column_stack([x, y, a0])
            if not np.all(np.isfinite(path)):
                return 1e6
            s = _score(path, X, Y)
            return -s
        
        init_xy = np.concatenate([x0, y0])
        xopt_xy, fopt_xy = _nelder_mead(f_xy, init_xy, max_iter=500, step=0.05)
        
        opt_x = xopt_xy[0:n]
        opt_y = xopt_xy[n:2*n]
        opt_path2 = np.column_stack([opt_x, opt_y, a0])
        opt_score2 = _score(opt_path2, X, Y)
        
        if opt_score2 > best_score:
            best_score = opt_score2
            best_path = opt_path2

    if best_path is None:
        # Fallback: simple linear path
        n = 10
        xs = np.linspace(0, -2, n)
        ys = np.zeros(n)
        a = np.linspace(0, 90, n)
        best_path = np.column_stack([xs, ys, a])
        # Shift a to be strictly between 0 and 90 for intermediate points
        a = np.linspace(0, 90, n + 2)[1:-1]
        best_path = np.column_stack([xs, ys, a])

    return best_path

# EVOLVE-BLOCK-END
