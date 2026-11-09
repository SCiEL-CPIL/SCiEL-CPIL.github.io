# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 620.0  # seconds, safety margin under 630s kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _base_inside(u, v):
    """Boolean: is (u,v) in the base L-shaped corridor?
    Base = {0<=u<=1, -3<=v<=1} U {-3<=u<=1, 0<=v<=1}."""
    horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    return horiz | vert


def _build_sofa_grid():
    """Return (xs, ys, X, Y) for the sofa grid (450 x 300)."""
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return xs, ys, X, Y


def _densify_path(path):
    """Add evaluator endpoints and densify. Returns dense (m,3) array."""
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


def _eval_path(path, X, Y):
    """path: (n,3) array of (x,y,a_deg). Returns number of sofa grid points
    inside EVERY placement (including evaluator-added endpoints).
    Uses active-set tracking for speed."""
    dense = _densify_path(path)
    
    # Flatten X, Y for easier index management
    Xf = X.ravel()
    Yf = Y.ravel()
    npts = Xf.size
    
    # Active set: indices of points still inside all placements so far
    active = np.arange(npts)
    
    for (cx, cy, a_deg) in dense:
        if len(active) == 0:
            break
        a = np.deg2rad(a_deg)
        c, s = np.cos(a), np.sin(a)
        dx = Xf[active] - cx
        dy = Yf[active] - cy
        u = c * dx + s * dy
        v = -s * dx + c * dy
        mask = _base_inside(u, v)
        active = active[mask]
    
    return int(len(active))


def _score(path, X, Y):
    return _eval_path(path, X, Y) * 0.0001


def _initial_path():
    return np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)


def _nelder_mead(f, x0, max_iter=3000, step=0.05, tol=1e-6):
    """Simple Nelder-Mead. x0 is a flat array; f takes a flat array."""
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        pt = x0.copy()
        pt[i] += step
        simplex[i + 1] = pt
    vals = np.array([f(simplex[i]) for i in range(n + 1)])
    best_i = int(np.argmin(vals))
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 30:
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


def _optimize_path(X, Y, angles, xy0, max_iter=2000, step=0.05):
    """Optimize (x,y) for given fixed angles."""
    n_ang = len(angles)
    
    def f(flat):
        path = np.empty((n_ang, 3))
        path[:, 0] = flat[0::2]
        path[:, 1] = flat[1::2]
        path[:, 2] = angles
        return -_score(path, X, Y)
    
    xopt, fopt = _nelder_mead(f, xy0, max_iter=max_iter, step=step)
    opt_path = np.empty((n_ang, 3))
    opt_path[:, 0] = xopt[0::2]
    opt_path[:, 1] = xopt[1::2]
    opt_path[:, 2] = angles
    return opt_path


def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path (n,3) array of corridor inner-corner placements (x,y,a_deg).
    """
    xs, ys, X, Y = _build_sofa_grid()
    best_path = _initial_path()
    best_score = _score(best_path, X, Y)
    
    # Try multiple angle configurations with different numbers of points
    angle_configs = [
        np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
        np.array([12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0]),
        np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]),
        np.array([8.0, 18.0, 28.0, 38.0, 48.0, 58.0, 68.0, 78.0, 88.0]),
        np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]),
        np.array([7.0, 17.0, 27.0, 37.0, 47.0, 57.0, 67.0, 77.0, 87.0]),
        np.array([10.0, 22.0, 34.0, 46.0, 58.0, 70.0, 82.0]),
        np.array([13.0, 26.0, 39.0, 52.0, 65.0, 78.0]),
    ]
    
    for angles in angle_configs:
        if _elapsed() > _DEADLINE - 60:
            break
        
        n_ang = len(angles)
        xy0 = np.zeros(2 * n_ang)
        for k, a in enumerate(angles):
            idx = int(np.argmin(np.abs(best_path[:, 2] - a)))
            xy0[2 * k] = best_path[idx, 0]
            xy0[2 * k + 1] = best_path[idx, 1]
        
        # Multi-scale: coarse then fine
        opt_path = _optimize_path(X, Y, angles, xy0, max_iter=800, step=0.06)
        opt_score = _score(opt_path, X, Y)
        
        # Fine refinement
        if opt_score > best_score and _elapsed() < _DEADLINE - 40:
            xy1 = np.zeros(2 * n_ang)
            for k in range(n_ang):
                xy1[2 * k] = opt_path[k, 0]
                xy1[2 * k + 1] = opt_path[k, 1]
            opt_path2 = _optimize_path(X, Y, angles, xy1, max_iter=800, step=0.02)
            opt_score2 = _score(opt_path2, X, Y)
            if opt_score2 > best_score:
                best_path = opt_path2
                best_score = opt_score2
        elif opt_score > best_score:
            best_path = opt_path
            best_score = opt_score
    
    # Final local refinement with small perturbations
    if _elapsed() < _DEADLINE - 30:
        for _ in range(5):
            if _elapsed() > _DEADLINE - 20:
                break
            n_ang = len(best_path)
            angles = best_path[:, 2].copy()
            xy0 = np.zeros(2 * n_ang)
            for k in range(n_ang):
                xy0[2 * k] = best_path[k, 0]
                xy0[2 * k + 1] = best_path[k, 1]
            
            # Add small random perturbation
            xy0 += np.random.normal(0, 0.02, size=2 * n_ang)
            
            def f(flat):
                path = np.empty((n_ang, 3))
                path[:, 0] = flat[0::2]
                path[:, 1] = flat[1::2]
                path[:, 2] = angles
                return -_score(path, X, Y)
            
            xopt, fopt = _nelder_mead(f, xy0, max_iter=400, step=0.02)
            opt_path = np.empty((n_ang, 3))
            opt_path[:, 0] = xopt[0::2]
            opt_path[:, 1] = xopt[1::2]
            opt_path[:, 2] = angles
            opt_score = _score(opt_path, X, Y)
            if opt_score > best_score:
                best_path = opt_path
                best_score = opt_score
    
    # Final sanity: ensure finite, shape (n,3), n>=1.
    best_path = np.asarray(best_path, dtype=np.float64)
    if best_path.ndim != 2 or best_path.shape[1] != 3 or best_path.shape[0] < 1:
        best_path = _initial_path()
    if not np.all(np.isfinite(best_path)):
        best_path = _initial_path()
    return best_path


# EVOLVE-BLOCK-END
