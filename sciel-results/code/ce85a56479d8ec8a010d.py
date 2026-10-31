"""Rewritten program: minimize squared max/min pairwise distance for 16 points in 2D."""
import time
import numpy as np
from scipy.optimize import minimize

N = 16
D = 2

def _pairwise_dists_sq(pts):
    """Compute squared pairwise distances for Nx2 array."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff * diff, axis=2)

def _objective(x, pts_shape):
    """Objective: squared ratio of max to min pairwise distance."""
    pts = x.reshape(pts_shape)
    d2 = _pairwise_dists_sq(pts)
    # Mask out diagonal
    n = pts.shape[0]
    mask = ~np.eye(n, dtype=bool)
    d2_masked = d2[mask]
    min_d2 = np.min(d2_masked)
    max_d2 = np.max(d2_masked)
    if min_d2 <= 1e-15:
        return 1e10
    return max_d2 / min_d2

def _objective_grad(x, pts_shape):
    """Gradient of the squared ratio via finite differences (expensive but robust)."""
    # Use analytic gradient would be complex; use numerical for reliability
    eps = 1e-7
    n_params = len(x)
    grad = np.zeros(n_params)
    f0 = _objective(x, pts_shape)
    for i in range(n_params):
        x_plus = x.copy()
        x_plus[i] += eps
        grad[i] = (_objective(x_plus, pts_shape) - f0) / eps
    return grad

def _hexagonal_init(n):
    """Generate hexagonal lattice points for n points."""
    # Hexagonal packing: points at (i + 0.5*(j%2), j*sqrt(3)/2)
    pts = []
    row = 0
    while len(pts) < n:
        for col in range(-row, row + 1):
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            pts.append([x, y])
            if len(pts) >= n:
                break
        row += 1
    return np.array(pts[:n], dtype=float)

def _circle_init(n):
    """Points on a circle plus center."""
    angles = np.linspace(0, 2 * np.pi, n - 1, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    pts = np.vstack([pts, [[0, 0]]])
    return pts

def _random_init(n, seed):
    rng = np.random.RandomState(seed)
    return rng.randn(n, 2)

def _optimize_from_init(init_pts, max_time):
    """Run optimization from a given initial configuration."""
    x0 = init_pts.flatten()
    pts_shape = (N, D)
    
    best_x = x0.copy()
    best_f = _objective(x0, pts_shape)
    
    start_time = time.time()
    
    # Try multiple optimization methods
    methods = ['Nelder-Mead', 'BFGS', 'Powell']
    
    for method in methods:
        if time.time() - start_time > max_time - 5:
            break
        try:
            if method == 'Nelder-Mead':
                res = minimize(_objective, x0, args=(pts_shape,), method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            elif method == 'BFGS':
                res = minimize(_objective, x0, args=(pts_shape,), method='BFGS',
                              options={'maxiter': 200, 'gtol': 1e-6})
            else:  # Powell
                res = minimize(_objective, x0, args=(pts_shape,), method='Powell',
                              options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-8})
            
            if res.success and res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x, best_f

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Multiple initializations
    inits = []
    
    # Hexagonal lattice
    try:
        inits.append(_hexagonal_init(N))
    except Exception:
        pass
    
    # Circle + center
    try:
        inits.append(_circle_init(N))
    except Exception:
        pass
    
    # Random inits with different seeds
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        try:
            inits.append(_random_init(N, seed))
        except Exception:
            pass
    
    best_x = None
    best_f = np.inf
    
    for init in inits:
        if time.time() > deadline - 5:
            break
        x_opt, f_opt = _optimize_from_init(init, deadline - time.time())
        if f_opt < best_f:
            best_f = f_opt
            best_x = x_opt
    
    if best_x is None:
        # Fallback: hexagonal
        best_x = _hexagonal_init(N).flatten()
    
    return best_x.reshape(N, D)
