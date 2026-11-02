"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

DEADLINE = 120.0
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return DEADLINE - _elapsed() - 5.0  # 5s safety margin


def _max_radius_for_circle(x, y, centers, radii, idx):
    """Max radius for circle at (x,y) given others."""
    r_max = min(x, 1.0 - x, y, 1.0 - y)
    for j in range(len(centers)):
        if j == idx:
            continue
        d = np.sqrt((x - centers[j, 0])**2 + (y - centers[j, 1])**2)
        r_max = min(r_max, d - radii[j])
    return max(0.0, r_max)


def _hex_initial(n_circles, jitter=0.0):
    """Generate a hexagonal packing initialization."""
    # Hexagonal lattice: rows offset by half spacing
    # For 26 circles, use roughly 6x5 arrangement
    rows = 5
    cols = 6
    dx = 1.0 / (cols + 0.5)
    dy = 1.0 / (rows + 0.5)
    # Hex offset: odd rows shifted
    hex_dy = dy * np.sqrt(3) / 2
    centers = []
    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5) * dx + (0.5 * dx if r % 2 == 1 else 0)
            y = (r + 0.5) * dy
            if jitter > 0:
                x += np.random.uniform(-jitter, jitter)
                y += np.random.uniform(-jitter, jitter)
            centers.append((x, y))
    centers = centers[:n_circles]
    return np.array(centers)


def _grid_initial(n_circles, jitter=0.0):
    """Grid initialization with equal spacing."""
    side = int(np.ceil(np.sqrt(n_circles)))
    dx = 1.0 / side
    dy = 1.0 / side
    centers = []
    for j in range(side):
        for i in range(side):
            if len(centers) >= n_circles:
                break
            x = (i + 0.5) * dx
            y = (j + 0.5) * dy
            if jitter > 0:
                x += np.random.uniform(-jitter, jitter)
                y += np.random.uniform(-jitter, jitter)
            centers.append((x, y))
    return np.array(centers[:n_circles])


def _random_initial(n_circles, seed=None):
    """Random initialization with minimum spacing."""
    rng = np.random.RandomState(seed)
    centers = []
    min_dist = 0.1
    attempts = 0
    while len(centers) < n_circles and attempts < 10000:
        attempts += 1
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        ok = True
        for cx, cy in centers:
            if (x - cx)**2 + (y - cy)**2 < min_dist**2:
                ok = False
                break
        if ok:
            centers.append((x, y))
    # Fill remaining randomly
    while len(centers) < n_circles:
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        centers.append((x, y))
    return np.array(centers)


def _get_neighbors(centers, idx, k=8):
    """Get k nearest neighbors using simple distance sort (avoid Delaunay overhead for small n)."""
    if len(centers) <= 2:
        return list(range(len(centers)))
    dists = np.array([np.sqrt((centers[idx, 0] - centers[j, 0])**2 + 
                               (centers[idx, 1] - centers[j, 1])**2) 
                      for j in range(len(centers))])
    dists[idx] = np.inf
    neighbors = np.argsort(dists)[:k]
    return neighbors.tolist()


def _constrained_objective(params, n):
    """Objective: negative sum of radii (to minimize)."""
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    return -np.sum(radii)


def _constrained_grad(params, n):
    """Gradient of negative sum of radii."""
    grad = np.zeros_like(params)
    grad[2*n:] = -1.0
    return grad


def _constrained_constraints(params, n):
    """Constraints: non-overlap and boundary."""
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    constraints = []
    
    # Boundary constraints: r <= min(x, 1-x, y, 1-y)
    for i in range(n):
        # x - r >= 0
        constraints.append({'type': 'ineq', 
                           'fun': lambda p, i=i: p[2*i] - p[2*n+i],
                           'jac': lambda p, i=i: np.concatenate([np.eye(1, 2*n)[i].flatten()[:2*n] * 1 - 0, np.zeros(2*n) + np.eye(1, n)[i].flatten()[:n] * (-1)]) if False else _boundary_jac_x(p, i, n)})
        # 1 - x - r >= 0
        constraints.append({'type': 'ineq',
                           'fun': lambda p, i=i: 1.0 - p[2*i] - p[2*n+i],
                           'jac': lambda p, i=i: _boundary_jac_1mx(p, i, n)})
        # y - r >= 0
        constraints.append({'type': 'ineq',
                           'fun': lambda p, i=i: p[2*i+1] - p[2*n+i],
                           'jac': lambda p, i=i: _boundary_jac_y(p, i, n)})
        # 1 - y - r >= 0
        constraints.append({'type': 'ineq',
                           'fun': lambda p, i=i: 1.0 - p[2*i+1] - p[2*n+i],
                           'jac': lambda p, i=i: _boundary_jac_1my(p, i, n)})
    
    # Non-overlap: for each pair, dist >= r_i + r_j
    # Use only nearest neighbors to reduce constraint count
    for i in range(n):
        neighbors = _get_neighbors(centers, i, k=10)
        for j in neighbors:
            if j <= i:
                continue
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < 1e-10:
                d = 1e-10
            # constraint: d - r_i - r_j >= 0
            constraints.append({'type': 'ineq',
                               'fun': lambda p, i=i, j=j: _overlap_fun(p, i, j, n),
                               'jac': lambda p, i=i, j=j: _overlap_jac(p, i, j, n)})
    
    # Radius positivity
    for i in range(n):
        constraints.append({'type': 'ineq',
                           'fun': lambda p, i=i: p[2*n+i],
                           'jac': lambda p, i=i: _radius_pos_jac(p, i, n)})
    
    return constraints


def _boundary_jac_x(p, i, n):
    g = np.zeros(3*n)
    g[2*i] = 1.0
    g[2*n+i] = -1.0
    return g

def _boundary_jac_1mx(p, i, n):
    g = np.zeros(3*n)
    g[2*i] = -1.0
    g[2*n+i] = -1.0
    return g

def _boundary_jac_y(p, i, n):
    g = np.zeros(3*n)
    g[2*i+1] = 1.0
    g[2*n+i] = -1.0
    return g

def _boundary_jac_1my(p, i, n):
    g = np.zeros(3*n)
    g[2*i+1] = -1.0
    g[2*n+i] = -1.0
    return g

def _overlap_fun(p, i, j, n):
    dx = p[2*i] - p[2*j]
    dy = p[2*i+1] - p[2*j+1]
    d = np.sqrt(dx*dx + dy*dy)
    if d < 1e-10:
        d = 1e-10
    return d - p[2*n+i] - p[2*n+j]

def _overlap_jac(p, i, j, n):
    g = np.zeros(3*n)
    dx = p[2*i] - p[2*j]
    dy = p[2*i+1] - p[2*j+1]
    d = np.sqrt(dx*dx + dy*dy)
    if d < 1e-10:
        d = 1e-10
    g[2*i] = dx/d
    g[2*i+1] = dy/d
    g[2*j] = -dx/d
    g[2*j+1] = -dy/d
    g[2*n+i] = -1.0
    g[2*n+j] = -1.0
    return g

def _radius_pos_jac(p, i, n):
    g = np.zeros(3*n)
    g[2*n+i] = 1.0
    return g


def _is_valid(centers, radii, tol=1e-6):
    """Check if packing is valid."""
    n = len(radii)
    for i in range(n):
        if radii[i] < -tol:
            return False
        if centers[i, 0] - radii[i] < -tol:
            return False
        if 1.0 - centers[i, 0] - radii[i] < -tol:
            return False
        if centers[i, 1] - radii[i] < -tol:
            return False
        if 1.0 - centers[i, 1] - radii[i] < -tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            if d < radii[i] + radii[j] - tol:
                return False
    return True


def _fix_packing(centers, radii):
    """Fix small violations by shrinking radii and clipping positions."""
    centers = centers.copy()
    radii = radii.copy()
    # Clip centers to valid range
    for i in range(len(radii)):
        centers[i, 0] = np.clip(centers[i, 0], 0.0, 1.0)
        centers[i, 1] = np.clip(centers[i, 1], 0.0, 1.0)
    # Ensure boundary validity
    for i in range(len(radii)):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], max_r)
        radii[i] = max(0.0, radii[i])
    # Iteratively fix overlaps
    for _ in range(50):
        changed = False
        for i in range(len(radii)):
            for j in range(i+1, len(radii)):
                d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                needed = radii[i] + radii[j]
                if d < needed:
                    # Shrink the smaller radius
                    if radii[i] < radii[j]:
                        radii[i] = max(0.0, (d - radii[j]) * 0.999)
                    else:
                        radii[j] = max(0.0, (d - radii[i]) * 0.999)
                    changed = True
        if not changed:
            break
    return centers, radii


def _optimize_from_start(centers_init, radii_init, time_budget):
    """Run SLSQP from a given start."""
    n = len(radii_init)
    params = np.concatenate([centers_init.flatten(), radii_init])
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
    for i in range(n):
        bounds.append((0.0, 0.5))  # radius
    
    try:
        result = minimize(
            _constrained_objective,
            params,
            args=(n,),
            method='SLSQP',
            jac=_constrained_grad,
            constraints=_constrained_constraints,
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
        )
        if result.success:
            centers = result.x[:2*n].reshape(n, 2)
            radii = result.x[2*n:]
            centers, radii = _fix_packing(centers, radii)
            return centers, radii, -result.fun
        else:
            # Try to salvage
            centers = params[:2*n].reshape(n, 2)
            radii = params[2*n:]
            centers, radii = _fix_packing(centers, radii)
            return centers, radii, np.sum(radii)
    except Exception:
        centers, radii = _fix_packing(centers_init, radii_init)
        return centers, radii, np.sum(radii)


def construct():
    n = N
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    # Generate initializations
    inits = []
    
    # Hexagonal
    for jitter in [0.0, 0.01, 0.02]:
        c = _hex_initial(n, jitter=jitter)
        r = np.array([min(c[i,0], 1-c[i,0], c[i,1], 1-c[i,1]) * 0.9 for i in range(n)])
        inits.append((c, r))
    
    # Grid
    for jitter in [0.0, 0.01]:
        c = _grid_initial(n, jitter=jitter)
        r = np.array([min(c[i,0], 1-c[i,0], c[i,1], 1-c[i,1]) * 0.9 for i in range(n)])
        inits.append((c, r))
    
    # Random
    for seed in [42, 123, 7]:
        c = _random_initial(n, seed=seed)
        r = np.array([min(c[i,0], 1-c[i,0], c[i,1], 1-c[i,1]) * 0.8 for i in range(n)])
        inits.append((c, r))
    
    # Optimize from each start
    for idx, (c_init, r_init) in enumerate(inits):
        if _remaining() < 10.0:
            break
        
        time_budget = min(_remaining() - 5.0, 15.0)
        centers, radii, total = _optimize_from_start(c_init, r_init, time_budget)
        
        if _is_valid(centers, radii, tol=1e-5):
            if total > best_sum:
                best_sum = total
                best_centers = centers
                best_radii = radii
        else:
            # Try to fix and check again
            centers_fixed, radii_fixed = _fix_packing(centers, radii)
            if _is_valid(centers_fixed, radii_fixed, tol=1e-5):
                total_fixed = np.sum(radii_fixed)
                if total_fixed > best_sum:
                    best_sum = total_fixed
                    best_centers = centers_fixed
                    best_radii = radii_fixed
    
    # If no valid result found, fall back to simple grid
    if best_centers is None:
        side = int(np.ceil(np.sqrt(n)))
        dx = 1.0 / side
        dy = 1.0 / side
        centers = []
        radii = []
        for j in range(side):
            for i in range(side):
                if len(centers) >= n:
                    break
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy
                r = 0.5 * dx - 1e-3
                centers.append((x, y))
                radii.append(r)
        best_centers = np.array(centers[:n])
        best_radii = np.array(radii[:n])
    
    result = np.column_stack([best_centers, best_radii])
    return result

# EVOLVE-BLOCK-END
