"""Circle packing in unit square n=32 — nearest-neighbor-limited radius + gradient search."""
import time
import numpy as np
from scipy.optimize import minimize

N = 32
DEADLINE = 140.0  # safety margin under 150s
START_TIME = None


def _radii_from_centers(c):
    """Given centers (n,2), compute radii as half the min clearance to boundary or other circles."""
    n = c.shape[0]
    # Distance to each boundary
    d_left = c[:, 0]
    d_right = 1.0 - c[:, 0]
    d_bottom = c[:, 1]
    d_top = 1.0 - c[:, 1]
    # Min boundary distance per circle
    r_boundary = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    # Min pairwise distance per circle (half of that is the clearance)
    r_pair = 0.5 * np.min(dist, axis=1)
    # Radius is min of boundary clearance and pair clearance
    r = np.minimum(r_boundary, r_pair)
    return r


def _objective_and_grad(c):
    """Objective = -sum(radii). Gradient computed analytically."""
    n = c.shape[0]
    r = _radii_from_centers(c)
    obj = -np.sum(r)
    
    # Gradient: for each circle i, dr_i/dc_i depends on which constraint is active
    # We compute subgradient: if boundary is limiting, grad is unit vector toward nearest boundary
    # if pair is limiting, grad is unit vector toward nearest neighbor
    grad = np.zeros_like(c)
    
    d_left = c[:, 0]
    d_right = 1.0 - c[:, 0]
    d_bottom = c[:, 1]
    d_top = 1.0 - c[:, 1]
    r_boundary = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist, axis=1)
    r_pair = 0.5 * min_pair_dist
    
    for i in range(n):
        if r[i] == r_boundary[i] and r_boundary[i] <= r_pair[i]:
            # Boundary-limited: find nearest boundary
            min_d = r_boundary[i]
            # Move away from nearest boundary => gradient of r w.r.t. c_i points toward center
            # r = min(d_left, d_right, d_bottom, d_top)
            # dr/dx = -1 if d_left is min, +1 if d_right is min, etc.
            # We want to maximize r, so gradient of -r is negative of that
            if d_left[i] == min_d:
                grad[i, 0] = 1.0   # moving right increases d_left? No: d_left = x, so dr/dx=1, d(-r)/dx=-1
                # Actually: r = x (if left is min), dr/dx = 1, so d(-r)/dx = -1
                # But we want to move to increase r, so the descent direction for -r is -grad
                # Let's just compute d(-r)/dc
                grad[i, 0] = -1.0
            elif d_right[i] == min_d:
                grad[i, 0] = 1.0   # r = 1-x, dr/dx = -1, d(-r)/dx = 1
            elif d_bottom[i] == min_d:
                grad[i, 1] = -1.0  # r = y, dr/dy = 1, d(-r)/dy = -1
            else:
                grad[i, 1] = 1.0   # r = 1-y, dr/dy = -1, d(-r)/dy = 1
        else:
            # Pair-limited: find nearest neighbor
            j = np.argmin(dist[i])
            if dist[i, j] > 1e-12:
                # r_i = 0.5 * ||c_i - c_j||, dr_i/dc_i = 0.5 * (c_i - c_j)/||c_i - c_j||
                # d(-r_i)/dc_i = -0.5 * (c_i - c_j)/||c_i - c_j||
                direction = (c[i] - c[j]) / dist[i, j]
                grad[i] += -0.5 * direction
                # Also affects r_j: dr_j/dc_j = 0.5 * (c_j - c_i)/||c_j - c_i||
                # d(-r_j)/dc_j = -0.5 * (c_j - c_i)/||c_j - c_i|| = 0.5 * (c_i - c_j)/||c_i - c_j||
                grad[j] += 0.5 * direction
    return obj, grad


def _grid_init(n=6):
    """Grid of n*n points, take first N."""
    pts = []
    for i in range(n):
        for j in range(n):
            pts.append(((i + 0.5) / n, (j + 0.5) / n))
    pts = np.array(pts[:N])
    return pts


def _hex_init():
    """Hexagonal packing pattern."""
    pts = []
    rows = 7
    cols = 7
    dx = 1.0 / (cols - 1 + 0.5)
    dy = dx * np.sqrt(3) / 2
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) * dx + (0.5 * dx if i % 2 else 0)
            y = (i + 0.5) * dy
            if x < 1.0 and y < 1.0:
                pts.append((x, y))
    pts = np.array(pts[:N])
    # Scale to fit in [0,1]^2
    if len(pts) > 0:
        mx = pts.max(axis=0)
        mn = pts.min(axis=0)
        scale = 0.9 / np.max(mx - mn)
        pts = (pts - mn) * scale + 0.05
    return pts


def _random_init(seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(0.1, 0.9, size=(N, 2))


def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    best_c = None
    best_val = -np.inf
    
    inits = []
    inits.append(('grid', _grid_init(6)))
    inits.append(('hex', _hex_init()))
    for s in [42, 123, 777, 2024, 31415]:
        inits.append((f'rand{s}', _random_init(s)))
    
    for name, c0 in inits:
        if time.time() - START_TIME > DEADLINE - 10:
            break
        try:
            res = minimize(
                _objective_and_grad,
                c0,
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            c_opt = res.x
            # Clip to [0,1]
            c_opt = np.clip(c_opt, 0, 1)
            r = _radii_from_centers(c_opt)
            val = np.sum(r)
            if val > best_val:
                best_val = val
                best_c = c_opt.copy()
        except Exception:
            continue
    
    # If no valid result, fall back to grid
    if best_c is None:
        best_c = _grid_init(6)
    
    r = _radii_from_centers(best_c)
    # Build output
    out = np.zeros((N, 3))
    out[:, :2] = best_c
    out[:, 2] = r
    return out
