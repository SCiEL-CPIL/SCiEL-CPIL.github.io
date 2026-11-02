import numpy as np
from scipy.optimize import minimize
import time

N = 50
HALF = 0.5
DEADLINE = 145.0  # Safety margin before 150s limit

def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r

def _hex_init(n):
    """Generate n points in a hexagonal lattice pattern."""
    # Hexagonal lattice basis
    a = 1.0
    b = np.sqrt(3) / 2.0
    
    points = []
    R = 8
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            x = i * a + j * 0.5 * a
            y = j * b
            points.append((x, y))
    
    points = np.array(points)
    dists = np.sum(points**2, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:n]]
    
    # Center and scale
    points -= np.mean(points, axis=0)
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        points *= 0.45 / max_coord
        
    return points

def _grid_init(n):
    """Generate n points in a grid pattern."""
    # Approximate square grid
    side = int(np.ceil(np.sqrt(n)))
    points = []
    for i in range(side):
        for j in range(side):
            if len(points) < n:
                points.append((i, j))
    
    points = np.array(points, dtype=float)
    points -= np.mean(points, axis=0)
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        points *= 0.45 / max_coord
        
    return points

def _soft_violation_obj(pts, r):
    """
    Objective: sum of squared violations of constraints.
    Violation for pair (i,j): max(0, 2r - dist_ij)^2
    Violation for point i: max(0, r - dist_to_wall)^2
    """
    pts = pts.reshape(N, 2)
    obj = 0.0
    
    # Pairwise violations
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Only consider i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    active = (2 * r - dist) > 0
    active &= mask
    violations = np.where(active, (2 * r - dist)**2, 0)
    obj += np.sum(violations)
    
    # Boundary violations
    d_wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                        np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = np.where(r - d_wall > 0, (r - d_wall)**2, 0)
    obj += np.sum(wall_viol)
    
    return obj

def _soft_violation_grad(pts, r):
    """Gradient of the soft violation objective."""
    pts = pts.reshape(N, 2)
    grad = np.zeros((N, 2))
    
    # Pairwise gradient
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    active = (2 * r - dist) > 0
    active &= mask
    
    # factor[i,j] = -2 * (2r - dist_ij) / dist_ij for active pairs
    safe_dist = np.where(dist > 1e-10, dist, 1.0)
    factor = np.zeros((N, N))
    factor[active] = -2 * (2 * r - dist[active]) / safe_dist[active]
    
    # grad[i] = sum_j factor[i,j] * (pts[i] - pts[j])
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    # Boundary gradient
    d_wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                        np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_active = (r - d_wall) > 0
    
    # Compute which wall is closest for each point
    dx = pts[:, 0]
    dy = pts[:, 1]
    d_left = HALF + dx
    d_right = HALF - dx
    d_bottom = HALF + dy
    d_top = HALF - dy
    
    # Gradient contribution from each wall
    gx = np.zeros(N)
    gy = np.zeros(N)
    
    for i in range(N):
        if not wall_active[i]:
            continue
        v = r - d_wall[i]
        # Determine which wall is closest
        walls = [(d_left[i], -1, 0), (d_right[i], 1, 0), 
                 (d_bottom[i], 0, -1), (d_top[i], 0, 1)]
        min_d, wx, wy = min(walls, key=lambda x: x[0])
        gx[i] = v * wx
        gy[i] = v * wy
    
    grad[:, 0] += gx
    grad[:, 1] += gy
    
    return grad.flatten()

def _simulated_annealing(pts, r, start_time, deadline, T_init=0.01, T_min=1e-6, alpha=0.999):
    """Simulated annealing to find a feasible configuration for radius r."""
    current = pts.copy()
    current_obj = _soft_violation_obj(current, r)
    best = current.copy()
    best_obj = current_obj
    T = T_init
    
    while time.time() < deadline:
        # Generate a perturbation
        step_size = 0.01 * (0.5 + 0.5 * T / T_init)
        delta = np.random.normal(0, step_size, size=current.shape)
        candidate = current + delta
        
        # Clip to bounds
        bounds = np.array([[-HALF + r, HALF - r]] * N)
        candidate[:, 0] = np.clip(candidate[:, 0], -HALF + r, HALF - r)
        candidate[:, 1] = np.clip(candidate[:, 1], -HALF + r, HALF - r)
        
        candidate_obj = _soft_violation_obj(candidate, r)
        
        # Accept if better or with probability exp(-(obj_new - obj_old) / T)
        if candidate_obj < current_obj:
            current = candidate
            current_obj = candidate_obj
            if current_obj < best_obj:
                best = candidate.copy()
                best_obj = current_obj
        else:
            delta_obj = candidate_obj - current_obj
            if np.random.random() < np.exp(-delta_obj / max(T, 1e-10)):
                current = candidate
                current_obj = candidate_obj
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
        
        # Check if we've found a feasible solution
        if best_obj < 1e-10:
            break
    
    return best, best_obj

def _lbfgs_polish(pts, r, start_time, deadline):
    """Polish a configuration using L-BFGS-B."""
    flat_pts = pts.flatten()
    bounds = [(-HALF + r, HALF - r)] * (2 * N)
    
    def obj_and_grad(x):
        pts_x = x.reshape(N, 2)
        o = _soft_violation_obj(pts_x, r)
        g = _soft_violation_grad(pts_x, r)
        return o, g
    
    try:
        res = minimize(
            obj_and_grad,
            flat_pts,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.success or res.fun < 1e-8:
            return res.x.reshape(N, 2), res.fun
    except Exception:
        pass
    
    return pts, _soft_violation_obj(pts, r)

def _refine_radius(pts, r, start_time, deadline):
    """Refine the radius by binary search with SA + polishing."""
    best_r = r
    best_pts = pts.copy()
    
    # Try to increase radius
    delta_r = 0.001
    for _ in range(20):
        if time.time() > deadline:
            break
        
        new_r = best_r + delta_r
        
        # Run SA to find feasible configuration
        sa_pts, sa_obj = _simulated_annealing(best_pts, new_r, time.time(), deadline - 5)
        
        # Polish with L-BFGS
        polished_pts, polish_obj = _lbfgs_polish(sa_pts, new_r, time.time(), deadline - 2)
        
        if polish_obj < 1e-6:
            best_r = new_r
            best_pts = polished_pts
            delta_r *= 0.8
        else:
            # Try smaller step
            delta_r *= 0.5
            if delta_r < 1e-5:
                break
    
    return best_r, best_pts

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initialize with hexagonal pattern
    pts = _hex_init(N)
    
    # Initial radius estimate
    r = _compute_r(pts)
    
    best_r = r
    best_pts = pts.copy()
    
    # Phase 1: SA to improve initial configuration
    sa_pts, sa_obj = _simulated_annealing(pts, r, time.time(), deadline - 30)
    best_pts = sa_pts
    best_r = _compute_r(sa_pts)
    
    # Phase 2: Polish with L-BFGS
    polished_pts, polish_obj = _lbfgs_polish(best_pts, best_r, time.time(), deadline - 10)
    best_pts = polished_pts
    best_r = _compute_r(polished_pts)
    
    # Phase 3: Refine radius
    final_r, final_pts = _refine_radius(best_pts, best_r, time.time(), deadline - 5)
    
    # Final validation and clipping
    final_pts = final_pts.reshape(N, 2)
    for i in range(N):
        for k in range(2):
            if abs(final_pts[i, k]) > HALF - final_r:
                final_pts[i, k] = np.clip(final_pts[i, k], -HALF + final_r, HALF - final_r)
    
    # Recompute actual radius
    actual_r = _compute_r(final_pts)
    
    return final_pts
