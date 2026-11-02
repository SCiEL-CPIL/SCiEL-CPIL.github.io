"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import Voronoi
import sys

N = 21
TIME_LIMIT = 110.0  # Safety margin before 120s kill
START_TIME = time.time()


def _remaining_time():
    return max(0.0, TIME_LIMIT - (time.time() - START_TIME))


def _voronoi_inradii(points, W, H):
    """
    Compute the inradius of each Voronoi cell for points in rectangle [0, W] x [0, H].
    Returns array of radii (one per point).
    """
    n = len(points)
    if n < 3:
        return np.full(n, min(W, H) / 2.0)
    
    try:
        vor = Voronoi(points)
    except Exception:
        # Fallback: use min distance to neighbors / 2
        from scipy.spatial.distance import cdist
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        return np.min(dists, axis=1) / 2.0
    
    radii = np.zeros(n)
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if len(region) == 0 or -1 in region:
            # Unbounded region: use distance to nearest neighbor / 2
            from scipy.spatial.distance import cdist
            dists = cdist(points[i:i+1], points)
            np.fill_diagonal(dists[0], np.inf)
            radii[i] = np.min(dists[0]) / 2.0
            continue
        
        # Get vertices of this Voronoi cell
        vertices = vor.vertices[region]
        if len(vertices) < 3:
            radii[i] = 1e-10
            continue
        
        # Compute inradius: max r such that circle centered at points[i] with radius r
        # is inside the polygon formed by vertices.
        # This is equivalent to: min over all edges of the polygon of the distance
        # from points[i] to the edge line (if the projection falls within the edge segment),
        # or distance to a vertex if projection falls outside.
        
        # More robust: compute distance from point to each edge segment, take min.
        min_dist = np.inf
        for j in range(len(vertices)):
            v1 = vertices[j]
            v2 = vertices[(j + 1) % len(vertices)]
            # Distance from point p to segment v1-v2
            p = points[i]
            d = v2 - v1
            if np.all(d == 0):
                dist = np.linalg.norm(p - v1)
            else:
                t = np.dot(p - v1, d) / np.dot(d, d)
                t = np.clip(t, 0.0, 1.0)
                closest = v1 + t * d
                dist = np.linalg.norm(p - closest)
            min_dist = min(min_dist, dist)
        
        radii[i] = min_dist
    
    return radii


def _objective(x, W, H):
    """
    Objective to maximize sum of Voronoi inradii.
    x = [x_0, y_0, x_1, y_1, ..., x_{n-1}, y_{n-1}]
    """
    points = x.reshape(-1, 2)
    # Ensure points are strictly inside rectangle
    eps = 1e-6
    points = np.clip(points, eps, np.array([W - eps, H - eps]))
    
    radii = _voronoi_inradii(points, W, H)
    return -np.sum(radii)


def _constraints_for_x(x, W, H):
    """
    Constraints: points must be inside rectangle.
    Returns list of constraint dicts for SLSQP.
    """
    points = x.reshape(-1, 2)
    cons = []
    for i in range(len(points)):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - 1e-6})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: W - 1e-6 - x[2*i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - 1e-6})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: H - 1e-6 - x[2*i+1]})
    return cons


def _optimize_for_aspect(ratio, W, H, x0, time_budget):
    """
    Optimize point positions for given W, H.
    Returns best x and objective value.
    """
    if time_budget <= 0:
        return x0, _objective(x0, W, H)
    
    n = len(x0) // 2
    bounds = [(1e-6, W - 1e-6), (1e-6, H - 1e-6)] * n
    
    best_x = x0.copy()
    best_val = _objective(x0, W, H)
    
    # Try a few local optimizations from perturbed starts
    n_starts = min(5, max(1, int(time_budget / 2.0)))
    
    for s in range(n_starts):
        if time_budget <= 0:
            break
        
        if s == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + 0.05 * np.random.randn(len(x0))
            x_start = np.clip(x_start, 1e-5, np.array([W-1e-5, H-1e-5] * n))
        
        cons = _constraints_for_x(x_start, W, H)
        
        try:
            res = minimize(
                _objective, x_start, args=(W, H),
                method='SLSQP', bounds=bounds, constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if res.success or res.fun < best_val:
                best_x = res.x
                best_val = res.fun
        except Exception:
            pass
        
        time_budget = _remaining_time()
    
    return best_x, best_val


def _try_aspect_ratio(W, H, time_budget):
    """
    For given W, H (perimeter = 2*(W+H) = 4, so W+H = 2), find best packing.
    """
    if time_budget <= 0:
        return None, -np.inf
    
    # Initialize with grid-like placement
    side = int(np.ceil(np.sqrt(N)))
    x0 = np.zeros(2 * N)
    for i in range(N):
        row = i // side
        col = i % side
        x0[2*i] = (col + 0.5) / side * W
        x0[2*i+1] = (row + 0.5) / side * H
    
    x_opt, val = _optimize_for_aspect(0, W, H, x0, time_budget)
    
    if val < -np.inf:
        return None, -np.inf
    
    # Reconstruct circles
    points = x_opt.reshape(-1, 2)
    radii = _voronoi_inradii(points, W, H)
    
    # Scale radii slightly to ensure disjoint interiors (safety margin)
    scale = 0.999
    radii = radii * scale
    
    circles = np.column_stack([points, radii])
    score = np.sum(radii)
    
    return circles, score


def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_circles = None
    best_score = -np.inf
    
    # Try several aspect ratios
    # Perimeter 4 => 2*(W+H) = 4 => W+H = 2
    # Try W from 0.3 to 1.7 (H = 2 - W)
    aspect_ratios = np.linspace(0.3, 1.7, 15)
    
    for W in aspect_ratios:
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        t_rem = _remaining_time()
        if t_rem < 3.0:
            break
        
        circles, score = _try_aspect_ratio(W, H, t_rem - 2.0)
        
        if circles is not None and score > best_score:
            best_score = score
            best_circles = circles
    
    # If we still have time, do a more detailed search around the best aspect
    if best_circles is not None and _remaining_time() > 10.0:
        # Find the best W from our trials
        # Re-try with finer grid around promising regions
        for W in np.linspace(0.5, 1.5, 20):
            H = 2.0 - W
            if H <= 0.1:
                continue
            t_rem = _remaining_time()
            if t_rem < 2.0:
                break
            circles, score = _try_aspect_ratio(W, H, t_rem - 1.0)
            if circles is not None and score > best_score:
                best_score = score
                best_circles = circles
    
    if best_circles is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        W, H = 1.0, 1.0
        cells = [((i + 0.5) / side * W, (j + 0.5) / side * H, 0.5 / side * min(W, H) - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
