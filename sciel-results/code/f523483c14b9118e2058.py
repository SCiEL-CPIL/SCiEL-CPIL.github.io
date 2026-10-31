"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Voronoi-based inradius maximization with gradient ascent.
- Each circle is inscribed in its Voronoi cell
- Optimize cell centroids (and radii) to maximize sum of inradii
- Use the dual graph (Delaunay) to enforce topological stability
- This transforms non-convex packing into smoother polygonal partition optimization
"""

import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay
from scipy.spatial.distance import cdist

N = 26
# EVOLVE-BLOCK-START

def _is_valid(circles, tol=1e-7):
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1.0 + tol:
            return False
        if y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - tol:
                return False
    return True


def _post_process(circles, tol=1e-9):
    out = []
    for x, y, r in circles:
        r = max(0.0, r - tol)
        x = min(1.0 - r, max(r, x))
        y = min(1.0 - r, max(r, y))
        out.append((x, y, r))
    return out


def _sum_radii(circles):
    return sum(c[2] for c in circles)


def _voronoi_inradii(points, radii):
    """
    Compute the inradius of each Voronoi cell for the given points.
    Returns array of inradii (distance from point to closest cell boundary).
    """
    n = len(points)
    inradii = np.zeros(n)
    
    # Create Voronoi diagram
    try:
        vor = Voronoi(points)
    except Exception:
        # Fallback: use minimum distance to other points / 2 and to boundary
        for i in range(n):
            min_dist = 1.0
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(points[i] - points[j])
                    min_dist = min(min_dist, d / 2.0)
            # Distance to boundary
            min_dist = min(min_dist, points[i][0], points[i][1], 1.0 - points[i][0], 1.0 - points[i][1])
            inradii[i] = min_dist
        return inradii
    
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            # Unbounded region - use distance to boundary
            min_dist = min(points[i][0], points[i][1], 1.0 - points[i][0], 1.0 - points[i][1])
            # Also consider distance to nearest other point
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(points[i] - points[j])
                    min_dist = min(min_dist, d / 2.0)
            inradii[i] = min_dist
        else:
            # Get the vertices of this region
            vertices = vor.vertices[region]
            # Compute distance from point i to each edge of the polygon
            min_dist = np.inf
            for k in range(len(vertices)):
                v1 = vertices[k]
                v2 = vertices[(k + 1) % len(vertices)]
                # Distance from point to line segment
                edge = v2 - v1
                edge_len = np.linalg.norm(edge)
                if edge_len < 1e-12:
                    continue
                edge_unit = edge / edge_len
                t = np.dot(points[i] - v1, edge_unit)
                t = np.clip(t, 0, edge_len)
                closest = v1 + t * edge_unit
                dist = np.linalg.norm(points[i] - closest)
                min_dist = min(min_dist, dist)
            
            # Also check distance to square boundary
            min_dist = min(min_dist, points[i][0], points[i][1], 1.0 - points[i][0], 1.0 - points[i][1])
            
            inradii[i] = min_dist
    
    return inradii


def _voronoi_objective(params, w=100.0):
    """
    Objective: -sum(min(inradius_i, r_i)) + w * sum(softplus(violations))
    where violations = r_i - inradius_i (if circle exceeds its cell)
    """
    n = N
    points = params[0:2*n].reshape(n, 2)
    radii = params[2*n:3*n]
    
    inradii = _voronoi_inradii(points, radii)
    
    # The effective radius is min(r_i, inradius_i)
    # We want to maximize sum of effective radii
    effective = np.minimum(radii, inradii)
    
    obj = -np.sum(effective)
    
    # Penalty for circles that exceed their Voronoi cell
    violations = radii - inradii
    softplus_violations = np.log1p(np.exp(np.maximum(violations * 50, -50))) / 50
    obj += w * np.sum(softplus_violations)
    
    return obj


def _voronoi_grad(params, w=100.0):
    """
    Numerical gradient of the Voronoi objective.
    """
    eps = 1e-6
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _voronoi_objective(params_plus, w)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = _voronoi_objective(params_minus, w)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return grad


def _voronoi_optimize(params, w=100.0, max_iter=50):
    """
    Optimize using L-BFGS-B with Voronoi-based objective.
    """
    bounds = [(0.01, 0.99)] * (2 * N) + [(0.001, 0.5)] * N
    
    try:
        result = minimize(
            lambda p: _voronoi_objective(p, w),
            params,
            jac=lambda p: _voronoi_grad(p, w),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return result.x
    except Exception:
        return params


def _slsqp_polish(circles):
    """SLSQP polish with hard inequality constraints."""
    params = np.array([val for c in circles for val in c])
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.5)] * N
    
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i+1] - p[3*i+2]})
    
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2
            })
    
    def objective(p):
        return -sum(p[3*i+2] for i in range(N))
    
    try:
        result = minimize(
            objective, params,
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        if result.success:
            return [(result.x[3*i], result.x[3*i+1], result.x[3*i+2]) for i in range(N)]
    except Exception:
        pass
    return circles


def _init_hexagonal(jitter=0.0):
    circles = []
    r = 0.075
    rows = [5, 5, 5, 5, 6]
    row_spacing = r * np.sqrt(3)
    col_spacing = 2 * r
    row_idx = 0
    for count in rows:
        offset = 0 if row_idx % 2 == 0 else col_spacing / 2
        total_width = col_spacing * (count - 1) + offset
        start_x = (1.0 - total_width) / 2.0
        y = r + row_spacing * row_idx + 0.03
        for col in range(count):
            x = start_x + offset + col * col_spacing
            if jitter > 0:
                x += random.uniform(-jitter, jitter)
                y += random.uniform(-jitter, jitter)
            if len(circles) < N:
                circles.append((x, y, r))
        row_idx += 1
    circles = circles[:N]
    while len(circles) < N:
        circles.append((0.5, 0.5, 0.01))
    return circles


def _init_grid():
    side = int(np.ceil(np.sqrt(N)))
    circles = []
    for k in range(N):
        i = k % side
        j = k // side
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.4 / side
        circles.append((x, y, r))
    return circles


def _init_corners():
    circles = []
    circles.append((0.08, 0.08, 0.08))
    circles.append((0.92, 0.08, 0.08))
    circles.append((0.08, 0.92, 0.08))
    circles.append((0.92, 0.92, 0.08))
    circles.append((0.5, 0.05, 0.06))
    circles.append((0.5, 0.95, 0.06))
    circles.append((0.05, 0.5, 0.06))
    circles.append((0.95, 0.5, 0.06))
    random.seed(42)
    while len(circles) < N:
        x = random.uniform(0.15, 0.85)
        y = random.uniform(0.15, 0.85)
        r = random.uniform(0.04, 0.07)
        circles.append((x, y, r))
    return circles


def _init_random(seed):
    random.seed(seed)
    circles = []
    for _ in range(N):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        r = random.uniform(0.03, 0.08)
        circles.append((x, y, r))
    return circles


def _circles_to_params(circles):
    return np.array([val for c in circles for val in c])


def _params_to_circles(params):
    return [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]


def _voronoi_refine(circles, n_iters=5, w_schedule=[10.0, 50.0, 200.0]):
    """
    Refine circles using Voronoi-based optimization.
    """
    params = _circles_to_params(circles)
    
    for w in w_schedule:
        params = _voronoi_optimize(params, w=w, max_iter=30)
    
    # Set radii to be the inradii of the Voronoi cells
    points = params[0:2*N].reshape(N, 2)
    inradii = _voronoi_inradii(points, params[2*N:3*N])
    
    # Clip radii to be valid
    for i in range(N):
        x, y = points[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        params[2*N + i] = min(inradii[i], max_r)
    
    return _params_to_circles(params)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Diverse initializations
    inits = [
        _init_hexagonal(0.0),
        _init_hexagonal(0.02),
        _init_hexagonal(0.04),
        _init_grid(),
        _init_corners(),
        _init_random(42),
        _init_random(123),
        _init_random(777),
        _init_random(999),
        _init_random(314),
        _init_random(271),
    ]
    
    best_circles = inits[0]
    best_sum = _sum_radii(best_circles)
    
    # For each init, run Voronoi-based refinement
    for init_idx, init_circles in enumerate(inits):
        if time.time() > deadline - 25.0:
            break
        
        # Voronoi refinement
        refined = _voronoi_refine(init_circles, n_iters=3, w_schedule=[10.0, 50.0, 200.0])
        
        if _is_valid(refined, tol=1e-6):
            s = _sum_radii(refined)
            if s > best_sum:
                best_sum = s
                best_circles = refined
    
    # SLSQP polish on best
    if time.time() < deadline - 15.0:
        polished = _slsqp_polish(best_circles)
        if _is_valid(polished, tol=1e-7):
            ps = _sum_radii(polished)
            if ps > best_sum:
                best_sum = ps
                best_circles = polished
    
    # One more round: small random perturbation + Voronoi
    if time.time() < deadline - 10.0:
        random.seed(int(time.time()) % 10000)
        perturbed = []
        for x, y, r in best_circles:
            x = np.clip(x + random.uniform(-0.01, 0.01), 0.01, 0.99)
            y = np.clip(y + random.uniform(-0.01, 0.01), 0.01, 0.99)
            perturbed.append((x, y, r))
        
        refined2 = _voronoi_refine(perturbed, n_iters=2, w_schedule=[50.0, 200.0])
        if _is_valid(refined2, tol=1e-6):
            ps = _sum_radii(refined2)
            if ps > best_sum:
                best_sum = ps
                best_circles = refined2
    
    # Final SLSQP
    if time.time() < deadline - 5.0:
        polished2 = _slsqp_polish(best_circles)
        if _is_valid(polished2, tol=1e-7):
            ps = _sum_radii(polished2)
            if ps > best_sum:
                best_sum = ps
                best_circles = polished2
    
    best_circles = _post_process(best_circles)
    
    if not _is_valid(best_circles):
        best_circles = _post_process(_init_grid())
    
    return np.array(best_circles)

# EVOLVE-BLOCK-END
