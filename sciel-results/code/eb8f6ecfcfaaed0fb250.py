"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

DEADLINE = 140.0  # safety margin before 150s kill

def _barycentric_to_cartesian(bary):
    """Convert barycentric coords (u, v) to cartesian in triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    u = bary[:, 0]
    v = bary[:, 1]
    x = u + 0.5 * v
    y = (3 ** 0.5) / 2.0 * v
    return np.column_stack([x, y])

def _triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                areas.append(0.5 * abs(cross))
    return np.array(areas)

def _soft_min(areas, tau):
    """Soft minimum using log-sum-exp."""
    return -tau * logsumexp(-areas / tau)

def _objective(bary, tau):
    """Objective to minimize: negative soft-min of triangle areas."""
    points = _barycentric_to_cartesian(bary)
    areas = _triangle_areas(points)
    return -_soft_min(areas, tau)

def _objective_grad(bary, tau):
    """Gradient of the objective (negative soft-min)."""
    points = _barycentric_to_cartesian(bary)
    n = len(points)
    areas = np.zeros((n, n, n))
    # Compute all triangle areas
    idx = []
    area_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                area_vals.append(area)
                idx.append((i, j, k))
    
    area_vals = np.array(area_vals)
    # Soft-min weights
    neg_areas = -area_vals / tau
    weights = np.exp(neg_areas - logsumexp(neg_areas))
    
    # Gradient w.r.t. each point
    grad_points = np.zeros_like(points)
    for w, (i, j, k) in zip(weights, idx):
        p1, p2, p3 = points[i], points[j], points[k]
        # d(area)/d(p) for area = 0.5 * |cross(p2-p1, p3-p1)|
        # For the signed cross product (without abs), gradient is straightforward
        # cross = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
        # d(cross)/dx1 = -(y3-y1), d(cross)/dy1 = (x3-x1)
        # d(cross)/dx2 = (y3-y1), d(cross)/dy2 = -(x3-y1)
        # d(cross)/dx3 = -(y2-y1), d(cross)/dy3 = (x2-x1)
        
        y3y1 = p3[1] - p1[1]
        x3x1 = p3[0] - p1[0]
        y2y1 = p2[1] - p1[1]
        x2x1 = p2[0] - p1[0]
        
        signed_cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        sign = 1.0 if signed_cross >= 0 else -1.0
        
        # Gradient of signed cross w.r.t. points
        # Then multiply by sign for abs, and by 0.5 for area
        dc_dx1 = -y3y1
        dc_dy1 = x3x1
        dc_dx2 = y3y1
        dc_dy2 = -x3y1 if False else -(x3x1)  # wait, let me recompute
        # Actually: cross = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
        # d/dx2 = (y3-y1), d/dy2 = -(x3-x1)
        # d/dx3 = -(y2-y1), d/dy3 = (x2-x1)
        dc_dx2 = y3y1
        dc_dy2 = -x3x1
        dc_dx3 = -y2y1
        dc_dy3 = x2x1
        
        factor = 0.5 * sign * w
        
        grad_points[i, 0] += factor * dc_dx1
        grad_points[i, 1] += factor * dc_dy1
        grad_points[j, 0] += factor * dc_dx2
        grad_points[j, 1] += factor * dc_dy2
        grad_points[k, 0] += factor * dc_dx3
        grad_points[k, 1] += factor * dc_dy3
    
    # Now map gradient from cartesian to barycentric
    # x = u + 0.5*v, y = sqrt(3)/2 * v
    # dx/du = 1, dx/dv = 0.5
    # dy/du = 0, dy/dv = sqrt(3)/2
    # So grad_u = grad_x, grad_v = 0.5*grad_x + sqrt(3)/2*grad_y
    grad_bary = np.zeros_like(bary)
    grad_bary[:, 0] = grad_points[:, 0]
    grad_bary[:, 1] = 0.5 * grad_points[:, 0] + (3 ** 0.5) / 2.0 * grad_points[:, 1]
    
    return -grad_bary  # negative because we're minimizing -soft_min

def _random_barycentric(n_points, rng):
    """Generate random points in barycentric coordinates inside the triangle."""
    bary = np.zeros((n_points, 2))
    for i in range(n_points):
        u = rng.random()
        v = rng.random() * (1 - u)
        bary[i, 0] = u
        bary[i, 1] = v
    return bary

def _evaluate(points):
    """Compute the minimum triangle area."""
    areas = _triangle_areas(points)
    return np.min(areas)

def construct():
    start_time = time.time()
    best_points = None
    best_value = -np.inf
    
    # Try multiple temperatures and restarts
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002]
    n_restarts = 8
    
    for tau in temperatures:
        for restart in range(n_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            rng = np.random.RandomState(42 + restart * 1000 + int(tau * 10000))
            bary0 = _random_barycentric(N, rng)
            
            try:
                result = minimize(
                    _objective,
                    bary0,
                    args=(tau,),
                    method='L-BFGS-B',
                    jac=_objective_grad,
                    bounds=[(0, 1)] * (2 * N),
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                bary_opt = result.x
                points = _barycentric_to_cartesian(bary_opt)
                value = _evaluate(points)
                
                if value > best_value:
                    best_value = value
                    best_points = points.copy()
                    
            except Exception:
                pass
        
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # If no good result, use a fallback
    if best_points is None:
        rng = np.random.RandomState(0)
        bary = _random_barycentric(N, rng)
        best_points = _barycentric_to_cartesian(bary)
    
    return best_points

# EVOLVE-BLOCK-END
