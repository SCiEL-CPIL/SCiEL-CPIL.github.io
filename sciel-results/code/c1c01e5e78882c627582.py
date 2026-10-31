import numpy as np
from scipy.optimize import minimize
import time
import random
from itertools import combinations

N = 11
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])

# Pre-compute triangle indices
TRI_INDICES = np.array(list(combinations(range(N), 3)))
N_TRI = len(TRI_INDICES)

def calc_min_area(points):
    """
    points: (11, 2)
    returns: min area
    """
    p1 = points[TRI_INDICES[:, 0]]
    p2 = points[TRI_INDICES[:, 1]]
    p3 = points[TRI_INDICES[:, 2]]
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def is_inside_triangle(points):
    """
    Check if all points are inside or on the boundary of the triangle.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    eps = 1e-9
    cond1 = y >= -eps
    cond2 = y <= np.sqrt(3) * x + eps
    cond3 = y <= -np.sqrt(3) * (x - 1) + eps
    
    return np.all(cond1) and np.all(cond2) and np.all(cond3)

def penalty(points):
    """
    Calculate penalty for points outside the triangle.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    d1 = np.maximum(0, -y)
    d2 = np.maximum(0, (y - np.sqrt(3)*x) / 2)
    d3 = np.maximum(0, (y + np.sqrt(3)*x - np.sqrt(3)) / 2)
    
    return np.sum(d1**2) + np.sum(d2**2) + np.sum(d3**2)

def _all_areas(points):
    """Return all 165 triangle areas."""
    p1 = points[TRI_INDICES[:, 0]]
    p2 = points[TRI_INDICES[:, 1]]
    p3 = points[TRI_INDICES[:, 2]]
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    areas = 0.5 * np.abs(cross)
    return areas

def _soft_min_area(points, temperature=0.01):
    """
    Smooth approximation of min area using log-sum-exp of reciprocal areas.
    min(a_i) ≈ -temperature * log(sum(exp(-a_i / temperature)))
    """
    areas = _all_areas(points)
    # Avoid log(0) by clipping
    areas_clipped = np.maximum(areas, 1e-12)
    # Log-sum-exp for numerical stability
    max_inv = np.max(-areas_clipped / temperature)
    log_sum_exp = max_inv + np.log(np.sum(np.exp(-areas_clipped / temperature - max_inv)))
    return -temperature * log_sum_exp

def _objective_smooth(points_flat, temperature=0.01):
    """
    Objective for L-BFGS-B: minimize -soft_min_area + penalty.
    """
    points = points_flat.reshape(N, 2)
    soft_min = _soft_min_area(points, temperature)
    pen = penalty(points)
    return -soft_min + 100.0 * pen

def _objective_grad(points_flat, temperature=0.01):
    """
    Gradient of the smooth objective.
    """
    points = points_flat.reshape(N, 2)
    areas = _all_areas(points)
    areas_clipped = np.maximum(areas, 1e-12)
    
    # Gradient of soft_min_area w.r.t. points
    # d/dp [-T * log(sum(exp(-a_i/T)))] = sum(exp(-a_i/T) * da_i/dp) / sum(exp(-a_i/T))
    weights = np.exp(-areas_clipped / temperature)
    weights /= np.sum(weights)
    
    # Gradient of each area w.r.t. points
    # Area_i = 0.5 * |cross_i|
    # For simplicity, assume all cross products are positive (we can handle sign later)
    # Actually, let's compute gradient properly
    
    p1 = points[TRI_INDICES[:, 0]]
    p2 = points[TRI_INDICES[:, 1]]
    p3 = points[TRI_INDICES[:, 2]]
    
    # cross_i = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
    # d(cross)/dx1 = -(y3-y1), d(cross)/dy1 = (x3-x1)
    # d(cross)/dx2 = (y3-y1), d(cross)/dy2 = -(x3-x1)
    # d(cross)/dx3 = -(y2-y1), d(cross)/dy3 = (x2-x1)
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    
    # Sign of cross for absolute value
    sign = np.sign(cross)
    sign[sign == 0] = 1  # avoid zero
    
    # Gradient of area_i = 0.5 * sign_i * cross_i
    # d(area_i)/dx1 = -0.5 * sign_i * (y3-y1)
    # d(area_i)/dy1 = 0.5 * sign_i * (x3-x1)
    # d(area_i)/dx2 = 0.5 * sign_i * (y3-y1)
    # d(area_i)/dy2 = -0.5 * sign_i * (x3-x1)
    # d(area_i)/dx3 = -0.5 * sign_i * (y2-y1)
    # d(area_i)/dy3 = 0.5 * sign_i * (x2-x1)
    
    # Initialize gradient
    grad = np.zeros((N, 2))
    
    # Accumulate weighted gradients
    for i in range(N_TRI):
        w = 0.5 * weights[i] * sign[i]
        idx1, idx2, idx3 = TRI_INDICES[i]
        
        # d(area_i)/dx1
        grad[idx1, 0] += -w * (y3[i] - y1[i])
        grad[idx1, 1] += w * (x3[i] - x1[i])
        
        # d(area_i)/dx2
        grad[idx2, 0] += w * (y3[i] - y1[i])
        grad[idx2, 1] += -w * (x3[i] - x1[i])
        
        # d(area_i)/dx3
        grad[idx3, 0] += -w * (y2[i] - y1[i])
        grad[idx3, 1] += w * (x2[i] - x1[i])
    
    # Gradient of -soft_min_area is -grad(soft_min_area)
    # But soft_min_area = -T * log_sum_exp, so d(soft_min_area)/dp = sum(w_i * d(area_i)/dp)
    # Wait, let me re-derive:
    # f = -T * log(sum(exp(-a_i/T)))
    # df/da_i = exp(-a_i/T) / sum(exp(-a_j/T)) = w_i
    # df/dp = sum_i w_i * da_i/dp
    # We want to minimize -f, so gradient is -df/dp = -sum_i w_i * da_i/dp
    
    # The grad above is sum_i w_i * da_i/dp (where w_i = 0.5*weights[i]*sign[i])
    # Actually, I computed d(area_i)/dp, and weights are normalized
    # Let me redo: the gradient of soft_min_area w.r.t. p is sum_i (exp(-a_i/T)/sum(exp)) * da_i/dp
    # But da_i/dp already includes the 0.5*sign factor
    # So grad computed above = sum_i w_i * da_i/dp where w_i = weights[i] (normalized exp)
    # And da_i/dp includes 0.5*sign
    # So this is correct for d(soft_min_area)/dp
    
    # We want to minimize -soft_min_area + penalty
    # Gradient = -d(soft_min_area)/dp + d(penalty)/dp
    
    # Penalty gradient
    x = points[:, 0]
    y = points[:, 1]
    
    d1 = np.maximum(0, -y)
    d2 = np.maximum(0, (y - np.sqrt(3)*x) / 2)
    d3 = np.maximum(0, (y + np.sqrt(3)*x - np.sqrt(3)) / 2)
    
    # d(penalty)/dx
    # d1/dx = 0
    # d2/dx = -sqrt(3)/2 if d2 > 0, else 0
    # d3/dx = sqrt(3)/2 if d3 > 0, else 0
    
    pen_grad_x = np.zeros(N)
    pen_grad_y = np.zeros(N)
    
    mask1 = d1 > 0
    pen_grad_y[mask1] += -2 * d1[mask1]  # d(-y)^2/dy = -2y, but d1 = max(0,-y), so if d1>0, d1=-y, d(d1)/dy = -1, d(d1^2)/dy = 2*d1*(-1) = -2*d1
    
    mask2 = d2 > 0
    pen_grad_x[mask2] += 2 * d2[mask2] * (-np.sqrt(3)/2)  # d((y-sqrt(3)x)/2)^2/dx = 2*d2 * (-sqrt(3)/2)
    pen_grad_y[mask2] += 2 * d2[mask2] * (1/2)
    
    mask3 = d3 > 0
    pen_grad_x[mask3] += 2 * d3[mask3] * (np.sqrt(3)/2)
    pen_grad_y[mask3] += 2 * d3[mask3] * (1/2)
    
    # Total gradient: -grad(soft_min_area) + 100 * grad(penalty)
    total_grad = -grad.flatten() + 100.0 * np.concatenate([pen_grad_x, pen_grad_y])
    
    return total_grad

def _random_inside(rng):
    """Generate a random point inside the triangle."""
    while True:
        x = rng.random()
        y = rng.random() * np.sqrt(3)/2
        if y <= np.sqrt(3)*x and y <= -np.sqrt(3)*(x-1):
            return np.array([x, y])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_val = -np.inf
    
    # Multiple random starts with L-BFGS-B
    n_starts = 10
    temperature = 0.01
    
    for start_idx in range(n_starts):
        if time.time() > deadline - 5:
            break
        
        # Generate random initial points
        pts = np.array([_random_inside(rng) for _ in range(N)])
        
        # Evaluate initial
        curr_val = calc_min_area(pts)
        if curr_val > best_val:
            best_val = curr_val
            best_pts = pts.copy()
        
        # Run L-BFGS-B from this start
        try:
            x0 = pts.flatten()
            
            result = minimize(
                _objective_smooth,
                x0,
                args=(temperature,),
                jac=_objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8}
            )
            
            new_pts = result.x.reshape(N, 2)
            
            # Check if inside
            if is_inside_triangle(new_pts):
                new_val = calc_min_area(new_pts)
                if new_val > best_val:
                    best_val = new_val
                    best_pts = new_pts.copy()
                    
        except Exception as e:
            pass
    
    # Final refinement with Nelder-Mead on the best point
    if best_pts is not None and time.time() < deadline - 10:
        try:
            x0 = best_pts.flatten()
            
            def _obj_nm(x):
                p = x.reshape(N, 2)
                return -calc_min_area(p) + 100.0 * penalty(p)
            
            result = minimize(_obj_nm, x0, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            new_pts = result.x.reshape(N, 2)
            if is_inside_triangle(new_pts):
                new_val = calc_min_area(new_pts)
                if new_val > best_val:
                    best_val = new_val
                    best_pts = new_pts.copy()
                    
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return random points
        best_pts = np.array([_random_inside(rng) for _ in range(N)])
    
    return best_pts
