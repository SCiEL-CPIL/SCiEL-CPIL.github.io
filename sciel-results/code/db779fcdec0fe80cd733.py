"""Heilbronn triangles: 11 points in unit-area equilateral triangle, maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)
# Triangle vertices: (0,0), (1,0), (0.5, S3/2)
# Area = S3/4. We work in coordinates where the triangle has these vertices.
# The evaluator divides by the triangle area, so we maximize min_area / (S3/4).

# Barycentric coordinates: point = a*A + b*B + c*C where a+b+c=1, a,b,c>=0
# A=(0,0), B=(1,0), C=(0.5, S3/2)
# So x = b + 0.5*c, y = (S3/2)*c
# Inverse: c = 2*y/S3, b = x - 0.5*c = x - y/S3, a = 1 - b - c

def _bary_to_xy(bary):
    """Convert (b, c) barycentric coords (a=1-b-c) to (x, y)."""
    b = bary[:, 0]
    c = bary[:, 1]
    x = b + 0.5 * c
    y = (S3 / 2.0) * c
    return np.column_stack([x, y])

def _xy_to_bary(xy):
    """Convert (x, y) to (b, c) barycentric coords."""
    x = xy[:, 0]
    y = xy[:, 1]
    c = 2.0 * y / S3
    b = x - y / S3
    return np.column_stack([b, c])

def _min_triangle_area(xy):
    """Compute minimum area of triangle formed by any 3 points."""
    n = len(xy)
    min_a = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                dx1 = xy[j, 0] - xy[i, 0]
                dy1 = xy[j, 1] - xy[i, 1]
                dx2 = xy[k, 0] - xy[i, 0]
                dy2 = xy[k, 1] - xy[i, 1]
                a = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if a < min_a:
                    min_a = a
    return min_a

def _soft_min_triangle_area(xy, beta):
    """Soft minimum of triangle areas using log-sum-exp (negative log-sum-exp for min)."""
    n = len(xy)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = xy[j, 0] - xy[i, 0]
                dy1 = xy[j, 1] - xy[i, 1]
                dx2 = xy[k, 0] - xy[i, 0]
                dy2 = xy[k, 1] - xy[i, 1]
                a = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                areas.append(a)
    areas = np.array(areas)
    # Soft min: -log(sum(exp(-beta * a))) / beta
    # Use log-sum-exp trick for stability
    m = np.min(areas)
    val = -m - np.log(np.sum(np.exp(-beta * (areas - m)))) / beta
    return val

def _objective_bary(bary_flat, beta):
    """Objective: negative soft-min triangle area (to minimize)."""
    bary = bary_flat.reshape(-1, 2)
    # Clip to valid barycentric: b >= 0, c >= 0, b+c <= 1
    b = np.clip(bary[:, 0], 0, 1)
    c = np.clip(bary[:, 1], 0, 1)
    # Ensure b + c <= 1
    s = b + c
    mask = s > 1
    if np.any(mask):
        scale = 1.0 / s[mask]
        b[mask] *= scale
        c[mask] *= scale
    xy = _bary_to_xy(np.column_stack([b, c]))
    return -_soft_min_triangle_area(xy, beta)

def _penalized_objective(bary_flat, beta, penalty=1000.0):
    """Objective with penalty for going outside triangle."""
    bary = bary_flat.reshape(-1, 2)
    b = bary[:, 0]
    c = bary[:, 1]
    # Penalty for b < 0, c < 0, b+c > 1
    pen = penalty * (np.sum(np.maximum(0, -b)**2) + np.sum(np.maximum(0, -c)**2) + np.sum(np.maximum(0, b + c - 1)**2))
    xy = _bary_to_xy(np.column_stack([np.clip(b,0,1), np.clip(c,0,1)]))
    return -_soft_min_triangle_area(xy, beta) + pen

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    best_score = 0.0
    best_points = None
    
    # Initial good configuration: vertices + centroid + others
    # Let's use a known good starting point
    # Vertices: (0,0), (1,0), (0.5, S3/2)
    # In barycentric: (0,0)->(0,0), (1,0)->(1,0), (0.5,S3/2)->(0,1)
    # Centroid: (1/3, 1/3, 1/3) -> b=1/3, c=1/3
    
    rng = np.random.default_rng(42)
    
    # Generate multiple random initial configurations
    n_restarts = 20
    max_beta_steps = 100
    beta_start = 10.0
    beta_end = 500.0
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random initial barycentric coords
        bary = rng.random((N, 2)) * 0.9
        # Ensure b+c < 1
        s = bary[:, 0] + bary[:, 1]
        mask = s > 0.95
        if np.any(mask):
            scale = 0.95 / s[mask]
            bary[mask] *= scale
        
        current_bary = bary.copy()
        current_score = 0.0
        
        # Simulated annealing / gradient descent with decaying beta
        for step in range(max_beta_steps):
            if time.time() > deadline - 20:
                break
            
            beta = beta_start * (beta_end / beta_start) ** (step / max_beta_steps)
            
            # Compute gradient-based step
            xy = _bary_to_xy(current_bary)
            score = _min_triangle_area(xy)
            if score > best_score:
                best_score = score
                best_points = xy.copy()
            
            # Use Nelder-Mead or gradient descent
            # For efficiency, use a simple gradient approximation
            eps = 0.001
            grad = np.zeros_like(current_bary)
            for i in range(N):
                for j in range(2):
                    orig = current_bary[i, j]
                    current_bary[i, j] = orig + eps
                    f_plus = -_soft_min_triangle_area(_bary_to_xy(current_bary), beta)
                    current_bary[i, j] = orig - eps
                    f_minus = -_soft_min_triangle_area(_bary_to_xy(current_bary), beta)
                    current_bary[i, j] = orig
                    grad[i, j] = (f_plus - f_minus) / (2 * eps)
            
            # Gradient descent step with adaptive learning rate
            lr = 0.01 * (1.0 + step / 10.0)
            current_bary -= lr * grad
            
            # Project back to valid region
            b = np.clip(current_bary[:, 0], 0, 1)
            c = np.clip(current_bary[:, 1], 0, 1)
            s = b + c
            mask = s > 1
            if np.any(mask):
                scale = 1.0 / s[mask]
                b[mask] *= scale
                c[mask] *= scale
            current_bary = np.column_stack([b, c])
        
        # Final polish with Nelder-Mead on the best configuration
        if time.time() > deadline - 10:
            break
        
        # Use the current best from this restart for final evaluation
        xy = _bary_to_xy(current_bary)
        score = _min_triangle_area(xy)
        if score > best_score:
            best_score = score
            best_points = xy.copy()
    
    # If no valid points found, return a default
    if best_points is None:
        # Return vertices and centroid
        pts = np.array([[0, 0], [1, 0], [0.5, S3/2], [1/3, S3/6], [0.5, S3/4],
                        [0.25, S3/8], [0.75, S3/8], [0.5, S3/8], [0.25, S3/4],
                        [0.75, S3/4], [0.5, S3/2*0.8]])
        return pts
    
    return best_points
