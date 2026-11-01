"""Improved circle packing using L-BFGS-B with penalty method."""
import numpy as np
import time
from scipy.optimize import minimize
import random

N = 32

def _objective_and_grad(v, penalty_weight=100.0):
    """
    Compute objective value and gradient for L-BFGS-B.
    v is a vector of length 3*N: [x0, y0, r0, x1, y1, r1, ...]
    Objective: -sum(r_i) + penalty_weight * sum(violations^2)
    """
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    obj = -np.sum(r)
    grad = np.zeros_like(v)
    grad[2::3] = -1.0  # Gradient of -sum(r)
    
    # Pairwise overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            d = np.sqrt(d_sq) if d_sq > 1e-12 else 1e-12
            r_sum = r[i] + r[j]
            
            if d < r_sum:
                viol = r_sum - d
                obj += penalty_weight * viol * viol
                
                # Gradient of penalty: d/dv (w * (r_i + r_j - d)^2)
                # = 2 * w * (r_i + r_j - d) * d/dv (r_i + r_j - d)
                # d/dx_i (r_i + r_j - d) = -dx/d
                # d/dy_i (r_i + r_j - d) = -dy/d
                # d/dr_i (r_i + r_j - d) = 1
                # d/dx_j (r_i + r_j - d) = dx/d
                # d/dy_j (r_i + r_j - d) = dy/d
                # d/dr_j (r_i + r_j - d) = 1
                
                coeff = 2.0 * penalty_weight * viol
                inv_d = 1.0 / d
                
                grad[3*i + 0] += coeff * (-dx * inv_d)
                grad[3*i + 1] += coeff * (-dy * inv_d)
                grad[3*i + 2] += coeff * 1.0
                
                grad[3*j + 0] += coeff * (dx * inv_d)
                grad[3*j + 1] += coeff * (dy * inv_d)
                grad[3*j + 2] += coeff * 1.0
    
    # Boundary penalties
    for i in range(N):
        # Left
        if x[i] < r[i]:
            viol = r[i] - x[i]
            obj += penalty_weight * viol * viol
            coeff = 2.0 * penalty_weight * viol
            grad[3*i + 0] += coeff * (-1.0)
            grad[3*i + 2] += coeff * 1.0
            
        # Right
        if x[i] + r[i] > 1.0:
            viol = x[i] + r[i] - 1.0
            obj += penalty_weight * viol * viol
            coeff = 2.0 * penalty_weight * viol
            grad[3*i + 0] += coeff * 1.0
            grad[3*i + 2] += coeff * 1.0
            
        # Bottom
        if y[i] < r[i]:
            viol = r[i] - y[i]
            obj += penalty_weight * viol * viol
            coeff = 2.0 * penalty_weight * viol
            grad[3*i + 1] += coeff * (-1.0)
            grad[3*i + 2] += coeff * 1.0
            
        # Top
        if y[i] + r[i] > 1.0:
            viol = y[i] + r[i] - 1.0
            obj += penalty_weight * viol * viol
            coeff = 2.0 * penalty_weight * viol
            grad[3*i + 1] += coeff * 1.0
            grad[3*i + 2] += coeff * 1.0
            
    return obj, grad

def _is_valid(x, y, r, tol=1e-5):
    """Check if configuration is valid."""
    for i in range(N):
        if x[i] < r[i] - tol or x[i] + r[i] > 1.0 + tol:
            return False
        if y[i] < r[i] - tol or y[i] + r[i] > 1.0 + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _hex_init(n):
    """Generate hexagonal packing initialization."""
    points = []
    # Generate a grid of hexagonal points
    for q in range(-5, 6):
        for r in range(-5, 6):
            x = 1.5 * q
            y = np.sqrt(3) * (r + 0.5 * (q & 1))
            points.append([x, y])
    points = np.array(points)
    
    # Center the grid
    center = np.array([0.5, 0.5])
    points += (center - np.mean(points, axis=0))
    
    # Select n closest to center
    dists = np.sum((points - center)**2, axis=1)
    sorted_idx = np.argsort(dists)
    init_points = points[sorted_idx[:n]]
    
    # Initial radii: estimate based on local spacing
    # For hex packing, distance between centers is ~ 2r.
    # Average distance in our grid is ~ sqrt(3) or 1.5.
    # Let's start with a conservative radius.
    radii = np.full(n, 0.08)
    return init_points, radii

def _random_init(n, seed):
    """Generate random initialization."""
    rng = np.random.RandomState(seed)
    points = rng.rand(n, 2)
    radii = rng.uniform(0.05, 0.1, n)
    return points, radii

def _grid_init(n):
    """Generate grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    points = []
    radii = []
    for i in range(side):
        for j in range(side):
            if len(points) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                points.append([x, y])
                radii.append(0.5 / side - 0.01)
    return np.array(points), np.array(radii)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_sum = 0.0
    best_vars = None
    
    # Define bounds for L-BFGS-B
    # x, y in [0, 1]
    # r in [0, 0.5] (theoretical max, but practically smaller)
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.01, 0.5)) # r
    
    # List of initializations
    inits = []
    
    # 1. Hexagonal
    pts, rads = _hex_init(N)
    v0 = np.zeros(3*N)
    v0[0::3] = pts[:, 0]
    v0[1::3] = pts[:, 1]
    v0[2::3] = rads
    inits.append(v0)
    
    # 2. Grid
    pts, rads = _grid_init(N)
    v0 = np.zeros(3*N)
    v0[0::3] = pts[:, 0]
    v0[1::3] = pts[:, 1]
    v0[2::3] = rads
    inits.append(v0)
    
    # 3. Random starts
    for seed in range(5):
        pts, rads = _random_init(N, seed=42+seed)
        v0 = np.zeros(3*N)
        v0[0::3] = pts[:, 0]
        v0[1::3] = pts[:, 1]
        v0[2::3] = rads
        inits.append(v0)
        
    # Run optimization for each initialization
    for i, v0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Use a moderate penalty weight. 
        # If penalty is too high, the optimizer focuses on constraints and ignores objective.
        # If too low, it violates constraints.
        # We can try increasing penalty weight if result is invalid.
        penalty_weights = [100.0, 500.0, 1000.0]
        
        for pw in penalty_weights:
            if time.time() > deadline:
                break
                
            try:
                # L-BFGS-B
                res = minimize(
                    _objective_and_grad, 
                    v0, 
                    args=(pw,), 
                    method='L-BFGS-B', 
                    jac=True, 
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
            except Exception as e:
                continue
                
            if res.success or res.status == 0:
                v_opt = res.x
                x = v_opt[0::3]
                y = v_opt[1::3]
                r = v_opt[2::3]
                
                # Check validity
                if _is_valid(x, y, r):
                    sum_r = np.sum(r)
                    if sum_r > best_sum:
                        best_sum = sum_r
                        best_vars = v_opt.copy()
                else:
                    # If invalid, try with higher penalty weight in next iteration
                    # Or just skip if we haven't tried higher penalty yet
                    pass
                    
    # If no valid solution found, fallback to grid
    if best_vars is None:
        pts, rads = _grid_init(N)
        v0 = np.zeros(3*N)
        v0[0::3] = pts[:, 0]
        v0[1::3] = pts[:, 1]
        v0[2::3] = rads
        best_vars = v0
        best_sum = np.sum(rads)
        
    # Final polish: if best_vars is slightly invalid, try to fix it
    # But L-BFGS-B with high penalty should produce valid results.
    # Let's do a quick check and if invalid, scale down radii slightly to ensure validity
    # This might reduce score but ensures validity.
    x = best_vars[0::3]
    y = best_vars[1::3]
    r = best_vars[2::3]
    
    if not _is_valid(x, y, r):
        # Try to fix by scaling radii
        scale = 1.0
        for _ in range(10):
            scale *= 0.95
            r_test = r * scale
            if _is_valid(x, y, r_test):
                r = r_test
                break
        best_vars[2::3] = r
        best_sum = np.sum(r)
        
    result = np.column_stack((x, y, r))
    return result
