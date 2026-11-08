"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _is_valid(circles, tol=1e-9):
    """Check if circles are valid (inside unit square, disjoint interiors)."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < r1 + r2 - tol:
                return False
    return True

def _objective_and_grad(z, n):
    """
    Objective: -sum(r_i)
    Variables: z = [x_1, y_1, r_1, x_2, y_2, r_2, ...]
    Constraints handled via penalty in objective to keep L-BFGS-B simple,
    or we can use bounds and a soft constraint approach.
    
    Actually, L-BFGS-B only handles bounds. We need to handle circle-circle and boundary constraints.
    We will use a penalty method: add a large penalty for violations.
    
    Let's define a penalty function:
    For each circle i:
      Boundary violations: max(0, r_i - x_i), max(0, r_i - y_i), max(0, r_i - (1-x_i)), max(0, r_i - (1-y_i))
    For each pair (i,j):
      dist = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
      violation = max(0, r_i + r_j - dist)
    
    Objective = -sum(r) + lambda * sum(penalties)
    Gradient computed analytically.
    """
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    obj = -np.sum(r)
    grad = np.zeros_like(z)
    grad[2::3] = -1.0  # d(-sum(r))/dr = -1
    
    lam = 100.0  # penalty weight
    
    # Boundary penalties
    # left: r - x > 0
    viol_left = np.maximum(0, r - x)
    if np.any(viol_left > 0):
        obj += lam * np.sum(viol_left)
        grad[0::3] += -lam * (r - x > 0)  # d(r-x)/dx = -1
        grad[2::3] += lam * (r - x > 0)   # d(r-x)/dr = 1
    
    viol_bottom = np.maximum(0, r - y)
    if np.any(viol_bottom > 0):
        obj += lam * np.sum(viol_bottom)
        grad[1::3] += -lam * (r - y > 0)
        grad[2::3] += lam * (r - y > 0)
    
    viol_right = np.maximum(0, r - (1 - x))
    if np.any(viol_right > 0):
        obj += lam * np.sum(viol_right)
        grad[0::3] += lam * (r - (1 - x) > 0)  # d(r-1+x)/dx = 1
        grad[2::3] += lam * (r - (1 - x) > 0)
    
    viol_top = np.maximum(0, r - (1 - y))
    if np.any(viol_top > 0):
        obj += lam * np.sum(viol_top)
        grad[1::3] += lam * (r - (1 - y) > 0)
        grad[2::3] += lam * (r - (1 - y) > 0)
    
    # Circle-circle penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-12:
                dist = 1e-12
            violation = r[i] + r[j] - dist
            if violation > 0:
                obj += lam * violation
                # d(r_i + r_j - dist)/dx_i = -dx/dist
                # d(r_i + r_j - dist)/dy_i = -dy/dist
                # d(r_i + r_j - dist)/dr_i = 1
                # d(r_i + r_j - dist)/dx_j = dx/dist
                # d(r_i + r_j - dist)/dy_j = dy/dist
                # d(r_i + r_j - dist)/dr_j = 1
                grad[3*i + 0] += -lam * dx / dist
                grad[3*i + 1] += -lam * dy / dist
                grad[3*i + 2] += lam
                grad[3*j + 0] += lam * dx / dist
                grad[3*j + 1] += lam * dy / dist
                grad[3*j + 2] += lam
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    init = np.array(cells[:N])
    z0 = np.array([[init[i][0], init[i][1], init[i][2]] for i in range(N)]).flatten()
    
    # Bounds: x, y in [0,1], r in [0, 0.5]
    bounds = []
    for _ in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((1e-6, 0.5))  # r
    
    best_z = z0.copy()
    best_val = np.sum(init[:, 2])
    
    # Try L-BFGS-B
    try:
        res = minimize(_objective_and_grad, z0, args=(N,), method='L-BFGS-B', 
                       jac=True, bounds=bounds, options={'maxiter': 200, 'maxfun': 5000})
        if time.time() < deadline:
            z_opt = res.x
            # Check validity
            circles = np.array([z_opt[3*i:3*i+3] for i in range(N)])
            if _is_valid(circles):
                val = np.sum(circles[:, 2])
                if val > best_val:
                    best_val = val
                    best_z = z_opt.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        # Return best found
        circles = np.array([best_z[3*i:3*i+3] for i in range(N)])
        return circles
    
    # Local search / perturbation to escape local minima
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        
        # Perturb initial guess
        z0_pert = best_z + np.random.normal(0, 0.05, size=best_z.shape)
        # Clamp to bounds
        z0_pert[0::3] = np.clip(z0_pert[0::3], 0, 1)
        z0_pert[1::3] = np.clip(z0_pert[1::3], 0, 1)
        z0_pert[2::3] = np.clip(z0_pert[2::3], 1e-6, 0.5)
        
        try:
            res = minimize(_objective_and_grad, z0_pert, args=(N,), method='L-BFGS-B', 
                           jac=True, bounds=bounds, options={'maxiter': 100, 'maxfun': 2000})
            if time.time() > deadline - 5:
                break
            z_opt = res.x
            circles = np.array([z_opt[3*i:3*i+3] for i in range(N)])
            if _is_valid(circles):
                val = np.sum(circles[:, 2])
                if val > best_val:
                    best_val = val
                    best_z = z_opt.copy()
        except Exception:
            continue
    
    circles = np.array([best_z[3*i:3*i+3] for i in range(N)])
    return circles

# EVOLVE-BLOCK-END
