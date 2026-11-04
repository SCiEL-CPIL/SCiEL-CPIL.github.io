import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial guess: Hexagonal packing or Grid
    # Let's try a grid first, then optimize
    # W, H
    W0 = 1.0
    H0 = 1.0
    
    # 5x5 grid, remove 4 corners? or center?
    # Seed did 5x5 grid of 0.5/side.
    # Let's start with a slightly perturbed grid to break symmetry.
    np.random.seed(42)
    
    # Variables: x (21), y (21), r (21), W (1)
    # Total 64 variables
    
    def objective(v):
        if time.time() > deadline:
            return 1e9
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = 2.0 - W
        
        # Penalty for boundary
        # x must be in [r, W-r]
        # y must be in [r, H-r]
        
        pen_boundary = 0.0
        # Use smooth max: softplus(x) ~ log(1+exp(x)) or just max(0, x)^2
        # For gradient, max(0, x)^2 is 0 if x<0, 2x if x>0.
        
        for i in range(N):
            # Left/Right
            if x[i] < r[i]:
                pen_boundary += (r[i] - x[i])**2
            if x[i] > W - r[i]:
                pen_boundary += (x[i] - (W - r[i]))**2
            # Top/Bottom
            if y[i] < r[i]:
                pen_boundary += (r[i] - y[i])**2
            if y[i] > H - r[i]:
                pen_boundary += (y[i] - (H - r[i]))**2
        
        # Penalty for overlap
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    pen_overlap += (min_dist - dist)**2
        
        # Objective: Max sum(r) -> Min -sum(r) + Penalty
        return -np.sum(r) + 100.0 * (pen_boundary + pen_overlap)

    def gradient(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = 2.0 - W
        
        grad = np.zeros_like(v)
        
        # Gradient of -sum(r)
        grad[2*N:3*N] -= 1.0
        
        # Gradient of boundary penalty
        # d/dx_i (r_i - x_i)^2 = -2(r_i - x_i) if x_i < r_i
        # d/dx_i (x_i - (W-r_i))^2 = 2(x_i - (W-r_i)) if x_i > W-r_i
        # Note: W also affects boundary.
        
        # We'll accumulate gradients
        # This is getting complex to do analytically with the if-statements in a vectorized way without care.
        # Let's use finite differences for gradient? 
        # N=64. 64 function evaluations per gradient step.
        # L-BFGS-B might take 100-200 iterations. 200 * 64 * Cost(func).
        # Cost(func) is O(N^2) = 441 ops. Very fast.
        # So finite difference gradient is likely acceptable and much less error-prone.
        pass

    # Let's just use finite difference gradient via scipy's '2-point' or similar if not provided.
    # Or provide a simple numerical gradient.
    
    def grad_fd(v):
        eps = 1e-8
        g = np.zeros_like(v)
        for i in range(len(v)):
            v_plus = v.copy()
            v_minus = v.copy()
            v_plus[i] += eps
            v_minus[i] -= eps
            g[i] = (objective(v_plus) - objective(v_minus)) / (2*eps)
        return g

    # Initial state
    # 5x5 grid
    side = 5
    cells_x = []
    cells_y = []
    cells_r = []
    # Place 21 circles in 5x5 grid (25 spots). Remove 4 corners?
    # Or just use first 21.
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            # Center of cell
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            # Initial radius: 0.5/side - small margin
            r_init = 0.5 / side - 0.01
            cells_x.append(cx)
            cells_y.append(cy)
            cells_r.append(r_init)
            idx += 1
            
    # Add some randomness to break symmetry
    cells_x = np.array(cells_x) + np.random.uniform(-0.01, 0.01, N)
    cells_y = np.array(cells_y) + np.random.uniform(-0.01, 0.01, N)
    cells_r = np.array(cells_r)
    
    v0 = np.concatenate([cells_x, cells_y, cells_r, [1.0]])
    
    # Bounds
    # x, y in [0, 1.5] (since W max 1.5)
    # r in [0, 0.7] (max radius in 1x1 box is 0.5, but if W=1.5, H=0.5, r<=0.25. If W=1, r<=0.5)
    # Let's be safe: r in [0, 0.6]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.5)) # x
    for i in range(N):
        bounds.append((0.0, 1.5)) # y
    for i in range(N):
        bounds.append((0.01, 0.6)) # r
    bounds.append((0.5, 1.5)) # W
    
    best_v = v0
    best_obj = objective(v0)
    
    # Run multiple restarts or just one long run?
    # Let's run L-BFGS-B with finite difference gradient.
    
    try:
        res = minimize(objective, v0, method='L-BFGS-B', jac=grad_fd, bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
        if res.fun < best_obj:
            best_v = res.x
            best_obj = res.fun
    except Exception as e:
        pass

    # Extract best
    x = best_v[0:N]
    y = best_v[N:2*N]
    r = best_v[2*N:3*N]
    W = best_v[3*N]
    H = 2.0 - W
    
    # Post-processing: Ensure validity
    # Shrink radii if overlapping or out of bounds
    # We can iteratively reduce r_i by a small factor if any constraint is violated
    
    def is_valid(x, y, r, W, H):
        if W + H > 4.0 + 1e-9:
            return False
        for i in range(N):
            if x[i] < r[i] - 1e-9 or x[i] > W - r[i] + 1e-9:
                return False
            if y[i] < r[i] - 1e-9 or y[i] > H - r[i] + 1e-9:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True

    # If not valid, shrink radii
    if not is_valid(x, y, r, W, H):
        # Simple shrink
        for _ in range(100):
            if is_valid(x, y, r, W, H):
                break
            r = r * 0.99
        # If still invalid, force valid by reducing more
        if not is_valid(x, y, r, W, H):
            # Fallback to seed-like grid if optimization failed badly
            side = 5
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01) for i in range(side) for j in range(side)]
            return np.array(cells[:N])

    return np.column_stack((x, y, r))
