import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _objective(x):
    """
    x: [x_1, y_1, r_1, ..., x_N, y_N, r_N, W]
    Maximize sum of radii => Minimize -sum of radii
    """
    # radii are at indices 2, 5, 8, ...
    # r_i is at index 3*i + 2
    radii = x[2::3]
    return -np.sum(radii)

def _objective_grad(x):
    grad = np.zeros_like(x)
    # grad for r_i is -1
    grad[2::3] = -1.0
    return grad

def _constraints(x):
    """
    Returns list of constraint dicts.
    """
    # Parse x
    x_coords = x[0::3]
    y_coords = x[1::3]
    r_coords = x[2::3]
    W = x[-1]
    H = 2.0 - W  # Perimeter 4 => 2(W+H)=4 => W+H=2
    
    constraints = []
    
    # Wall constraints
    # x_i - r_i >= 0
    # x_i + r_i <= W => -(x_i + r_i) + W >= 0
    # y_i - r_i >= 0
    # y_i + r_i <= H => -(y_i + r_i) + H >= 0
    
    for i in range(N):
        # x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2], 'jac': lambda x, i=i: _wall_jac(x, i, 'x_minus')})
        # -x - r + W >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: -x[3*i] - x[3*i+2] + x[-1], 'jac': lambda x, i=i: _wall_jac(x, i, 'x_plus')})
        # y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2], 'jac': lambda x, i=i: _wall_jac(x, i, 'y_minus')})
        # -y - r + H >= 0 => -y - r + (2-W) >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: -x[3*i+1] - x[3*i+2] + (2.0 - x[-1]), 'jac': lambda x, i=i: _wall_jac(x, i, 'y_plus')})
        
    # Collision constraints
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: _collision_fun(x, i, j), 'jac': lambda x, i=i, j=j: _collision_jac(x, i, j)})
            
    return constraints

def _wall_jac(x, i, typ):
    grad = np.zeros_like(x)
    idx_x = 3*i
    idx_y = 3*i+1
    idx_r = 3*i+2
    idx_W = -1
    
    if typ == 'x_minus':
        grad[idx_x] = 1.0
        grad[idx_r] = -1.0
    elif typ == 'x_plus':
        grad[idx_x] = -1.0
        grad[idx_r] = -1.0
        grad[idx_W] = 1.0
    elif typ == 'y_minus':
        grad[idx_y] = 1.0
        grad[idx_r] = -1.0
    elif typ == 'y_plus':
        grad[idx_y] = -1.0
        grad[idx_r] = -1.0
        grad[idx_W] = -1.0
    return grad

def _collision_fun(x, i, j):
    xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
    xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
    return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2

def _collision_jac(x, i, j):
    grad = np.zeros_like(x)
    xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
    xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
    
    dx = xi - xj
    dy = yi - yj
    
    grad[3*i] = 2 * dx
    grad[3*i+1] = 2 * dy
    grad[3*i+2] = -2 * (ri + rj)
    
    grad[3*j] = -2 * dx
    grad[3*j+1] = -2 * dy
    grad[3*j+2] = -2 * (ri + rj)
    
    return grad

def _pack_circle_set(W_init, H_init, rng):
    """
    Attempt to pack circles in a rectangle W x H.
    """
    # Initial positions
    # Grid layout
    cols = int(np.round(np.sqrt(N * W_init / H_init)))
    if cols < 1: cols = 1
    rows = int(np.ceil(N / cols))
    
    x0 = np.zeros(3*N + 1)
    x0[-1] = W_init
    
    for i in range(N):
        col = i % cols
        row = i // cols
        
        dx = W_init / cols
        dy = H_init / rows
        
        x_pos = (col + 0.5) * dx
        y_pos = (row + 0.5) * dy
        
        # Add some random jitter to escape symmetry
        x_pos += rng.uniform(-0.1, 0.1) * dx
        y_pos += rng.uniform(-0.1, 0.1) * dy
        
        # Clamp to be strictly inside
        x_pos = np.clip(x_pos, 0.01, W_init - 0.01)
        y_pos = np.clip(y_pos, 0.01, H_init - 0.01)
        
        # Small initial radius
        r_init = 0.05
        
        x0[3*i] = x_pos
        x0[3*i+1] = y_pos
        x0[3*i+2] = r_init

    # Define bounds
    # x_i in [0, W], y_i in [0, H], r_i in [0, min(W,H)/2]
    # W in [0.5, 1.5] (since W+H=2, W cannot be 0 or 2 realistically for packing)
    
    # Note: Bounds on x, y depend on W.
    # trust-constr supports bounds.
    # But W is a variable.
    # We can set loose bounds and rely on constraints.
    # x_i >= 0, x_i <= W. Since W is variable, we can't fix x_i bounds easily.
    # However, the constraints x_i - r_i >= 0 and x_i + r_i <= W handle this.
    # We just need to ensure x_i, y_i, r_i don't go to infinity.
    # Let's set bounds:
    # x_i in [0, 2], y_i in [0, 2], r_i in [0, 1]
    # W in [0.1, 1.9]
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0)) # x
        bounds.append((0.0, 2.0)) # y
        bounds.append((0.0, 1.0)) # r
    bounds.append((0.1, 1.9)) # W

    try:
        res = minimize(
            _objective,
            x0,
            method='trust-constr',
            jac=_objective_grad,
            hess=None,
            constraints=_constraints(x0),
            bounds=bounds,
            options={'maxiter': 500, 'gtol': 1e-5, 'xtol': 1e-5, 'verbose': 0}
        )
        
        if res.success or res.status == 0:
            # Extract result
            x_res = res.x
            W_res = x_res[-1]
            H_res = 2.0 - W_res
            
            # Check validity
            valid = True
            x_coords = x_res[0::3]
            y_coords = x_res[1::3]
            r_coords = x_res[2::3]
            
            if W_res <= 0 or H_res <= 0:
                return None, 0.0
                
            # Check walls
            for i in range(N):
                if x_coords[i] - r_coords[i] < -1e-6 or x_coords[i] + r_coords[i] > W_res + 1e-6:
                    valid = False
                    break
                if y_coords[i] - r_coords[i] < -1e-6 or y_coords[i] + r_coords[i] > H_res + 1e-6:
                    valid = False
                    break
            if not valid:
                return None, 0.0
                
            # Check collisions
            for i in range(N):
                for j in range(i+1, N):
                    dist_sq = (x_coords[i] - x_coords[j])**2 + (y_coords[i] - y_coords[j])**2
                    rad_sum = r_coords[i] + r_coords[j]
                    if dist_sq < rad_sum**2 - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
                    
            if valid:
                # Create result array (x, y, r)
                result = np.column_stack([x_coords, y_coords, r_coords])
                score = np.sum(r_coords)
                return result, score
            else:
                return None, 0.0
        else:
            return None, 0.0
            
    except Exception:
        return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_score = -np.inf
    best_result = None
    
    rng = np.random.default_rng(42)
    
    # Try different aspect ratios
    # W + H = 2
    # W in [0.5, 1.5]
    aspect_ratios = np.linspace(0.5, 1.5, 10)
    
    # To use time effectively, we can run multiple restarts for the best aspect ratios
    # or just iterate through them.
    
    # Let's do a few random restarts for each aspect ratio
    restarts_per_ar = 3
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        W = ar
        H = 2.0 - ar
        
        for r in range(restarts_per_ar):
            if time.time() > deadline:
                break
                
            result, score = _pack_circle_set(W, H, rng)
            
            if result is not None and score > best_score:
                best_score = score
                best_result = result
                
    # Fallback: Simple grid if nothing worked
    if best_result is None:
        # Use a square 1x1
        W = 1.0
        H = 1.0
        cols = 5
        rows = 5
        cells = []
        for i in range(5):
            for j in range(5):
                if len(cells) < N:
                    x = (i + 0.5) / 5.0
                    y = (j + 0.5) / 5.0
                    r = 0.5 / 5.0 - 1e-3
                    cells.append([x, y, r])
        best_result = np.array(cells[:N])
        
    return best_result
