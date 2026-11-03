"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 145.0 # 5s safety margin

def check_validity(x, y, r):
    """Check if the configuration is valid (disjoint, inside square)."""
    # Check bounds
    if np.any(x < -1e-9) or np.any(x > 1 + 1e-9) or np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
        return False
    if np.any(r < -1e-9):
        return False
        
    # Check boundary containment
    # r_i <= x_i, r_i <= 1-x_i, etc.
    if np.any(r > x + 1e-7) or np.any(r > 1 - x + 1e-7) or np.any(r > y + 1e-7) or np.any(r > 1 - y + 1e-7):
        return False
        
    # Check pairwise distances
    # Vectorized check
    # dist_sq[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
    # req_sq[i, j] = (r_i + r_j)^2
    # Need dist_sq >= req_sq - tol
    
    x_diff = x[:, None] - x[None, :]
    y_diff = y[:, None] - y[None, :]
    dist_sq = x_diff**2 + y_diff**2
    
    r_sum = r[:, None] + r[None, :]
    req_sq = r_sum**2
    
    # We only care about i < j, but checking all is fine
    # dist_sq[i,j] >= req_sq[i,j]
    # violation = req_sq - dist_sq
    # if max(violation) > tol: invalid
    viol = req_sq - dist_sq
    if np.max(viol) > 1e-7:
        return False
        
    return True

def cost_and_grad(params):
    """
    Objective: Minimize -sum(r) + Penalty
    Penalty: Sum of squared violations of constraints
    """
    n = N
    x = params[0:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    # Objective part
    obj = -np.sum(r)
    grad_obj = np.zeros(3*n)
    grad_obj[2*n:3*n] = -1.0
    
    # Penalty part
    penalty = 0.0
    grad_pen = np.zeros(3*n)
    
    # 1. Boundary constraints
    # x - r >= 0  => viol = r - x
    # 1-x - r >= 0 => viol = r - (1-x) = r + x - 1
    # y - r >= 0
    # 1-y - r >= 0
    
    # Helper to add squared penalty for constraint val >= 0
    # If val < 0, penalty = (-val)^2, grad = -2*(-val)*grad_val
    # Actually, let v = -val (violation magnitude). If v > 0, pen = v^2.
    
    # Boundary X
    v1 = r - x
    mask = v1 > 0
    if np.any(mask):
        pen_val = np.sum(v1[mask]**2)
        penalty += pen_val
        # d(pen)/dx = 2*v * (-1) for masked elements
        # d(pen)/dr = 2*v * (1) for masked elements
        grad_pen[0:n][mask] -= 2 * v1[mask]
        grad_pen[2*n:3*n][mask] += 2 * v1[mask]
        
    v2 = r + x - 1
    mask = v2 > 0
    if np.any(mask):
        pen_val = np.sum(v2[mask]**2)
        penalty += pen_val
        # d(pen)/dx = 2*v * (1)
        # d(pen)/dr = 2*v * (1)
        grad_pen[0:n][mask] += 2 * v2[mask]
        grad_pen[2*n:3*n][mask] += 2 * v2[mask]
        
    # Boundary Y
    v3 = r - y
    mask = v3 > 0
    if np.any(mask):
        pen_val = np.sum(v3[mask]**2)
        penalty += pen_val
        grad_pen[n:2*n][mask] -= 2 * v3[mask]
        grad_pen[2*n:3*n][mask] += 2 * v3[mask]
        
    v4 = r + y - 1
    mask = v4 > 0
    if np.any(mask):
        pen_val = np.sum(v4[mask]**2)
        penalty += pen_val
        grad_pen[n:2*n][mask] += 2 * v4[mask]
        grad_pen[2*n:3*n][mask] += 2 * v4[mask]
        
    # 2. Pairwise constraints
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    # viol = (r_i + r_j)^2 - dist_sq
    # Use vectorized calculation for speed
    
    x_diff = x[:, None] - x[None, :]
    y_diff = y[:, None] - y[None, :]
    dist_sq = x_diff**2 + y_diff**2
    
    r_sum = r[:, None] + r[None, :]
    req_sq = r_sum**2
    
    viol = req_sq - dist_sq
    
    # Only consider upper triangle to avoid double counting?
    # The penalty sum should be consistent. If we sum over all i,j, we double count.
    # Let's sum over i < j.
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    v_pair = viol[idx_i, idx_j]
    mask = v_pair > 0
    
    if np.any(mask):
        # v_pair = (r_i+r_j)^2 - (dx^2+dy^2)
        # pen = sum(v_pair[mask]^2)
        # grad wrt x_i: d(v)/dx_i = 2(r_i+r_j)*0 - 2(x_i-x_j) = -2(x_i-x_j)
        # grad pen = 2*v * d(v)/dx
        
        v_m = v_pair[mask]
        i_m = idx_i[mask]
        j_m = idx_j[mask]
        
        dx_m = x_diff[i_m, j_m]
        dy_m = y_diff[i_m, j_m]
        
        # d(v)/dx_i = -2 * dx
        # d(v)/dx_j = 2 * dx
        # d(v)/dy_i = -2 * dy
        # d(v)/dy_j = 2 * dy
        # d(v)/dr_i = 2 * (r_i + r_j)
        # d(v)/dr_j = 2 * (r_i + r_j)
        
        r_sum_m = r_sum[i_m, j_m]
        
        # Accumulate gradients
        # Note: grad_pen is a flat array.
        # x indices: 0..n-1
        # y indices: n..2n-1
        # r indices: 2n..3n-1
        
        # For each pair (i, j) in mask:
        # contrib to x_i: 2 * v * (-2 * dx)
        # contrib to x_j: 2 * v * (2 * dx)
        # ...
        
        np.add.at(grad_pen, i_m, -4 * v_m * dx_m)
        np.add.at(grad_pen, j_m, 4 * v_m * dx_m)
        
        np.add.at(grad_pen, n + i_m, -4 * v_m * dy_m)
        np.add.at(grad_pen, n + j_m, 4 * v_m * dy_m)
        
        np.add.at(grad_pen, 2*n + i_m, 4 * v_m * r_sum_m)
        np.add.at(grad_pen, 2*n + j_m, 4 * v_m * r_sum_m)
        
        penalty += np.sum(v_m**2)
        
    return obj + penalty, grad_obj + grad_pen

def construct():
    global START_TIME
    start_time = time.time()
    
    # Define bounds for L-BFGS-B
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    
    best_score = -np.inf
    best_params = None
    
    # Initializations
    inits = []
    
    # 1. Grid
    side = int(np.ceil(np.sqrt(N)))
    grid_cells = []
    for i in range(side):
        for j in range(side):
            grid_cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01))
    grid_cells = grid_cells[:N]
    x_g = np.array([c[0] for c in grid_cells])
    y_g = np.array([c[1] for c in grid_cells])
    r_g = np.array([c[2] for c in grid_cells])
    inits.append(np.concatenate([x_g, y_g, r_g]))
    
    # 2. Hexagonal-ish
    # 5 cols, 5 rows approx
    # Let's just use random with some structure
    for k in range(5):
        # Random
        x_r = np.random.rand(N)
        y_r = np.random.rand(N)
        # Small radii to start, avoid immediate huge penalties
        r_r = np.random.rand(N) * 0.1 + 0.05 
        inits.append(np.concatenate([x_r, y_r, r_r]))
        
    # 3. "Spread out" random
    for k in range(5):
        x_r = np.random.rand(N)
        y_r = np.random.rand(N)
        # Sort and space them out? No, just random is fine for multi-start
        r_r = np.random.rand(N) * 0.1 + 0.05
        inits.append(np.concatenate([x_r, y_r, r_r]))
        
    # Add a few more varied randoms
    for k in range(10):
        x_r = np.random.rand(N)
        y_r = np.random.rand(N)
        r_r = np.random.rand(N) * 0.2 + 0.05
        inits.append(np.concatenate([x_r, y_r, r_r]))
        
    # Optimization Loop
    for init in inits:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        try:
            # Use L-BFGS-B
            # maxiter: limit iterations to prevent hanging on one start
            res = minimize(
                cost_and_grad, 
                init, 
                method='L-BFGS-B', 
                jac=True, 
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e-6: # If it converged or penalty is low
                params = res.x
                x, y, r = params[:N], params[N:2*N], params[2*N:3*N]
                
                # Check validity strictly
                if check_validity(x, y, r):
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        
        except Exception as e:
            pass

    if best_params is None:
        # Fallback to grid if nothing worked
        x_g = np.array([c[0] for c in grid_cells])
        y_g = np.array([c[1] for c in grid_cells])
        r_g = np.array([c[2] for c in grid_cells])
        best_params = np.concatenate([x_g, y_g, r_g])
        # Ensure validity of fallback
        # Grid might be invalid if radii too big? 
        # 0.5/side - 0.01 is small enough.
        
    x, y, r = best_params[:N], best_params[N:2*N], best_params[2*N:3*N]
    return np.column_stack((x, y, r))

# EVOLVE-BLOCK-END
