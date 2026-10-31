import numpy as np
import time
from scipy.optimize import minimize

N = 26
TIME_LIMIT = 100  # Safety margin before 120s limit

def construct():
    start_time = time.time()
    
    # Initial best result
    best_score = -np.inf
    best_solution = None
    
    # Penalty weights
    # These need to be large enough to force constraints, but not infinite
    lambda_overlap = 50.0
    lambda_boundary = 50.0
    
    def unpack(z):
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        return x, y, r

    def objective(z):
        x, y, r = unpack(z)
        
        # Sum of radii (to maximize, so we minimize negative)
        sum_r = np.sum(r)
        
        # Boundary penalties
        # x_i in [r_i, 1-r_i], y_i in [r_i, 1-r_i]
        # Violations:
        # r_i - x_i > 0
        # r_i - y_i > 0
        # x_i - (1 - r_i) > 0
        # y_i - (1 - r_i) > 0
        
        v_x_left = np.maximum(0, r - x)
        v_y_left = np.maximum(0, r - y)
        v_x_right = np.maximum(0, x - (1 - r))
        v_y_right = np.maximum(0, y - (1 - r))
        
        pen_boundary = np.sum(v_x_left**2) + np.sum(v_y_left**2) + np.sum(v_x_right**2) + np.sum(v_y_right**2)
        
        # Overlap penalties
        # d_ij >= r_i + r_j
        # Violation: r_i + r_j - d_ij > 0
        
        # Compute pairwise distances
        # Using broadcasting
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        # r_i + r_j matrix
        r_sum = r[:, None] + r[None, :]
        
        # Violations
        viol = r_sum - dist
        # Only consider i < j to avoid double counting and self-overlap (diagonal is 0)
        # Set diagonal to 0
        np.fill_diagonal(viol, 0)
        
        # Take upper triangle
        iu = np.triu_indices(N, 1)
        viol_upper = viol[iu]
        
        pos_viol = np.maximum(0, viol_upper)
        pen_overlap = np.sum(pos_viol**2)
        
        total_penalty = lambda_boundary * pen_boundary + lambda_overlap * pen_overlap
        
        return -sum_r + total_penalty

    def gradient(z):
        x, y, r = unpack(z)
        
        grad = np.zeros(3*N)
        
        # Gradient of -sum(r)
        grad[2*N:] = -1.0
        
        # Boundary gradient
        # d/dx_i (max(0, r_i - x_i)^2) = -2 * max(0, r_i - x_i)
        # d/dx_i (max(0, x_i - (1 - r_i))^2) = 2 * max(0, x_i - (1 - r_i))
        # d/dr_i (max(0, r_i - x_i)^2) = 2 * max(0, r_i - x_i)
        # d/dr_i (max(0, x_i - (1 - r_i))^2) = 2 * max(0, x_i - (1 - r_i)) * (-1) ? No, d/dr (x - 1 + r) = 1
        # Wait, let's be precise.
        # f(r, x) = max(0, r - x)^2. df/dr = 2 max(0, r-x). df/dx = -2 max(0, r-x).
        # g(r, x) = max(0, x - (1-r))^2 = max(0, x + r - 1)^2. df/dr = 2 max(0, x+r-1). df/dx = 2 max(0, x+r-1).
        
        v_x_left = np.maximum(0, r - x)
        v_y_left = np.maximum(0, r - y)
        v_x_right = np.maximum(0, x - (1 - r))
        v_y_right = np.maximum(0, y - (1 - r))
        
        # Grads for x
        grad_x = -2 * lambda_boundary * v_x_left + 2 * lambda_boundary * v_x_right
        # Grads for y
        grad_y = -2 * lambda_boundary * v_y_left + 2 * lambda_boundary * v_y_right
        # Grads for r
        # From v_x_left: 2 * v_x_left
        # From v_y_left: 2 * v_y_left
        # From v_x_right: 2 * v_x_right
        # From v_y_right: 2 * v_y_right
        grad_r_boundary = 2 * lambda_boundary * (v_x_left + v_y_left + v_x_right + v_y_right)
        
        # Overlap gradient
        # P = sum_{i<j} max(0, r_i + r_j - d_ij)^2
        # Let v_ij = r_i + r_j - d_ij
        # dP/dr_i = sum_{j != i} 2 * max(0, v_ij) * (1 - 0) ? No, d/dr_i (r_i + r_j - d) = 1
        # dP/dx_i = sum_{j != i} 2 * max(0, v_ij) * (- dx_ij/dx_i)
        # dx_ij/dx_i = (x_i - x_j)/d_ij
        # So dP/dx_i = sum_{j != i} -2 * max(0, v_ij) * (x_i - x_j)/d_ij
        # dP/dy_i = sum_{j != i} -2 * max(0, v_ij) * (y_i - y_j)/d_ij
        
        # Compute full matrix terms
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Avoid division by zero
        dist_safe = np.where(dist > 1e-8, dist, 1.0)
        
        r_sum = r[:, None] + r[None, :]
        viol = r_sum - dist
        np.fill_diagonal(viol, 0)
        
        pos_viol = np.maximum(0, viol)
        
        # Gradient of overlap penalty w.r.t x
        # term_ij = 2 * lambda_overlap * pos_viol_ij
        # dP/dx_i = - sum_{j != i} term_ij * (x_i - x_j) / dist_ij
        # Note: pos_viol is symmetric.
        
        # We can compute this using matrix multiplication
        # grad_x_overlap[i] = - sum_j 2 * lambda * pos_viol[i,j] * (x[i]-x[j])/dist[i,j]
        
        # Let W_ij = 2 * lambda_overlap * pos_viol[i,j] / dist[i,j]
        # Then dP/dx_i = - sum_j W_ij * (x_i - x_j) = - sum_j W_ij * x_i + sum_j W_ij * x_j
        # = - x_i * sum_j W_ij + sum_j (W_ij * x_j)
        
        # Construct W matrix
        W = 2.0 * lambda_overlap * pos_viol / dist_safe
        # Zero out diagonal
        np.fill_diagonal(W, 0)
        
        # Sum W along rows and columns
        W_row_sum = np.sum(W, axis=1)
        W_col_sum = np.sum(W, axis=0) # Same as row sum since symmetric
        
        # dP/dx_i = - W_row_sum[i] * x[i] + np.sum(W[i, :] * x)
        # This is (W @ x) - x * (W.sum(axis=1))
        
        grad_x_overlap = -(W @ x) + x * W_row_sum
        grad_y_overlap = -(W @ y) + y * W_row_sum
        
        # dP/dr_i = sum_{j != i} 2 * lambda_overlap * pos_viol[i,j] * 1
        grad_r_overlap = 2.0 * lambda_overlap * np.sum(pos_viol, axis=1)
        
        # Combine gradients
        grad[:N] += grad_x + grad_x_overlap
        grad[N:2*N] += grad_y + grad_y_overlap
        grad[2*N:] += grad_r_boundary + grad_r_overlap
        
        return grad

    def is_valid(x, y, r, tol=1e-4):
        # Check boundaries
        if np.any(x < r - tol) or np.any(y < r - tol):
            return False
        if np.any(x > 1 - r + tol) or np.any(y > 1 - r + tol):
            return False
        if np.any(r < 0):
            return False
        
        # Check overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        # d_ij >= r_i + r_j - tol
        if np.any(dist < r_sum - tol):
            return False
        return True

    def get_score(x, y, r):
        return np.sum(r)

    # Initializations
    # 1. Hexagonal packing
    # 2. Random
    
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        if i == 0:
            # Hexagonal init
            # Generate a grid of points
            # Hexagonal lattice vectors
            # a = 1, b = 1/sqrt(3) ?
            # Let's just use a dense grid and pick closest to hex
            # Or simpler: place circles in rows
            # Row 0: y=0.5, x spaced
            # Row 1: y=0.5+delta, x shifted
            # For N=26, roughly 5x5 grid
            # Let's use a simple grid init for first try, then random
            x0 = np.linspace(0.1, 0.9, 5)
            y0 = np.linspace(0.1, 0.9, 5)
            xx, yy = np.meshgrid(x0, y0)
            x_flat = xx.flatten()
            y_flat = yy.flatten()
            # Shuffle and take N
            indices = np.random.choice(len(x_flat), N, replace=False)
            x_init = x_flat[indices]
            y_init = y_flat[indices]
            r_init = np.full(N, 0.1)
        else:
            # Random init
            x_init = np.random.uniform(0.1, 0.9, N)
            y_init = np.random.uniform(0.1, 0.9, N)
            r_init = np.full(N, 0.05)
            
        z0 = np.concatenate([x_init, y_init, r_init])
        
        # Bounds
        # x, y in [0, 1]
        # r in [0, 0.5]
        bounds = [(0, 1)] * (2*N) + [(0, 0.5)] * N
        
        try:
            res = minimize(objective, z0, jac=gradient, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success:
                z_opt = res.x
                x_opt, y_opt, r_opt = unpack(z_opt)
                
                if is_valid(x_opt, y_opt, r_opt):
                    score = get_score(x_opt, y_opt, r_opt)
                    if score > best_score:
                        best_score = score
                        best_solution = np.column_stack((x_opt, y_opt, r_opt))
                        
        except Exception as e:
            continue

    if best_solution is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_solution

