import numpy as np
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
def construct():
    # 1. Initialization: Hexagonal Grid
    # Estimate spacing to fit 200 points in unit square
    # Hex area per point ~ 0.866 * d^2
    # 200 * 0.866 * d^2 ~ 1 => d ~ 0.076
    d = 0.075
    
    # Generate hex grid
    # Rows are horizontal.
    # Vertical spacing between rows: d * sqrt(3) / 2
    # Horizontal spacing within row: d
    # Odd rows shifted by d/2
    
    n_rows = 16
    n_cols = 13
    # Total points: 208. We will select 200.
    
    x_coords = np.linspace(-0.5, 0.5, n_cols)
    y_coords = np.linspace(-0.5, 0.5, n_rows)
    
    pts = []
    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            if i % 2 == 1:
                x = x + (x_coords[1] - x_coords[0]) / 2.0
            pts.append([x, y])
            
    pts = np.array(pts)
    
    # Select 200 points closest to the center
    dists_to_center = np.sum(pts**2, axis=1)
    idx = np.argsort(dists_to_center)[:N]
    pts = pts[idx]
    
    # 2. Optimization using L-BFGS-B with analytical gradient
    x0 = pts.flatten()
    
    def objective_and_grad(x):
        p = x.reshape((N, 2))
        
        # Pairwise distances
        # diff[i, j] = p[i] - p[j]
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        dists_sq_safe = np.where(dists_sq < 1e-12, 1e-12, dists_sq)
        
        # Energy: sum 1/d^2
        # E_pair = sum_{i<j} 1/d_ij^2
        # Note: sum_{i!=j} 1/d_ij^2 = 2 * E_pair
        # So E_pair = 0.5 * sum_{i!=j} 1/d_ij^2
        
        inv_dists_sq = 1.0 / dists_sq_safe
        np.fill_diagonal(inv_dists_sq, 0.0)
        
        E_pair = 0.5 * np.sum(inv_dists_sq)
        
        # Gradient of E_pair w.r.t p_i
        # d(E_pair)/dp_i = sum_{j!=i} d/dp_i (0.5 * (p_i - p_j)^2)^(-1)
        # = sum_{j!=i} 0.5 * (-1) * (dists_sq_ij)^(-2) * 2 * (p_i - p_j)
        # = - sum_{j!=i} (p_i - p_j) / (dists_sq_ij^2)
        
        # We have inv_dists_sq = 1/dists_sq
        # We need 1/dists_sq^2
        inv_dists_sq_2 = inv_dists_sq ** 2
        
        # grad_pair[i] = - sum_j (p_i - p_j) * inv_dists_sq_2[i, j]
        # diff[i, j] = p_i - p_j
        grad_pair = -np.sum(diff * inv_dists_sq_2[:, :, np.newaxis], axis=1)
        
        # Boundary Energy
        # E_boundary = lambda * sum_i (1/dx_i^2 + 1/dy_i^2)
        lambda_b = 10.0
        
        dx = 0.5 - np.abs(p[:, 0])
        dy = 0.5 - np.abs(p[:, 1])
        
        dx_safe = np.where(dx < 1e-12, 1e-12, dx)
        dy_safe = np.where(dy < 1e-12, 1e-12, dy)
        
        E_boundary = lambda_b * np.sum(1.0 / dx_safe**2 + 1.0 / dy_safe**2)
        
        # Gradient of Boundary Energy
        # d/dx_i (1/dx_i^2) = -2 * dx_i^(-3) * d/dx_i(dx_i)
        # dx_i = 0.5 - |x_i|
        # if x_i > 0, d/dx_i = -1 => grad = 2 * dx_i^(-3)
        # if x_i < 0, d/dx_i = 1 => grad = -2 * dx_i^(-3)
        # grad = -2 * sign(x_i) * dx_i^(-3) ?
        # Let's check: x_i > 0, sign=1. -2 * 1 * dx^-3 = -2 dx^-3.
        # But we calculated +2 dx^-3 above.
        # Let's re-verify.
        # f(x) = (0.5 - x)^-2 for x > 0.
        # f'(x) = -2 (0.5 - x)^-3 * (-1) = 2 (0.5 - x)^-3.
        # So for x > 0, grad is positive.
        # For x < 0, f(x) = (0.5 + x)^-2.
        # f'(x) = -2 (0.5 + x)^-3 * (1) = -2 (0.5 + x)^-3.
        # So for x < 0, grad is negative.
        # So grad = 2 * sign(x) * dx^-3.
        
        sign_x = np.sign(p[:, 0])
        sign_y = np.sign(p[:, 1])
        
        grad_bx = lambda_b * 2.0 * sign_x / (dx_safe ** 3)
        grad_by = lambda_b * 2.0 * sign_y / (dy_safe ** 3)
        
        # Handle x=0 (sign is 0, but derivative is undefined/discontinuous). 
        # In practice, points won't be exactly 0. 
        # If they are, we can set grad to 0 or small value.
        # L-BFGS-B handles bounds, so points shouldn't cross 0 often in a way that breaks this.
        
        E_total = E_pair + E_boundary
        
        grad_total = np.zeros_like(p)
        grad_total[:, 0] = grad_pair[:, 0] + grad_bx
        grad_total[:, 1] = grad_pair[:, 1] + grad_by
        
        return E_total, grad_total.flatten()

    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Use L-BFGS-B
    # maxiter 1000 should be enough
    res = minimize(objective_and_grad, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    p_opt = res.x.reshape((N, 2))
    
    # Final cleanup: ensure inside bounds
    p_opt[:, 0] = np.clip(p_opt[:, 0], -0.5, 0.5)
    p_opt[:, 1] = np.clip(p_opt[:, 1], -0.5, 0.5)
    
    return p_opt
# EVOLVE-BLOCK-END
