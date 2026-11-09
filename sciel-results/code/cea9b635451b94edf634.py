import numpy as np
import time
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    # 1. Initialization: Hexagonal Lattice
    # Approximate number of rows and cols
    # Hex packing density is higher. 
    # Side length 1.
    # Let's generate a hex grid slightly larger than the square and clip.
    
    # Generate hex grid points
    # Spacing dx = 1, dy = sqrt(3)/2
    # We want ~200 points.
    # Area = 1. Density ~ 1/sqrt(3) * 2/pi? No.
    # Let's just make a grid with spacing 1/sqrt(N) roughly, but hex.
    
    # Simple hex grid generation
    def hex_grid(n_rows, n_cols, dx, dy, offset_x=0, offset_y=0):
        pts = []
        for i in range(n_rows):
            for j in range(n_cols):
                x = j * dx + (i % 2) * dx / 2.0 + offset_x
                y = i * dy + offset_y
                pts.append([x, y])
        return np.array(pts)

    # Estimate grid size
    # For N=200, sqrt(N) ~ 14.
    # Hex grid with 15x15 = 225 points.
    # dx = 1/15, dy = sqrt(3)/2 * dx
    # Total width = 15 * dx = 1.
    # Total height = 15 * dy = 15 * sqrt(3)/2 * 1/15 = sqrt(3)/2 ~ 0.866.
    # This fits in the square [-0.5, 0.5] x [-0.5, 0.5].
    
    n_rows = 15
    n_cols = 15
    dx = 1.0 / n_cols
    dy = np.sqrt(3) / 2.0 * dx
    
    # Center the grid
    total_w = (n_cols - 1) * dx + dx/2.0 # max x extent
    total_h = (n_rows - 1) * dy
    offset_x = -total_w / 2.0
    offset_y = -total_h / 2.0
    
    pts = hex_grid(n_rows, n_cols, dx, dy, offset_x, offset_y)
    
    # We have 225 points. We need 200.
    # Randomly select 200? Or take the first 200?
    # Taking first 200 creates a hole.
    # Better: Generate a larger grid, keep those inside the square, then sample 200?
    # Or just generate 200 points by taking a dense hex grid and subsampling?
    # Let's generate a dense hex grid covering the square, keep all inside, 
    # and if >200, randomly select 200 to avoid bias.
    # If <200, add random points?
    # Actually, 15x15 hex is 225. The square is 1x1.
    # The hex grid of 15x15 with dx=1/15 has width ~1.
    # Let's check bounds.
    # x ranges from -0.5 to 0.5 approx.
    # y ranges from -0.43 to 0.43.
    # So all 225 points are inside.
    # We need to remove 25 points.
    # Removing points randomly might break symmetry.
    # Let's just use the first 200? No, that leaves a rectangular hole.
    # Let's try to generate exactly 200?
    # Maybe 14x14 square grid is 196. Add 4?
    # Let's stick to the hex grid and randomly pick 200 indices.
    # To be deterministic, use a fixed seed.
    
    rng = np.random.RandomState(42)
    if len(pts) > N:
        idx = rng.choice(len(pts), N, replace=False)
        pts = pts[idx]
    elif len(pts) < N:
        # Pad with random points inside square
        extra = np.random.uniform(-0.49, 0.49, (N - len(pts), 2))
        pts = np.vstack([pts, extra])
        
    # Ensure pts is float64
    pts = pts.astype(np.float64)
    
    # 2. Optimization
    # Objective: Minimize Sum of inverse powers of distances
    # J(x) = sum_{i<j} 1/d_ij^p + sum_i (1/w_ix^p + 1/w_+x^p + 1/w_iy^p + 1/w_+y^p)
    
    p = 4.0 # Exponent
    eps = 1e-6
    
    def objective(x_flat):
        x = x_flat.reshape(-1, 2)
        # Pairwise distances
        # Efficient calculation
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dist_sq = dist_sq[mask]
        
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, eps**2)
        
        # Sum of inverse powers
        # d^{-p} = (d^2)^{-p/2}
        pair_energy = np.sum(dist_sq ** (-p/2.0))
        
        # Wall energy
        # x in [-0.5, 0.5]
        wx_neg = 0.5 - x[:, 0]
        wx_pos = 0.5 + x[:, 0]
        wy_neg = 0.5 - x[:, 1]
        wy_pos = 0.5 + x[:, 1]
        
        wx_neg = np.maximum(wx_neg, eps)
        wx_pos = np.maximum(wx_pos, eps)
        wy_neg = np.maximum(wy_neg, eps)
        wy_pos = np.maximum(wy_pos, eps)
        
        wall_energy = np.sum(wx_neg ** (-p) + wx_pos ** (-p) + wy_neg ** (-p) + wy_pos ** (-p))
        
        return pair_energy + wall_energy

    def gradient(x_flat):
        x = x_flat.reshape(-1, 2)
        grad = np.zeros_like(x)
        
        # Pairwise gradient
        # d/dx_i sum_{j != i} (||x_i - x_j||^2)^{-p/2}
        # = sum_{j != i} -p/2 * (||x_i - x_j||^2)^{-p/2 - 1} * 2 * (x_i - x_j)
        # = -p * sum_{j != i} (x_i - x_j) * ||x_i - x_j||^{-2p - 2}
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, eps**2)
        
        # Factor for each pair (i,j)
        # We need to sum over j for each i.
        # grad_i = -p * sum_j diff_ij * (dist_sq_ij)^{-p/2 - 1}
        
        # dist_sq is N x N
        # We want to compute sum over axis 1 (j)
        # term = -p * diff * (dist_sq[..., np.newaxis]) ** (-p/2.0 - 1.0)
        
        # Set diagonal to 0 to avoid self-interaction
        mask = ~np.eye(N, dtype=bool)
        dist_sq_masked = np.where(mask, dist_sq, 1.0) # Use 1.0 so 1^... is 1, but we multiply by diff which is 0 on diag
        # Actually, diff is 0 on diagonal, so we can just compute and it will be 0.
        # But 0^negative is inf. So we must mask dist_sq before power.
        
        # Safe dist_sq for power
        safe_dist_sq = np.maximum(dist_sq, eps**2)
        
        # Gradient contribution from pairs
        # g_ij = -p * (x_i - x_j) * (d_ij^2)^{-p/2 - 1}
        # Note: (d^2)^{-p/2 - 1} = d^{-p - 2}
        
        factor = safe_dist_sq ** (-p/2.0 - 1.0)
        # factor is N x N
        # g_i = -p * sum_j (x_i - x_j) * factor_ij
        
        # diff is (x_i - x_j)
        # We sum over j (axis 1)
        pair_grad = -p * np.sum(diff * factor[:, :, np.newaxis], axis=1)
        
        # Wall gradient
        # d/dx ( (0.5-x)^{-p} ) = -p * (0.5-x)^{-p-1} * (-1) = p * (0.5-x)^{-p-1}
        # d/dx ( (0.5+x)^{-p} ) = -p * (0.5+x)^{-p+1} * (1) = -p * (0.5+x)^{-p-1}
        
        wx_neg = np.maximum(0.5 - x[:, 0], eps)
        wx_pos = np.maximum(0.5 + x[:, 0], eps)
        wy_neg = np.maximum(0.5 - x[:, 1], eps)
        wy_pos = np.maximum(0.5 + x[:, 1], eps)
        
        grad[:, 0] += pair_grad[:, 0] + p * (wx_neg ** (-p - 1.0)) - p * (wx_pos ** (-p - 1.0))
        grad[:, 1] += pair_grad[:, 1] + p * (wy_neg ** (-p - 1.0)) - p * (wy_pos ** (-p - 1.0))
        
        return grad.flatten()

    # Optimize
    best_x = pts.flatten()
    best_f = objective(best_x)
    
    # L-BFGS-B
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # We might run multiple restarts or just one long run?
    # Given time limit, one long run with L-BFGS-B is good.
    # L-BFGS-B can handle 400 variables easily.
    
    try:
        res = minimize(
            objective, 
            best_x, 
            jac=gradient, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.success or res.fun < best_f:
            best_x = res.x
            best_f = res.fun
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > timeout:
        # Return best found so far
        pass
    
    # Final cleanup: ensure points are strictly inside
    x_final = best_x.reshape(-1, 2)
    # Clip to [-0.5, 0.5] with small margin to avoid floating point issues on boundary
    # The score calculation uses 1/2 - |c_ik|. If c_ik = 0.5, dist=0.
    # So we should keep them strictly inside.
    x_final = np.clip(x_final, -0.499999, 0.499999)
    
    return x_final
