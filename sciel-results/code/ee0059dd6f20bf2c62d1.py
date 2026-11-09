import numpy as np
import time
import sys

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N  # tube width = 0.125
    start_time = time.time()
    deadline = 600.0
    
    # Precompute grids
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    def compute_volume_fast(xk, yk, num_samples=50000):
        """Fast volume estimator for local search. xk, yk are shape (64,)."""
        # Compute bounding box
        # x_k + c*i/N, max over c in [0,1] is x_k + i/N
        X0 = np.min(xk)
        X1 = np.max(xk + i_flat / N)
        Y0 = np.min(yk)
        Y1 = np.max(yk + j_flat / N)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return float('inf')
        
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, num_samples)
        b = rng.uniform(Y0, Y1, num_samples)
        c = rng.uniform(0, 1, num_samples)
        
        # Vectorized check: for each sample, check if in any tube
        # x_left[k] = xk[k] + c * i_flat[k] / N
        # y_left[k] = yk[k] + c * j_flat[k] / N
        c_col = c[:, np.newaxis]  # (M, 1)
        x_left = xk[np.newaxis, :] + c_col * (i_flat / N)[np.newaxis, :]  # (M, 64)
        y_left = yk[np.newaxis, :] + c_col * (j_flat / N)[np.newaxis, :]  # (M, 64)
        
        in_x = (a[:, np.newaxis] >= x_left) & (a[:, np.newaxis] <= x_left + w)
        in_y = (b[:, np.newaxis] >= y_left) & (b[:, np.newaxis] <= y_left + w)
        in_any = np.any(in_x & in_y, axis=1)
        
        fraction = np.mean(in_any)
        volume = fraction * area
        return volume
    
    def compute_volume_full(xk, yk, num_samples=500000):
        """Full precision volume estimator for final evaluation."""
        X0 = np.min(xk)
        X1 = np.max(xk + i_flat / N)
        Y0 = np.min(yk)
        Y1 = np.max(yk + j_flat / N)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return float('inf')
        
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, num_samples)
        b = rng.uniform(Y0, Y1, num_samples)
        c = rng.uniform(0, 1, num_samples)
        
        c_col = c[:, np.newaxis]
        x_left = xk[np.newaxis, :] + c_col * (i_flat / N)[np.newaxis, :]
        y_left = yk[np.newaxis, :] + c_col * (j_flat / N)[np.newaxis, :]
        
        in_x = (a[:, np.newaxis] >= x_left) & (a[:, np.newaxis] <= x_left + w)
        in_y = (b[:, np.newaxis] >= y_left) & (b[:, np.newaxis] <= y_left + w)
        in_any = np.any(in_x & in_y, axis=1)
        
        fraction = np.mean(in_any)
        volume = fraction * area
        return volume
    
    # Initial configuration: symmetric "fan" construction
    # x_k depends only on i, y_k depends only on j
    # Try to make tubes converge as c increases
    # x_i = -i * alpha, y_j = -j * alpha
    # Optimal alpha to be found
    
    def initial_config(alpha=0.0625):
        xk = -i_flat * alpha
        yk = -j_flat * alpha
        return xk, yk
    
    # Start with a reasonable initial guess
    xk, yk = initial_config(0.0625)
    best_xk, best_yk = xk.copy(), yk.copy()
    best_vol = compute_volume_fast(xk, yk, 50000)
    
    # Coordinate descent local search
    # For each tube k, try to optimize its (x, y) position
    # Use a simple step-size schedule
    
    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() - start_time < deadline - 30:  # Leave 30s margin
        improved = False
        for k in range(64):
            if time.time() - start_time > deadline - 30:
                break
            
            # Try small perturbations in x and y
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    xk_new = xk.copy()
                    yk_new = yk.copy()
                    xk_new[k] += dx
                    yk_new[k] += dy
                    
                    vol = compute_volume_fast(xk_new, yk_new, 50000)
                    if vol < best_vol - 1e-10:
                        xk, yk = xk_new, yk_new
                        best_vol = vol
                        improved = True
                        no_improve_count = 0
                        break
                if improved:
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
        
        if not improved:
            # Reduce step size
            step_size *= 0.7
            no_improve_count = 0
            if step_size < 1e-6:
                break
    
    # Final evaluation with full samples
    final_vol = compute_volume_full(best_xk, best_yk, 500000)
    
    # Reshape to (64, 2)
    positions = np.stack([best_xk, best_yk], axis=1)
    return positions
