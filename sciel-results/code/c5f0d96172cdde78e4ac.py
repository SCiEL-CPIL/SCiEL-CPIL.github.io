import numpy as np
import scipy.optimize as opt
import time

N = 200

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin

    # 1. Hexagonal Initialization
    # Estimate radius
    # Area per circle approx 2r^2 * (2/sqrt(3)) for hex packing?
    # Area = 1. N circles. Area/circle = 1/N.
    # 2r^2 * 1.155 = 1/N => r = 0.5 * sqrt(1/(1.155*N))
    r_est = 0.5 * np.sqrt(1 / (1.155 * N))
    d = 2 * r_est
    
    # Create grid
    # Number of rows
    rows = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    cols = int(np.ceil(np.sqrt(N * np.sqrt(3) / 2)))
    
    points = []
    for i in range(rows):
        for j in range(cols):
            x = j * d
            y = i * d * np.sqrt(3) / 2
            if i % 2 == 1:
                x += d / 2
            points.append([x, y])
            
    points = np.array(points[:N])
    
    # Center and scale to fit in [-0.5, 0.5]
    # Current extent
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    
    # Scale to fit with margin
    # We want the bounding box to be slightly smaller than 1 to allow movement
    scale = 0.9 / max(max_x - min_x, max_y - min_y)
    points = points * scale
    points -= points.mean(axis=0)
    
    # Ensure within bounds
    points = np.clip(points, -0.49, 0.49)

    best_x = points.copy()
    best_score = -np.inf

    def compute_score(x):
        # x is (N, 2)
        diff = x[:, None, :] - x[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2
        
        wall_dist = np.minimum(0.5 - np.abs(x[:, 0]), 0.5 - np.abs(x[:, 1]))
        min_wall = np.min(wall_dist)
        
        return min(min_pair, min_wall)

    current_score = compute_score(best_x)
    best_score = current_score

    def objective(x_flat):
        x = x_flat.reshape(N, 2)
        # Check time
        if time.time() - start_time > time_limit:
            return 1e10  # Penalize heavily to stop? Or just return current value
            # Actually, minimize will stop. We should catch the timeout outside.
        
        k = current_k[0]
        
        # Pairwise distances
        diff = x[:, None, :] - x[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        
        # Wall distances
        wall_dist = np.minimum(0.5 - np.abs(x[:, 0]), 0.5 - np.abs(x[:, 1]))
        
        # Log-sum-exp
        # To avoid overflow, subtract max?
        # Terms are exp(-k * val). val >= 0. Max term is exp(0)=1.
        # So sum is <= N^2/2 + N. Log is safe.
        
        idx = np.triu_indices(N, 1)
        d_pairs = dist[idx]
        e_pairs = np.exp(-k * d_pairs / 2)
        e_walls = np.exp(-k * wall_dist)
        
        S = np.sum(e_pairs) + np.sum(e_walls)
        
        if S == 0:
            return 0.0, np.zeros_like(x_flat)
            
        f = np.log(S) / k
        
        # Gradient
        grad = np.zeros_like(x_flat)
        coeff = 1.0 / (k * S)
        
        # Pair gradients
        # grad_e = -k/2 * e_pairs / d_pairs
        # Handle d_pairs == 0
        safe_d = np.where(d_pairs < 1e-8, 1e-8, d_pairs)
        grad_e_pairs = -0.5 * k * e_pairs / safe_d
        
        # Vectorized update for pairs
        # For each pair (i, j), grad[i] += coeff * grad_e * (x[i]-x[j])
        # grad[j] -= coeff * grad_e * (x[i]-x[j])
        
        i_idx, j_idx = idx
        dx = x[i_idx, 0] - x[j_idx, 0]
        dy = x[i_idx, 1] - x[j_idx, 1]
        
        grad_vals = coeff * grad_e_pairs
        
        # Accumulate using np.add.at
        np.add.at(grad, i_idx * 2, grad_vals * dx)
        np.add.at(grad, i_idx * 2 + 1, grad_vals * dy)
        np.add.at(grad, j_idx * 2, -grad_vals * dx)
        np.add.at(grad, j_idx * 2 + 1, -grad_vals * dy)
        
        # Wall gradients
        grad_e_walls = -k * e_walls
        grad_walls = coeff * grad_e_walls
        
        limit_x = 0.5 - np.abs(x[:, 0])
        limit_y = 0.5 - np.abs(x[:, 1])
        is_x = limit_x < limit_y
        
        gx = np.where(is_x, -np.sign(x[:, 0]), 0)
        gy = np.where(~is_x, -np.sign(x[:, 1]), 0)
        
        grad[0::2] += grad_walls * gx
        grad[1::2] += grad_walls * gy
        
        return f, grad

    # Annealing loop
    k_vals = [5, 10, 20, 50, 100, 200, 500]
    current_k = [k_vals[0]]
    
    for k in k_vals:
        current_k[0] = k
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = opt.minimize(
                objective, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.x is not None:
                new_x = res.x.reshape(N, 2)
                # Check validity
                if np.all(np.isfinite(new_x)) and np.all(np.abs(new_x) <= 0.5):
                    score = compute_score(new_x)
                    if score > best_score:
                        best_score = score
                        best_x = new_x
        except Exception:
            pass
            
    # Final polish with Nelder-Mead if time permits?
    # Or just return best_x
    
    return best_x
