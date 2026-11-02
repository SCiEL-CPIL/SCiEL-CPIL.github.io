import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    # 1. Initialization
    # Hexagonal lattice
    def hex_init():
        # Estimate spacing
        s_est = np.sqrt(np.pi / (N * np.sqrt(3)/2))
        
        # Try a few spacings to ensure we get enough points
        for s_mult in [0.9, 0.95, 1.0, 1.05]:
            s = s_est * s_mult
            dx = s
            dy = s * np.sqrt(3) / 2
            
            # Grid
            max_extent = 1.0
            nx = int(2 * max_extent / dx) + 2
            ny = int(2 * max_extent / dy) + 2
            
            points = []
            for i in range(-nx, nx+1):
                for j in range(-ny, ny+1):
                    x = i * dx + (j % 2) * dx / 2
                    y = j * dy
                    if x*x + y*y <= 1.0:
                        points.append([x, y])
                        
            points = np.array(points)
            if len(points) >= N:
                # Random subset
                idx = np.random.choice(len(points), N, replace=False)
                return points[idx]
                
        # Fallback
        return np.random.randn(N, 2) * 0.8

    c = hex_init()
    
    # Precompute indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Optimization
    beta = 10.0
    lr = 0.05
    best_score = -1.0
    best_c = c.copy()
    
    num_steps = 300000
    
    for step in range(num_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Distances
        diff = c[i_idx] - c[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        
        # Boundary
        norms = np.linalg.norm(c, axis=1)
        bound_dists = 1.0 - norms
        
        # Weights
        # x_pair = dists / 2
        # x_bound = bound_dists
        
        safe_dists = np.maximum(dists, 1e-9)
        safe_norms = np.maximum(norms, 1e-9)
        
        exp_pair = np.exp(-beta * dists / 2.0)
        exp_bound = np.exp(-beta * bound_dists)
        
        Z = np.sum(exp_pair) + np.sum(exp_bound)
        
        w_pair = exp_pair / Z
        w_bound = exp_bound / Z
        
        # Matrix M for pair gradients
        # M_ij = w_pair_k / (2 * dist_k) for pair (i,j)
        M = np.zeros((N, N))
        factor = w_pair / (2.0 * safe_dists)
        M[i_idx, j_idx] = factor
        M[j_idx, i_idx] = factor
        
        S = M.sum(axis=1)
        V = M @ c
        grad_pair = S[:, None] * c - V
        
        # Boundary gradient
        grad_bound = w_bound[:, None] * (-c / safe_norms[:, None])
        
        grad = grad_pair + grad_bound
        
        # Update
        c += lr * grad
        
        # Project to disk
        new_norms = np.linalg.norm(c, axis=1)
        mask = new_norms > 1.0
        if np.any(mask):
            c[mask] = c[mask] / new_norms[mask, None]
            
        # Score
        curr_score = min(np.min(dists) / 2.0, np.min(bound_dists))
        
        if curr_score > best_score:
            best_score = curr_score
            best_c = c.copy()
            
        # Schedule
        if step % 100 == 0:
            beta *= 1.01
            lr *= 0.995
            
    return best_c
