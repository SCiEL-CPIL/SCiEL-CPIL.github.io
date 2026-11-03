import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate spacing
    d_est = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    d_lattice = d_est * 0.85
    
    v1 = np.array([d_lattice, 0])
    v2 = np.array([d_lattice / 2.0, d_lattice * np.sqrt(3.0) / 2.0])
    
    range_coord = int(np.ceil(1.0 / d_lattice)) + 2
    
    points = []
    for i in range(-range_coord, range_coord + 1):
        for j in range(-range_coord, range_coord + 1):
            p = i * v1 + j * v2
            points.append(p)
            
    points = np.array(points)
    
    # Filter
    margin = d_lattice / 2.0
    valid_mask = (points[:, 0] >= -0.5 + margin) & (points[:, 0] <= 0.5 - margin) & \
                 (points[:, 1] >= -0.5 + margin) & (points[:, 1] <= 0.5 - margin)
    
    valid_points = points[valid_mask]
    
    if len(valid_points) < N:
        # Fallback
        d_lattice = d_lattice * 0.8
        v1 = np.array([d_lattice, 0])
        v2 = np.array([d_lattice / 2.0, d_lattice * np.sqrt(3.0) / 2.0])
        range_coord = int(np.ceil(1.0 / d_lattice)) + 2
        points = []
        for i in range(-range_coord, range_coord + 1):
            for j in range(-range_coord, range_coord + 1):
                p = i * v1 + j * v2
                points.append(p)
        points = np.array(points)
        margin = d_lattice / 2.0
        valid_mask = (points[:, 0] >= -0.5 + margin) & (points[:, 0] <= 0.5 - margin) & \
                     (points[:, 1] >= -0.5 + margin) & (points[:, 1] <= 0.5 - margin)
        valid_points = points[valid_mask]
        
    if len(valid_points) < N:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # Select N points closest to center
    dists_center = np.linalg.norm(valid_points, axis=1)
    indices = np.argsort(dists_center)[:N]
    centers = valid_points[indices].copy()
    
    # Optimization
    beta = 20.0
    alpha = 0.005
    
    def get_grads(c, b):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Safe division
        safe_dists = np.where(dists > 1e-6, dists, 1.0)
        V = diff / safe_dists[..., np.newaxis]
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # Boundary dists
        b_dists = np.zeros((N, 4))
        b_dists[:, 0] = 0.5 - c[:, 0]
        b_dists[:, 1] = 0.5 + c[:, 0]
        b_dists[:, 2] = 0.5 - c[:, 1]
        b_dists[:, 3] = 0.5 + c[:, 1]
        
        # Flatten
        pair_dists = dists[mask]
        all_dists = np.concatenate([pair_dists, b_dists.flatten()])
        
        # Weights
        max_d = np.max(all_dists)
        exp_vals = np.exp(-b * (all_dists - max_d))
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        w_pairs = weights[:N*(N-1)]
        w_bounds = weights[N*(N-1):].reshape(N, 4)
        
        w_mat = np.zeros((N, N))
        w_mat[mask] = w_pairs
        
        # Pair Grad
        grads = np.einsum('ij,ijk->ik', w_mat, V)
        
        # Bound Grad
        grads[:, 0] += -w_bounds[:, 0] + w_bounds[:, 1]
        grads[:, 1] += -w_bounds[:, 2] + w_bounds[:, 3]
        
        return grads, np.min(all_dists)

    best_centers = centers.copy()
    best_score = 0.0
    
    for step in range(2000):
        if time.time() > deadline:
            break
            
        grads, score = get_grads(centers, beta)
        
        centers += alpha * grads
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
        if step % 100 == 0:
            beta *= 1.05
            alpha *= 0.98

    return best_centers
# EVOLVE-BLOCK-END
