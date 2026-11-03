import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # Initialize with 5x5x5 grid
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_score = _score(centers)
    best_centers = centers.copy()
    
    # Repulsion-based local search
    alpha = 0.001
    for it in range(20000):
        if time.time() > deadline:
            break
        
        # Compute pairwise distances
        diff = centers[:, None, :] - centers[None, :, :]  # (N, N, 3)
        dist = np.linalg.norm(diff, axis=2)  # (N, N)
        np.fill_diagonal(dist, 1.0)
        
        # Repulsion forces: sum over j of (c_i - c_j) / |c_i - c_j|^2
        # force[i] = sum_j (c_i - c_j) / |c_i - c_j|^3 * (1/|c_i - c_j|) ... 
        # Actually: F_i = sum_{j≠i} (c_i - c_j) / |c_i - c_j|^2
        inv_dist2 = 1.0 / dist**2
        inv_dist2[np.diag_indices(N)] = 0
        forces = np.sum(diff * inv_dist2[:, :, None], axis=1)
        
        # Wall repulsion
        for i in range(N):
            for k in range(3):
                wall_dist = min(centers[i, k] + 0.5, 0.5 - centers[i, k])
                if wall_dist < 0.15:
                    if centers[i, k] + 0.5 < 0.5 - centers[i, k]:
                        forces[i, k] += 0.5 / wall_dist
                    else:
                        forces[i, k] -= 0.5 / wall_dist
        
        centers += alpha * forces
        centers = np.clip(centers, -0.5, 0.5)
        
        if it % 50 == 0:
            score = _score(centers)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
    
    return best_centers

def _score(centers):
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    min_radius = 1.0
    for i in range(N):
        for j in range(i+1, N):
            if dist[i, j] / 2 < min_radius:
                min_radius = dist[i, j] / 2
        for k in range(3):
            wd = min(centers[i, k] + 0.5, 0.5 - centers[i, k])
            if wd < min_radius:
                min_radius = wd
    return min_radius
