"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = N
    # 1. Initialization
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:n].astype(np.float64)
    
    # Jitter to break symmetry
    np.random.seed(42)
    pts += np.random.normal(0, 0.02, size=pts.shape)
    pts = np.clip(pts, -0.4, 0.4)

    # 2. Iterative Radius Increase with Langevin Relaxation
    r = 0.05
    dr = 0.0005
    relax_steps = 200
    max_outer_iter = 200
    
    # Simulation parameters
    dt = 0.01
    gamma = 0.5
    k_rep = 10.0
    
    v = np.zeros_like(pts)
    best_pts = pts.copy()
    best_r = 0.0
    fail_count = 0
    
    for _ in range(max_outer_iter):
        if r > 0.12:
            break
            
        # Relax the system with current radius r
        for step in range(relax_steps):
            f = np.zeros_like(pts)
            
            # Pairwise forces
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            
            threshold = (2 * r) ** 2
            dist = np.sqrt(dist_sq)
            overlap = np.maximum(0, 2 * r - dist)
            
            safe_dist = np.where(dist > 1e-8, dist, 1.0)
            dir_vec = diff / safe_dist[:, :, np.newaxis]
            
            force_mag = k_rep * overlap
            f += np.sum(dir_vec * force_mag[:, :, np.newaxis], axis=1)
            
            # Wall forces
            wall_overlap = np.maximum(0, np.abs(pts) - (0.5 - r))
            f -= np.sign(pts) * wall_overlap * k_rep
            
            # Update
            v = v * (1 - gamma) + f * dt
            pts = pts + v * dt
            pts = np.clip(pts, -0.5, 0.5)
            
        # Check success
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        wall_dist = 0.5 - np.max(np.abs(pts))
        current_r = min(min_dist / 2.0, wall_dist)
        
        if current_r >= r - 1e-5:
            r += dr
            best_r = current_r
            best_pts = pts.copy()
            fail_count = 0
        else:
            fail_count += 1
            if fail_count > 5:
                break
            break
            
    return best_pts
# EVOLVE-BLOCK-END
