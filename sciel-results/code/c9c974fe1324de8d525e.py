# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree

N = 100

def _radius(c):
    """Compute the packing radius for a configuration c of shape (N, 3)."""
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Use KDTree for faster nearest neighbor search? 
    # pdist is O(N^2) but optimized. KDTree query is O(N log N).
    # For N=100, pdist is likely faster due to lower constant factor.
    # Let's stick to pdist for the global evaluation as it's robust.
    from scipy.spatial.distance import pdist
    dists = pdist(c, metric='euclidean')
    if len(dists) > 0:
        min_pair = np.min(dists)
    else:
        min_pair = np.inf
        
    return min(min_wall, min_pair / 2.0)

def _local_refine_repulsion(c_flat, steps=100, step_size=0.01):
    """
    Perform local refinement using a repulsion-based heuristic.
    """
    c = c_flat.reshape(N, 3).copy()
    for _ in range(steps):
        forces = np.zeros_like(c)
        
        # Wall repulsion
        margin = 0.05
        for k in range(3):
            # Push from left wall
            mask = c[:, k] < -0.5 + margin
            if np.any(mask):
                forces[mask, k] += (margin - (c[mask, k] + 0.5))
            # Push from right wall
            mask = c[:, k] > 0.5 - margin
            if np.any(mask):
                forces[mask, k] -= (margin - (0.5 - c[mask, k]))
            
        # Pairwise repulsion using KDTree
        tree = cKDTree(c)
        dists, indices = tree.query(c, k=2)
        
        for i in range(N):
            j = indices[i, 1]
            d = dists[i, 1]
            if d > 1e-6:
                vec = c[i] - c[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    # Push if close
                    if d < 0.25:
                        # Force magnitude
                        f = 0.1 * (1.0 - d / 0.25)
                        forces[i] += (vec / norm) * f
                        forces[j] -= (vec / norm) * f
                        
        c += forces * step_size
        c = np.clip(c, -0.5, 0.5)
        
    return c.ravel()

def _generate_lattice_init():
    """
    Generate an initial configuration based on a simple cubic lattice.
    """
    nx, ny, nz = 4, 5, 5
    a = 1.0 / max(nx, ny, nz) * 0.95
    
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = (i - (nx - 1) / 2.0) * a
                y = (j - (ny - 1) / 2.0) * a
                z = (k - (nz - 1) / 2.0) * a
                points.append([x, y, z])
                
    points = np.array(points[:N])
    points += np.random.randn(N, 3) * 0.01
    return points.ravel()

def construct():
    np.random.seed(42)
    
    swarm_size = 30
    max_iter = 300
    w_start = 0.9
    w_end = 0.4
    c1 = 1.5
    c2 = 1.5
    
    particles = np.zeros((swarm_size, N * 3))
    
    base_lattice = _generate_lattice_init()
    
    for i in range(swarm_size):
        if i == 0:
            particles[i] = base_lattice
        elif i < 5:
            particles[i] = base_lattice + np.random.randn(N * 3) * 0.05
        else:
            particles[i] = np.random.uniform(-0.4, 0.4, N * 3)
            
    personal_bests = np.zeros_like(particles)
    personal_best_vals = np.zeros(swarm_size)
    
    for i in range(swarm_size):
        refined = _local_refine_repulsion(particles[i], steps=50)
        particles[i] = refined
        val = -_radius(particles[i].reshape(N, 3))
        personal_best_vals[i] = val
        personal_bests[i] = particles[i].copy()
        
    global_best_idx = np.argmin(personal_best_vals)
    global_best = personal_bests[global_best_idx].copy()
    global_best_val = personal_best_vals[global_best_idx]
    
    velocities = np.zeros_like(particles)
    
    for it in range(max_iter):
        w = w_start - (w_start - w_end) * (it / max_iter)
        
        for i in range(swarm_size):
            r1 = np.random.rand(N * 3)
            r2 = np.random.rand(N * 3)
            
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (personal_bests[i] - particles[i]) + 
                           c2 * r2 * (global_best - particles[i]))
            
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], -0.5, 0.5)
            
            # Local refinement
            iter_limit = 50 if it < max_iter // 2 else 20
            refined = _local_refine_repulsion(particles[i], steps=iter_limit)
            particles[i] = refined
            
            val = -_radius(particles[i].reshape(N, 3))
            
            if val < personal_best_vals[i]:
                personal_best_vals[i] = val
                personal_bests[i] = particles[i].copy()
                
                if val < global_best_val:
                    global_best_val = val
                    global_best = particles[i].copy()
                    
    # Final refinement
    c_best = global_best.reshape(N, 3)
    final_refined = _local_refine_repulsion(global_best, steps=200)
    c_final = np.clip(final_refined.reshape(N, 3), -0.5, 0.5)
    
    return c_final
# EVOLVE-BLOCK-END
