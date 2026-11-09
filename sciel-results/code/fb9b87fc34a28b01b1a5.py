"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # 1. Initialization: Grid
    # 5x5x2 = 50 points
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.4, 0.4, 2)
    
    centers = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    
    rng = np.random.default_rng(42)
    # Jitter
    centers += rng.normal(0, 0.05, (N, 3))
    centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
    
    # 2. Optimization: Jiggling / Repulsive Relaxation
    # We want to maximize min_dist.
    # We use a repulsive potential to push spheres apart.
    # Force F ~ 1/d^2
    
    n_iter = 2000
    step_size = 0.02
    noise_scale = 0.01
    
    for it in range(n_iter):
        # Calculate forces
        forces = np.zeros_like(centers)
        
        # Pairwise
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0)
        
        # Force magnitude: 1 / dist^2
        # Force direction: diff / dist
        # Total: diff / dist^3
        # Clip dist to avoid inf
        dists_safe = np.maximum(dists, 1e-6)
        
        # scale = 1 / dists_safe^3
        scale = 1.0 / (dists_safe ** 3)
        np.fill_diagonal(scale, 0.0)
        
        sphere_forces = np.sum(scale[:, :, np.newaxis] * diff, axis=1)
        
        # Wall forces
        # Push away from walls with strength 1/w^2
        # F_x = (1/(0.5-x)^2 - 1/(0.5+x)^2) ? No, gradient of potential.
        # Potential V_wall = sum 1/w^2. Grad = -2/w^3 * n.
        # Force = -Grad = 2/w^3 * n.
        # Let's just use a simple push:
        # If x > 0, push left. Force_x = -1/(0.5-x)^2.
        # If x < 0, push right. Force_x = 1/(0.5+x)^2.
        
        x = centers[:, 0]
        y = centers[:, 1]
        z = centers[:, 2]
        
        # Safe distances
        dx_pos = np.maximum(0.5 - x, 1e-6)
        dx_neg = np.maximum(0.5 + x, 1e-6)
        dy_pos = np.maximum(0.5 - y, 1e-6)
        dy_neg = np.maximum(0.5 + y, 1e-6)
        dz_pos = np.maximum(0.5 - z, 1e-6)
        dz_neg = np.maximum(0.5 + z, 1e-6)
        
        # Forces
        fx = -1.0 / (dx_pos ** 2) + 1.0 / (dx_neg ** 2)
        fy = -1.0 / (dy_pos ** 2) + 1.0 / (dy_neg ** 2)
        fz = -1.0 / (dz_pos ** 2) + 1.0 / (dz_neg ** 2)
        
        wall_forces = np.column_stack((fx, fy, fz))
        
        total_forces = sphere_forces + wall_forces
        
        # Add noise (Jiggle)
        noise = rng.normal(0, noise_scale, (N, 3))
        
        # Update
        centers += total_forces * step_size + noise
        
        # Clip
        centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Decay step size and noise
        step_size *= 0.995
        noise_scale *= 0.995
        
    return centers
# EVOLVE-BLOCK-END
