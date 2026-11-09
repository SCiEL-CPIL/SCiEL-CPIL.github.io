"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Use a deterministic seed for reproducibility
    rng = np.random.default_rng(42)
    
    # Step 1: Initialize on a dense grid inside the cube
    # Use a grid slightly finer than needed to have more than N candidates
    m = int(np.ceil(N ** (1.0 / 3.0) + 1))  # e.g., for N=50, m=5 -> 125 points
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Randomly select N positions from the grid
    indices = rng.choice(len(grid), size=N, replace=False)
    centers = grid[indices].copy()
    
    # Step 2: Relaxation with repulsive forces
    # We want to maximize min pairwise distance and keep spheres inside cube
    # Use iterative repulsion: each pair pushes apart, walls push inward
    
    n_iter = 2000
    learning_rate = 0.001
    
    for it in range(n_iter):
        # Decay learning rate
        lr = learning_rate * (1.0 - it / n_iter) ** 2
        if lr < 1e-6:
            break
        
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Jitter if coincident
                    diff = rng.normal(0, 1e-4, 3)
                    dist = np.linalg.norm(diff)
                # Inverse-square repulsion
                force = diff / (dist ** 2)
                forces[i] += force
                forces[j] -= force
        
        # Wall repulsion: push spheres away from boundaries
        # Distance to each wall
        for i in range(N):
            for axis in range(3):
                # Left wall at -0.5
                dist_left = centers[i, axis] - (-0.5)
                if dist_left < 0.1:
                    forces[i, axis] += (0.1 - dist_left) * 10.0
                # Right wall at 0.5
                dist_right = 0.5 - centers[i, axis]
                if dist_right < 0.1:
                    forces[i, axis] -= (0.1 - dist_right) * 10.0
        
        # Update positions
        centers += lr * forces
        
        # Project back into cube (hard constraint)
        centers = np.clip(centers, -0.5, 0.5)
    
    # Final: ensure strict interior (small margin to avoid numerical issues)
    margin = 1e-9
    centers = np.clip(centers, -0.5 + margin, 0.5 - margin)
    
    return centers
# EVOLVE-BLOCK-END
