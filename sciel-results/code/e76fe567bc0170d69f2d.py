"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    # Check bounds
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    
    # Distance from each center to nearest wall
    wall_dists = np.min(np.array([
        centers[:, 0] + 0.5,
        -centers[:, 0] + 0.5,
        centers[:, 1] + 0.5,
        -centers[:, 1] + 0.5,
        centers[:, 2] + 0.5,
        -centers[:, 2] + 0.5,
    ]), axis=1)
    
    if np.any(wall_dists < -1e-10):
        return 0.0
    
    # Minimum distance to walls
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    r = 0.5 * min(min_pair, min_wall)
    return r if r > 0 else 0.0

def _force_directed_relax(centers, max_iters=2000, dt=0.001, tol=1e-9):
    """Relax centers using force-directed method with soft repulsion."""
    centers = centers.copy()
    n = len(centers)
    
    for it in range(max_iters):
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, 1.0)
        
        # Soft repulsion: F = 1/d^2 when close
        for i in range(n):
            for j in range(i+1, n):
                d = dists[i, j]
                if d < 0.01:
                    f = 1.0 / (d * d + 1e-6)
                    diff = centers[i] - centers[j]
                    forces[i] += f * diff / d
                    forces[j] -= f * diff / d
        
        # Wall repulsion
        for i in range(n):
            for axis in range(3):
                pos = centers[i, axis]
                if pos > 0.5:
                    forces[i, axis] -= 100.0 * (pos - 0.5)
                elif pos < -0.5:
                    forces[i, axis] += 100.0 * (-0.5 - pos)
        
        # Update positions
        centers += dt * forces
        
        # Clamp to cube
        centers = np.clip(centers, -0.4999, 0.4999)
        
        if np.max(np.abs(forces)) < tol:
            break
    
    return centers

def _local_refine(centers):
    """Refine a packing by pushing spheres apart when they overlap."""
    centers = centers.copy()
    n = len(centers)
    
    for iteration in range(100):
        improved = False
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        
        for i in range(n):
            for j in range(i+1, n):
                d = dists[i, j]
                if d < 0.01:
                    # Push apart
                    diff = centers[i] - centers[j]
                    norm = np.linalg.norm(diff)
                    if norm > 1e-10:
                        push = 0.001 * diff / norm
                        centers[i] += push
                        centers[j] -= push
                        improved = True
        
        # Keep inside cube
        centers = np.clip(centers, -0.4999, 0.4999)
        
        if not improved:
            break
    
    return centers

def construct():
    """Construct centers for N equal spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial: grid placement
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    best_radius = _compute_radius(best_centers)
    
    # Try multiple random restarts with force-directed relaxation
    rng = np.random.RandomState(42)
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Random initial positions
        centers = rng.uniform(-0.4, 0.4, size=(N, 3))
        
        # Force-directed relaxation
        centers = _force_directed_relax(centers, max_iters=500, dt=0.005)
        
        # Local refinement
        centers = _local_refine(centers)
        
        radius = _compute_radius(centers)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Final local refinement on best
    best_centers = _local_refine(best_centers)
    final_radius = _compute_radius(best_centers)
    if final_radius > best_radius:
        best_radius = final_radius
    
    return best_centers
# EVOLVE-BLOCK-END
