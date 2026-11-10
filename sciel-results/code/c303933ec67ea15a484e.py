import numpy as np
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a molecular dynamics simulation with repulsive forces to find a stable, well-separated configuration.
    """
    np.random.seed(42)
    n = 14
    
    # Initialize points randomly on a sphere of radius 1
    # Using Gaussian distribution and normalizing
    points = np.random.randn(n, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Simulation parameters
    steps = 5000
    dt = 0.005
    # Force constant for repulsion
    k_rep = 10.0
    # Force constant for restoring force to origin (to prevent drift)
    k_restore = 0.1
    
    velocities = np.zeros_like(points)
    
    for _ in range(steps):
        forces = np.zeros_like(points)
        
        # Calculate repulsive forces
        # We want to maximize min distance, so we push points apart.
        # Force is proportional to 1/r^2 or similar.
        # Using a soft repulsion to avoid singularities.
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    # If coincident, push in random direction
                    diff = np.random.randn(3)
                    dist = np.linalg.norm(diff)
                # Force magnitude
                # Inverse square law is standard for electrostatics
                f_mag = k_rep / (dist * dist)
                f_vec = (diff / dist) * f_mag
                
                forces[i] += f_vec
                forces[j] -= f_vec
        
        # Add restoring force to keep points near unit sphere/origin
        # Actually, for Tammes problem, points should be on the sphere.
        # We can project to sphere after each step or add a force towards the sphere surface.
        # Let's add a force that pushes them towards radius 1.
        for i in range(n):
            r = np.linalg.norm(points[i])
            if r > 1e-6:
                # Force towards unit sphere
                # If r < 1, push out; if r > 1, push in.
                # F = k_restore * (1 - r) * (points[i] / r)
                f_restore = k_restore * (1.0 - r) * (points[i] / r)
                forces[i] += f_restore
        
        # Update velocities and positions
        # Simple Euler integration with damping
        velocities = 0.9 * velocities + dt * forces
        points += dt * velocities
        
        # Project back to unit sphere to enforce constraint
        # This is a hard constraint approach
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
    # Final centering (should be near 0)
    points -= np.mean(points, axis=0)
    
    # Normalize to ensure max distance is controlled
    # The metric is (d_min / d_max)^2.
    # Scaling doesn't change the ratio.
    # But we should return finite points.
    
    return points
