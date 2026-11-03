import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the centers of 50 equal circles inside a unit circle.
    Uses Particle Swarm Optimization (PSO) to maximize the minimum radius.
    """
    
    def calculate_radius(centers):
        """
        Calculates the maximum common radius for a given set of centers.
        """
        # Pairwise distances
        # For N=50, this is 2500 pairs. Vectorized computation is fast.
        dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distance
        norms = np.linalg.norm(centers, axis=1)
        min_boundary_dist = 1.0 - np.max(norms)
        
        # The radius is half the minimum pair distance, limited by the boundary distance
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def project_to_circle(centers):
        """
        Ensures centers are within the unit circle (hard constraint).
        If a center is outside, it is projected back to the boundary.
        """
        norms = np.linalg.norm(centers, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            centers[outside] = centers[outside] / norms[outside, np.newaxis]
        return centers

    # PSO Parameters
    num_particles = 30
    max_iters = 500
    w = 0.9  # Inertia weight
    c1 = 1.5  # Cognitive parameter
    c2 = 1.5  # Social parameter
    
    # Time limit check
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Initialize Swarm
    # We use random initialization for simplicity, but structured init might be better.
    # Let's use random points in the disk.
    X = np.zeros((num_particles, N, 2))
    V = np.zeros((num_particles, N, 2))
    
    for i in range(num_particles):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        X[i, :, 0] = radii * np.cos(angles)
        X[i, :, 1] = radii * np.sin(angles)
        X[i] = project_to_circle(X[i])
        
    # Initialize personal and global bests
    Pbest = X.copy()
    Pbest_fitness = np.zeros(num_particles)
    Gbest = X[0].copy()
    Gbest_fitness = 0.0
    
    for i in range(num_particles):
        f = calculate_radius(X[i])
        Pbest_fitness[i] = f
        if f > Gbest_fitness:
            Gbest_fitness = f
            Gbest = X[i].copy()
    
    # Main Loop
    for t in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Update velocities and positions
        for i in range(num_particles):
            # Random factors
            r1 = np.random.uniform(0, 1, (N, 2))
            r2 = np.random.uniform(0, 1, (N, 2))
            
            # Velocity update
            V[i] = w * V[i] + c1 * r1 * (Pbest[i] - X[i]) + c2 * r2 * (Gbest - X[i])
            
            # Position update
            X[i] = X[i] + V[i]
            
            # Project to circle
            X[i] = project_to_circle(X[i])
            
            # Evaluate fitness
            f = calculate_radius(X[i])
            
            # Update personal best
            if f > Pbest_fitness[i]:
                Pbest_fitness[i] = f
                Pbest[i] = X[i].copy()
                
                # Update global best
                if f > Gbest_fitness:
                    Gbest_fitness = f
                    Gbest = X[i].copy()
                    
    return Gbest

# EVOLVE-BLOCK-END
