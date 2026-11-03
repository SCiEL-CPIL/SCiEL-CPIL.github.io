"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Uses Particle Swarm Optimization (PSO) to find the optimal packing of 50 equal circles in a unit circle.
    Initializes with a Fibonacci spiral (phyllotaxis) pattern.
    """
    def fitness(centers):
        """
        Calculate the maximum radius r such that circles are non-overlapping and inside the unit circle.
        r = min( min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|) )
        """
        # Vectorized pairwise distances
        # centers: (N, 2)
        # diff: (N, N, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Get upper triangle indices to avoid double counting and self-distance
        iu, ju = np.triu_indices(N, k=1)
        min_pair_dist = np.min(dists[iu, ju])
        
        # Distance to boundary
        radii = np.linalg.norm(centers, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        
        # The radius is the minimum of half the min pair distance and the min boundary distance
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        # If any center is outside, r is negative. This is a valid penalty.
        return r

    def init_spiral():
        """
        Initialize centers using a Fibonacci spiral (phyllotaxis).
        This provides a good initial distribution.
        """
        centers = np.zeros((N, 2))
        # Golden angle
        theta = np.pi * (3.0 - np.sqrt(5.0))
        # Scale factor to fit within unit circle
        # r_n = c * sqrt(n)
        # We want r_{N-1} to be close to 1.
        # c * sqrt(N-1) = 1 => c = 1 / sqrt(N-1)
        c = 1.0 / np.sqrt(N - 1)
        
        for i in range(N):
            r = c * np.sqrt(i)
            angle = i * theta
            centers[i, 0] = r * np.cos(angle)
            centers[i, 1] = r * np.sin(angle)
            
        return centers

    # PSO Parameters
    n_particles = 50
    n_iter = 1000
    w = 0.5  # Inertia
    c1 = 1.5 # Cognitive
    c2 = 1.5 # Social
    
    # Initialize positions and velocities
    X = np.zeros((n_particles, N, 2))
    V = np.zeros((n_particles, N, 2))
    
    # Initial positions: mix of spiral and random perturbations
    base_spiral = init_spiral()
    for i in range(n_particles):
        # Add small random noise to spiral
        noise = np.random.uniform(-0.05, 0.05, (N, 2))
        X[i] = base_spiral + noise
        
    # Initialize velocities to be small
    V = np.random.uniform(-0.01, 0.01, (n_particles, N, 2))
    
    # Initialize bests
    P_best = np.zeros((n_particles, N, 2))
    P_best_fit = np.zeros(n_particles)
    G_best = np.zeros((N, 2))
    G_best_fit = -np.inf
    
    # Evaluate initial fitness
    for i in range(n_particles):
        fit = fitness(X[i])
        P_best_fit[i] = fit
        P_best[i] = X[i]
        if fit > G_best_fit:
            G_best_fit = fit
            G_best = X[i]
            
    # Main loop
    for iter_num in range(n_iter):
        # Update velocities and positions
        for i in range(n_particles):
            r1 = np.random.random((N, 2))
            r2 = np.random.random((N, 2))
            
            V[i] = w * V[i] + c1 * r1 * (P_best[i] - X[i]) + c2 * r2 * (G_best - X[i])
            X[i] = X[i] + V[i]
            
            # Optional: Clamp positions to be within the unit circle to prevent extreme values
            # If a particle goes far out, fitness is very negative, it will be corrected.
            # But clamping can help stability.
            radii = np.linalg.norm(X[i], axis=1)
            mask = radii > 1.0
            if np.any(mask):
                # Project back to the circle boundary? Or just let fitness handle it.
                # Letting fitness handle it is standard for penalty methods.
                pass
            
            # Evaluate new fitness
            fit = fitness(X[i])
            P_best_fit[i] = fit
            P_best[i] = X[i]
            
            if fit > G_best_fit:
                G_best_fit = fit
                G_best = X[i]
                
        # Adaptive parameters? Keep simple for now.
        # Reduce w slightly over time?
        # w = 0.9 - 0.4 * (iter_num / n_iter)
        
    return G_best

# EVOLVE-BLOCK-END
