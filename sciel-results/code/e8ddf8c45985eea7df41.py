"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_score(centers):
    """Compute the packing score: min of pairwise distances/2 and distance to boundary."""
    # Distance from each center to origin
    dist_to_origin = np.sqrt(np.sum(centers ** 2, axis=1))
    # Distance to boundary
    boundary_dist = 1.0 - dist_to_origin
    # If any center is outside, return a very negative score
    if np.any(boundary_dist < -1e-10):
        return -1e10
    
    # Pairwise distances
    n = len(centers)
    # Use broadcasting to compute all pairwise distances
    # This creates an (n, n) matrix
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is limited by both the boundary and pairwise distances
    r = min(np.min(boundary_dist), min_pair_dist / 2.0)
    
    # If any boundary distance is negative, the configuration is invalid
    if r < 0:
        return -1e10 + r  # Penalize but keep some gradient info
    
    return r


def _particle_swarm_optimization():
    """PSO to find optimal packing of N equal circles in unit circle."""
    np.random.seed(42)
    
    n_particles = 20
    n_iterations = 200
    n_dimensions = N * 2
    
    # Initialize particles
    particles = np.zeros((n_particles, N, 2))
    velocities = np.zeros((n_particles, N, 2))
    
    # Initialize with random points inside unit circle
    for i in range(n_particles):
        # Generate random points
        while True:
            pts = np.random.uniform(-1, 1, size=(N, 2))
            norms = np.sqrt(np.sum(pts ** 2, axis=1))
            if np.all(norms <= 0.95):
                particles[i] = pts
                break
    
    # Compute initial scores
    scores = np.array([_compute_score(particles[i]) for i in range(n_particles)])
    
    # Personal bests
    personal_best = particles.copy()
    personal_best_scores = scores.copy()
    
    # Global best
    global_best_idx = np.argmax(scores)
    global_best = particles[global_best_idx].copy()
    global_best_score = scores[global_best_idx]
    
    # PSO parameters
    w = 0.9  # Inertia weight
    c1 = 1.5  # Cognitive coefficient
    c2 = 1.5  # Social coefficient
    
    for it in range(n_iterations):
        # Update inertia weight (decrease over time)
        w = 0.9 - 0.5 * (it / n_iterations)
        
        for i in range(n_particles):
            # Random coefficients
            r1 = np.random.uniform(0, 1, size=(N, 2))
            r2 = np.random.uniform(0, 1, size=(N, 2))
            
            # Velocity update
            velocities[i] = (w * velocities[i] +
                            c1 * r1 * (personal_best[i] - particles[i]) +
                            c2 * r2 * (global_best - particles[i]))
            
            # Position update
            particles[i] = particles[i] + velocities[i]
            
            # Ensure particles stay within bounds (clip to unit circle with margin)
            norms = np.sqrt(np.sum(particles[i] ** 2, axis=1))
            mask = norms > 0.98
            if np.any(mask):
                particles[i][mask] *= 0.98 / norms[mask, np.newaxis]
            
            # Compute score
            score = _compute_score(particles[i])
            
            # Update personal best
            if score > personal_best_scores[i]:
                personal_best[i] = particles[i].copy()
                personal_best_scores[i] = score
                
                # Update global best
                if score > global_best_score:
                    global_best = particles[i].copy()
                    global_best_score = score
    
    return global_best, global_best_score


def _local_refinement(centers):
    """Use scipy minimize to refine the packing."""
    def neg_score(x):
        pts = x.reshape(N, 2)
        return -_compute_score(pts)
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.99, 0.99)] * (N * 2)
    result = minimize(neg_score, centers.flatten(), method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
    
    refined = result.x.reshape(N, 2)
    # Ensure points are within unit circle
    norms = np.sqrt(np.sum(refined ** 2, axis=1))
    mask = norms > 0.99
    if np.any(mask):
        refined[mask] *= 0.99 / norms[mask, np.newaxis]
    
    return refined, -result.fun


def construct():
    """Construct centers of N equal circles inside unit circle."""
    # Run PSO to get a good starting point
    pso_centers, pso_score = _particle_swarm_optimization()
    
    # Local refinement using gradient-based optimization
    refined_centers, refined_score = _local_refinement(pso_centers)
    
    # Final check and cleanup
    norms = np.sqrt(np.sum(refined_centers ** 2, axis=1))
    mask = norms > 0.999
    if np.any(mask):
        refined_centers[mask] *= 0.999 / norms[mask, np.newaxis]
    
    return refined_centers
# EVOLVE-BLOCK-END
