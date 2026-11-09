"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _score(positions):
    """Compute the radius for a given set of positions."""
    if positions.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(positions)):
        return 0.0
    
    # Check if all points are inside the unit circle (with some margin for numerical safety)
    dist_from_origin = np.linalg.norm(positions, axis=1)
    if np.any(dist_from_origin > 1.0):
        return 0.0
    
    # Compute minimum distance between any pair of points
    # Using broadcasting for pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to ignore self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum distance between centers
    r_pair = min_pair_dist / 2.0
    
    # The radius is also limited by the distance from the boundary
    r_boundary = np.min(1.0 - dist_from_origin)
    
    return min(r_pair, r_boundary)

def _construct_pso():
    """Particle Swarm Optimization for circle packing."""
    np.random.seed(42)
    
    n_particles = 50
    max_iter = 2000
    w = 0.7  # inertia weight
    c1 = 1.5  # cognitive parameter
    c2 = 1.5  # social parameter
    
    # Initialize particles
    # Each particle is a (N, 2) array of positions
    particles = np.zeros((n_particles, N, 2))
    velocities = np.zeros((n_particles, N, 2))
    
    # Initialize within the unit circle
    for i in range(n_particles):
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        particles[i, :, 0] = radii * np.cos(angles)
        particles[i, :, 1] = radii * np.sin(angles)
    
    # Evaluate initial fitness
    fitness = np.array([_score(particles[i]) for i in range(n_particles)])
    personal_best = particles.copy()
    personal_best_fitness = fitness.copy()
    
    global_best_idx = np.argmax(fitness)
    global_best = particles[global_best_idx].copy()
    global_best_fitness = fitness[global_best_idx]
    
    for iteration in range(max_iter):
        # Update velocities and positions
        for i in range(n_particles):
            r1 = np.random.uniform(0, 1, (N, 2))
            r2 = np.random.uniform(0, 1, (N, 2))
            
            velocities[i] = (w * velocities[i] +
                           c1 * r1 * (personal_best[i] - particles[i]) +
                           c2 * r2 * (global_best - particles[i]))
            
            particles[i] = particles[i] + velocities[i]
            
            # Project back into unit circle if needed
            dist = np.linalg.norm(particles[i], axis=1)
            mask = dist > 1.0
            if np.any(mask):
                particles[i][mask] /= dist[mask][:, np.newaxis]
                particles[i][mask] *= 0.999  # Keep slightly inside
        
        # Evaluate new fitness
        for i in range(n_particles):
            new_fitness = _score(particles[i])
            
            # Update personal best
            if new_fitness > personal_best_fitness[i]:
                personal_best[i] = particles[i].copy()
                personal_best_fitness[i] = new_fitness
                
                # Update global best
                if new_fitness > global_best_fitness:
                    global_best = particles[i].copy()
                    global_best_fitness = new_fitness
        
        # Decay inertia weight
        w = 0.7 - 0.4 * (iteration / max_iter)
        
        # Early stopping if we've reached a very high score
        if global_best_fitness > 0.125:
            break
    
    return global_best

def _construct_grid_refinement():
    """Grid-based initialization followed by local refinement."""
    # Start with a hexagonal grid arrangement
    points = []
    spacing = 0.25
    sqrt3 = np.sqrt(3)
    
    for j in range(-20, 21):
        for i in range(-20, 21):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2
            if x**2 + y**2 <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # Sort by distance from origin and take the closest N
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:N]]
    
    # Refine using local optimization
    def negative_score(x_flat):
        positions = x_flat.reshape(N, 2)
        return -_score(positions)
    
    result = minimize(negative_score, points.flatten(), method='L-BFGS-B',
                     options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
    
    return result.x.reshape(N, 2)

def _construct_hybrid():
    """Combine PSO with local refinement."""
    # Get a good initial guess from PSO
    positions_pso = _construct_pso()
    
    # Refine using local optimization
    def negative_score(x_flat):
        positions = x_flat.reshape(N, 2)
        return -_score(positions)
    
    result = minimize(negative_score, positions_pso.flatten(), method='L-BFGS-B',
                     options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
    
    return result.x.reshape(N, 2)

def construct():
    """Construct the optimal packing of 50 circles in a unit circle."""
    # Try hybrid approach
    positions = _construct_hybrid()
    
    # Ensure all points are strictly inside the unit circle
    dist = np.linalg.norm(positions, axis=1)
    mask = dist > 0.9999
    if np.any(mask):
        positions[mask] *= 0.9999 / dist[mask][:, np.newaxis]
    
    # Verify the result
    score = _score(positions)
    if score <= 0:
        # Fallback to grid refinement
        positions = _construct_grid_refinement()
        dist = np.linalg.norm(positions, axis=1)
        mask = dist > 0.9999
        if np.any(mask):
            positions[mask] *= 0.9999 / dist[mask][:, np.newaxis]
    
    return positions
# EVOLVE-BLOCK-END
