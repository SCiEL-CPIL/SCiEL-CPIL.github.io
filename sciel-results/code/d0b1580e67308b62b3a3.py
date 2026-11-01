"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # New Approach: Simulated Annealing with Repulsion Potentials
    # Objective: Maximize min(pairwise_dist/2, boundary_dist)
    # Strategy: Minimize an energy function E = sum(1/d^k) + sum(1/b^k)
    # This effectively pushes points apart and away from the boundary.
    
    # 1. Initialization: Vogel Spiral for a good start
    golden_angle = np.pi * (3 - np.sqrt(5))
    points = np.zeros((N, 2))
    for i in range(N):
        r = np.sqrt(i / N)
        theta = i * golden_angle
        points[i] = [r * np.cos(theta), r * np.sin(theta)]
    
    # Parameters
    k_power = 4.0  # Power for repulsion
    lambda_boundary = 1.0 # Weight for boundary penalty
    T_start = 0.05
    T_end = 1e-5
    cooling_rate = 0.9995
    max_iters = 15000
    noise_scale = 0.03
    
    start_time = time.time()
    time_budget = 115.0 # seconds
    
    current_energy = None
    
    def calculate_energy(pts):
        # Vectorized energy calculation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero for self
        
        eps = 1e-12
        dists_sq_safe = dists_sq + eps
        
        # Pair energy: sum(1/d^k) = sum(1/(d^2)^(k/2))
        # k=4 => 1/(d^2)^2
        pair_energy = np.sum(1.0 / (dists_sq_safe ** (k_power / 2.0)))
        
        # Boundary energy
        norms = np.linalg.norm(pts, axis=1)
        boundary_dist = 1.0 - norms
        
        # Penalize if boundary_dist is small or negative
        # If boundary_dist > eps, use 1/boundary_dist^k
        # Else, use a large penalty
        valid_boundary = boundary_dist > 1e-6
        b_energy = np.zeros(N)
        b_energy[valid_boundary] = 1.0 / (boundary_dist[valid_boundary] ** k_power)
        b_energy[~valid_boundary] = 1e10 # Huge penalty if outside or on boundary
        
        total_energy = pair_energy + lambda_boundary * np.sum(b_energy)
        return total_energy

    # Initial Energy
    current_energy = calculate_energy(points)
    best_energy = current_energy
    best_points = points.copy()
    
    T = T_start
    
    for i in range(max_iters):
        if time.time() - start_time > time_budget:
            break
            
        # Generate perturbation
        noise = np.random.normal(0, noise_scale, size=(N, 2))
        new_points = points + noise
        
        # Calculate new energy
        new_energy = calculate_energy(new_points)
        
        # Metropolis criterion
        delta_E = new_energy - current_energy
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            points = new_points
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
                
        # Cool down
        T *= cooling_rate
        
    # Final Local Refinement: 
    # Use smaller steps to polish.
    T = T * 0.5
    noise_scale = 0.01
    current_energy = calculate_energy(points)
    best_energy_local = current_energy
    best_points_local = points.copy()
    
    for i in range(5000):
        if time.time() - start_time > time_budget:
            break
        noise = np.random.normal(0, noise_scale, size=(N, 2))
        new_points = points + noise
        new_energy = calculate_energy(new_points)
        
        delta_E = new_energy - current_energy
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            points = new_points
            current_energy = new_energy
            if current_energy < best_energy_local:
                best_energy_local = current_energy
                best_points_local = points.copy()
        
        T *= 0.999
        
    # Return the best points found
    if best_energy_local < best_energy:
        return best_points_local
    else:
        return best_points

# EVOLVE-BLOCK-END
