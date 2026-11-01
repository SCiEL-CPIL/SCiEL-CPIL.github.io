"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # We only want upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Clamp to avoid log of very small numbers
    d = np.maximum(d, 1e-15)
    # Energy is sum of log(1/d) = -sum log(d)
    # Use math.fsum for precision as per task description
    log_vals = np.log(d)
    return -math.fsum(list(log_vals))

def _gradient_log_energy(points):
    """Compute gradient of the logarithmic energy with respect to points.
    
    Energy E = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    grad = np.zeros_like(points)
    # Vectorized computation for speed
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero on diagonal
    dist_sq_diag = np.diag(dist_sq)
    np.fill_diagonal(dist_sq, 1.0) # Temporary fix to avoid 0/0 in division, we'll mask later
    
    # Gradient term: (x_i - x_j) / |x_i - x_j|^2
    # This is diff / dist_sq
    grad_terms = diff / dist_sq[:, :, np.newaxis]
    
    # Sum over j for each i
    # Note: The diagonal term was diff=0, so it contributes 0. 
    # But we replaced dist_sq diag with 1, so 0/1 = 0. Correct.
    grad = np.sum(grad_terms, axis=1)
    return grad

def _exponential_map(points, velocities, step_size):
    """Move points on the sphere using the exponential map.
    x_new = cos(step*|v|) * x + sin(step*|v|) * (v/|v|)
    """
    n = len(points)
    v_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    # Avoid division by zero
    v_norms_safe = np.where(v_norms < 1e-12, 1e-12, v_norms)
    
    cos_term = np.cos(step_size * v_norms_safe)
    sin_term = np.sin(step_size * v_norms_safe) / v_norms_safe
    
    # If v is 0, sin_term should be 0. 
    # With safe norm, if v_norm is tiny, sin(tiny)/tiny ~ 1. But v is 0, so 0*1=0.
    # Actually, if v is exactly 0, v_norm_safe is 1e-12. sin(1e-12*step)/1e-12 ~ step.
    # v * step = 0. So result is cos(0)*x + 0 = x. Correct.
    
    new_points = cos_term * points + sin_term * velocities
    return new_points

def _local_refine(points, max_iters=500, initial_lr=0.01, final_lr=1e-7):
    """Perform local gradient descent refinement on the sphere."""
    current_energy = _log_energy(points)
    lr = initial_lr
    momentum = 0.9
    velocity = np.zeros_like(points)
    
    for it in range(max_iters):
        grad = _gradient_log_energy(points)
        
        # Update velocity with momentum
        velocity = momentum * velocity - lr * grad
        
        # Move points
        new_points = _exponential_map(points, velocity, 1.0)
        
        # Ensure points are on sphere
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_energy = _log_energy(new_points)
        
        # If energy increases significantly, reject and reduce step size
        if new_energy > current_energy:
            lr *= 0.5
            # Revert
            continue
        else:
            points = new_points
            current_energy = new_energy
            # Decay learning rate
            lr *= 0.999
            
            # Check convergence
            if lr < final_lr:
                break

    return points

def construct():
    # Use Differential Evolution for global search
    # followed by local refinement.
    
    pop_size = 40
    max_generations = 100
    F = 0.5  # Mutation factor
    CR = 0.9  # Crossover probability
    
    # Initialize population with random points on the sphere
    rng = np.random.default_rng(42)
    population = []
    for _ in range(pop_size):
        # Generate random points using normal distribution and normalize
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        population.append(pts)
        
    # Evaluate initial population
    energies = [_log_energy(pts) for pts in population]
    
    best_idx = np.argmin(energies)
    best_energy = energies[best_idx]
    best_points = population[best_idx].copy()
    
    for gen in range(max_generations):
        # Check time limit roughly
        # if time.time() > start_time + 100: break # We don't have start_time easily here without passing it, but 100 gens * 40 pop * 100 pts should be fast.
        
        new_population = []
        new_energies = []
        
        for i in range(pop_size):
            # Select 3 distinct random indices different from i
            r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
            while r1 == i or r2 == i or r3 == i:
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
            
            # Mutation: v = x_r1 + F * (x_r2 - x_r3)
            mutant = population[r1] + F * (population[r2] - population[r3])
            
            # Normalize mutant to sphere
            mutant /= np.linalg.norm(mutant, axis=1, keepdims=True)
            
            # Crossover: Binary crossover
            trial = population[i].copy()
            # Random vector to determine which dimensions to take from mutant
            cross_mask = rng.random((N, 3)) < CR
            # Ensure at least one dimension is taken from mutant
            if not np.any(cross_mask):
                j = rng.integers(0, N)
                k = rng.integers(0, 3)
                cross_mask[j, k] = True
            
            trial[cross_mask] = mutant[cross_mask]
            
            # Normalize trial to sphere
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            
            # Evaluate trial
            trial_energy = _log_energy(trial)
            
            # Selection
            if trial_energy < energies[i]:
                new_population.append(trial)
                new_energies.append(trial_energy)
            else:
                new_population.append(population[i])
                new_energies.append(energies[i])
        
        population = new_population
        energies = new_energies
        
        # Update best
        gen_best_idx = np.argmin(energies)
        if energies[gen_best_idx] < best_energy:
            best_energy = energies[gen_best_idx]
            best_points = population[gen_best_idx].copy()

    # Refine the best point found
    best_points = _local_refine(best_points, max_iters=1000, initial_lr=0.05, final_lr=1e-8)
    
    # Final normalization
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    # Check for close points
    min_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < min_dist:
                min_dist = d
                
    if min_dist < 1e-9:
        # Perturb if necessary
        rng = np.random.default_rng(42)
        best_points = best_points + rng.normal(0, 1e-6, best_points.shape)
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
        # Re-refine after perturbation
        best_points = _local_refine(best_points, max_iters=500, initial_lr=0.01, final_lr=1e-8)
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points
# EVOLVE-BLOCK-END
