"""
Strategy: Physics-Based Repulsion with Nelder-Mead Polish
Instead of stochastic Simulated Annealing which moves one point at a time,
we model the spheres as particles with strong repulsive forces (inverse square law)
and attractive forces toward the center (to keep them in the cube).
We solve for the equilibrium configuration using vectorized Newton-Raphson iterations
with adaptive step sizes. This allows moving all 50 points simultaneously in a
coordinated way, escaping local minima much faster than single-point moves.
Finally, we use scipy's Nelder-Mead to polish the exact radius, which is a
non-smooth function but well-suited for direct search in low dimensions (150D).
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    # Centers are in [-0.5, 0.5]^3
    # Distance to walls
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists, axis=1)
    
    radii = np.minimum(wall_dists, min_pair_dist / 2.0)
    
    return np.min(radii)

def _repulsion_force(centers, k_rep=1.0, k_att=0.1, r_target=0.13):
    """
    Compute the net force on each center.
    Repulsion is inverse square law between pairs.
    Attraction is linear spring to center (0,0,0) to prevent drifting out.
    We also add a hard wall repulsion if too close to boundary.
    """
    forces = np.zeros_like(centers)
    
    # Pairwise repulsion
    # diff[i, j] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    np.fill_diagonal(dists, 1.0)
    
    # Force magnitude: k * (1/d^2) * unit_vector
    # We want to push them apart.
    # F_ij = k_rep * (centers[i] - centers[j]) / d^3
    # To vectorize sum over j for each i:
    # F_i = sum_j k_rep * (c_i - c_j) / |c_i - c_j|^3
    
    # dists_cubed = dists^3
    dists_cubed = dists**3
    # Mask diagonal to avoid self-force (already set to 1, but 1/1^3 is fine if we subtract self)
    # Actually, let's just zero out the diagonal contribution explicitly in the sum
    # term = k_rep * diff / dists_cubed[:,:,np.newaxis]
    # term[i,i] is 0/1 = 0, so it's naturally zero.
    
    term = k_rep * diff / dists_cubed[:, :, np.newaxis]
    
    # Sum over j (axis=1) to get force on i
    pair_forces = np.sum(term, axis=1)
    
    # Wall repulsion
    # If center is close to wall (dist < r_target), push it in.
    # dist to wall is 0.5 - |c|
    # Let's use a simple linear spring to the center for global containment
    # and a stronger repulsion if |c| > 0.5 - r_target
    
    # Linear attraction to center
    center_force = -k_att * centers
    
    # Wall repulsion
    # dist_to_wall = 0.5 - np.max(np.abs(centers), axis=1)
    # This is tricky because it's the max over coordinates.
    # Let's just use a strong penalty if any coordinate is out of bounds.
    # But we are clipping in the update step anyway.
    # Let's add a soft wall: if |c_k| > 0.4, push back.
    soft_wall_limit = 0.4
    wall_viol = np.maximum(np.abs(centers) - soft_wall_limit, 0.0)
    wall_forces = -k_rep * 2.0 * wall_viol * np.sign(centers)
    
    return pair_forces + center_force + wall_forces

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Initial configuration: Jittered grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    centers = grid[:N].copy()
    centers += rng.normal(0, 0.01, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Phase 1: Physics-based relaxation
    # We run multiple iterations.
    # Force = F(centers)
    # Update: centers += step_size * F
    
    current_centers = centers.copy()
    
    # Parameters
    k_rep = 10.0
    k_att = 0.5
    r_target = 0.135 # Roughly the target radius
    
    # We will use a simple gradient ascent on the "repulsion energy"
    # which is equivalent to gradient descent on the "potential energy"
    # E = - sum 1/d^2 - k_att * sum |c|^2 + wall_penalty
    
    # Let's just do iterative force application.
    # Step size needs to be small enough for stability.
    
    num_physics_steps = 2000
    step_size = 0.005
    
    for i in range(num_physics_steps):
        if time.time() > deadline - 20: # Leave time for polish
            break
            
        forces = _repulsion_force(current_centers, k_rep, k_att, r_target)
        
        # Normalize forces to prevent explosion
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 1e-8:
            forces = forces / max_force * 0.1 # Scale to max 0.1 displacement
        
        current_centers += step_size * forces
        current_centers = np.clip(current_centers, -0.499, 0.499)
        
        # Check improvement every 100 steps
        if i % 100 == 0:
            r = _compute_radius(current_centers)
            if r > best_radius:
                best_radius = r
                best_centers = current_centers.copy()
                
    # Phase 2: Nelder-Mead Polish
    # We want to maximize the minimum distance.
    # scipy.optimize.minimize minimizes. So we minimize -radius.
    # But radius is non-smooth. Nelder-Mead is good for this.
    # However, 150 dimensions is high for NM.
    # Let's try a local refinement using a smaller neighborhood or just NM on the whole thing.
    # NM in 150D might be slow.
    # Alternative: Use a local SA or just a few iterations of coordinate ascent.
    
    # Let's try a simple coordinate ascent / local search.
    # For each point, try to move it in 6 directions ( +/- x, +/- y, +/- z)
    # and accept if it improves the radius.
    
    current_centers = best_centers.copy()
    current_radius = best_radius
    
    # Local Search
    step = 0.001
    improvements = 0
    no_improve_count = 0
    
    while time.time() < deadline - 2:
        improved = False
        for idx in range(N):
            # Try 6 directions
            for axis in range(3):
                for direction in [-1, 1]:
                    new_c = current_centers.copy()
                    new_c[idx, axis] += direction * step
                    
                    # Check bounds
                    if np.any(new_c < -0.5) or np.any(new_c > 0.5):
                        continue
                        
                    r = _compute_radius(new_c)
                    if r > current_radius + 1e-9:
                        current_centers = new_c
                        current_radius = r
                        best_radius = r
                        best_centers = new_c.copy()
                        improved = True
                        improvements += 1
                        break # Move to next axis if improved
                if improved:
                    break # Move to next point if improved
            if improved:
                break # Move to next point if improved
                
        if not improved:
            no_improve_count += 1
            # Reduce step size
            step *= 0.9
            if step < 1e-6:
                break
        else:
            no_improve_count = 0
            
    return best_centers

# EVOLVE-BLOCK-END
