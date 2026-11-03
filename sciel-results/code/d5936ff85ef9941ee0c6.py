"""
Strategy: Physics-based Lennard-Jones Molecular Dynamics with Simulated Annealing.
Instead of direct geometric optimization, we model spheres as particles with repulsive
LJ-like potentials. We use a multi-phase annealing schedule:
1. High temperature: Allow large moves to explore the configuration space and find a
   coarse lattice structure.
2. Cooling: Reduce temperature to settle into local minima of the potential energy.
3. Refinement: Use a very low temperature or direct local search to polish the minimum
   distance.

The potential is designed to penalize overlaps heavily but smoothly, allowing the
system to "flow" around obstacles.
"""

import numpy as np
import time

N = 50

def _compute_radius(centres):
    """Compute the maximum common radius for given centres in [-0.5, 0.5]^3."""
    # Distance to nearest face
    # centres shape (N, 3)
    # face_dist is distance from center to the closest face
    face_dist = 0.5 - np.max(np.abs(centres), axis=1)
    min_face = np.min(face_dist)
    
    # Minimum pairwise distance
    n = len(centres)
    if n < 2:
        return min_face
    
    # Compute all pairwise distances efficiently
    # diff[i, j] = centres[i] - centres[j]
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal to avoid min being 0
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_face, min_pair / 2.0)

def _lj_potential_and_force(centres, sigma=0.25, epsilon=1.0, wall_sigma=0.5, wall_epsilon=10.0):
    """
    Compute total potential energy and forces for Lennard-Jones interaction + Wall repulsion.
    We use a soft-core variant to handle close approaches without infinite forces.
    
    U(r) = epsilon * [ (sigma/r)^12 - 2*(sigma/r)^6 ] + 1/2 * wall_epsilon * (max(0, wall_sigma - dist_to_wall)^2)
    
    Actually, standard LJ has a minimum at r=2^(1/6)*sigma. 
    For packing, we want to maximize the minimum distance. 
    Minimizing a repulsive potential like (1/r)^12 or (1/r)^6 will push particles apart.
    Let's use a simple repulsive potential: U = sum (1/d_ij^6) + wall_penalty.
    This is steeper than LJ at close range and simpler.
    
    However, the prompt suggests Lennard-Jones. Let's stick to a repulsive core.
    U_pair = epsilon * (sigma / d)^12  (Pure repulsion, no attraction to form clusters, just push apart)
    Wait, pure repulsion will push them to the walls. We need the walls to repel too.
    
    Let's use:
    U = sum_{i<j} (sigma/d_ij)^12 + sum_i (wall_sigma / d_wall_i)^12
    where d_wall_i is distance to nearest face.
    
    Forces:
    F_pair_ij = -dU/dd * u_ij = 12 * sigma^12 / d^13 * u_ij (direction from j to i? 
    U = (sigma/d)^12. dU/dd = -12 sigma^12 / d^13.
    F = -grad U. 
    grad U wrt r_i: dU/dd * dd/dr_i. 
    d = |r_i - r_j|. dd/dr_i = (r_i - r_j)/d = u_ij.
    F_i = -dU/dd * u_ij = 12 sigma^12 / d^13 * u_ij. (Repulsive, points away from j).
    
    Wall Force:
    U_wall = (wall_sigma / d_wall)^12.
    If particle is at x, wall at 0.5. d_wall = 0.5 - x (if x < 0.5).
    dU/dx = dU/dd * dd/dx = (-12 wall_sigma^12 / d^13) * (-1) = 12 wall_sigma^12 / d^13.
    F_x = -dU/dx = -12 wall_sigma^12 / d^13. (Pushes towards center, i.e., negative x if x is positive).
    
    We will implement this.
    """
    n = len(centres)
    forces = np.zeros_like(centres)
    energy = 0.0
    
    # Pairwise interactions
    # Vectorized calculation for speed
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0) # Avoid div by 0 in calculation, mask later
    
    # d = dists
    # Term = sigma^12 / d^13
    # Force magnitude on i from j is 12 * sigma^12 / d^13
    # Direction is diff[i,j] / d
    # Force vector on i from j: 12 * sigma^12 / d^13 * (diff[i,j] / d) = 12 * sigma^12 * diff[i,j] / d^14
    
    # To avoid overflow/underflow, we can compute in log space or just clip distances.
    # Let's clip distances to a small epsilon to prevent explosion.
    d_safe = np.clip(dists, 1e-6, 100.0)
    
    # Coefficient for force: 12 * sigma^12 / d^14
    # Energy: sigma^12 / d^12
    
    sigma_pair = 0.25
    sigma_wall = 0.45 # Slightly less than 0.5 to allow some compression? No, 0.5 is the wall.
    # Actually, if we use d_wall, min distance is 0.
    
    # Calculate pairwise forces
    # F_ij = 12 * sigma_pair^12 * diff[i,j] / d_ij^14
    # We sum over j for each i.
    
    # d_safe^14
    d_pow_14 = d_safe**14
    d_pow_12 = d_safe**12
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # Force contribution matrix: F_mat[i,j] is force on i due to j
    # F_mat[i,j] = 12 * sigma_pair^12 * diff[i,j] / d_safe[i,j]^14
    # Note: diff[i,j] is vector.
    
    coeff = 12.0 * (sigma_pair**12) / d_pow_14
    coeff[~mask] = 0.0 # Zero out diagonal
    
    # Force on i is sum_j coeff[i,j] * diff[i,j]
    # diff shape (N, N, 3). coeff shape (N, N).
    # F_i = sum_j coeff[i,j] * diff[i,j, :]
    forces_pair = np.einsum('ij,ijk->ik', coeff, diff)
    
    # Energy
    energy_pair = np.sum((sigma_pair**12) / d_pow_12) / 2.0 # Divide by 2 because i<j and j<i
    # Wait, the sum in einsum includes all pairs.
    # Energy = 0.5 * sum_{i!=j} U(d_ij)
    energy_pair = 0.5 * np.sum((sigma_pair**12) / d_pow_12)
    
    # Wall interactions
    # Distance to nearest face
    # x in [-0.5, 0.5]
    # d_x = 0.5 - |x|
    d_wall_x = 0.5 - np.abs(centres[:, 0])
    d_wall_y = 0.5 - np.abs(centres[:, 1])
    d_wall_z = 0.5 - np.abs(centres[:, 2])
    
    # Clip to avoid div by 0
    d_wall_x = np.clip(d_wall_x, 1e-6, 1.0)
    d_wall_y = np.clip(d_wall_y, 1e-6, 1.0)
    d_wall_z = np.clip(d_wall_z, 1e-6, 1.0)
    
    sigma_w = 0.45
    eps_w = 1.0 # Scale factor for wall force
    
    # Force from wall: F = -12 * sigma_w^12 / d^13 * sign(x)
    # If x > 0, d = 0.5 - x. d/dx d = -1. F_x = -dU/dx.
    # U = (sigma_w/d)^12. dU/dx = dU/dd * dd/dx = (-12 sigma_w^12/d^13) * (-1) = 12 sigma_w^12/d^13.
    # F_x = -12 sigma_w^12/d^13. (Negative, pushes to 0).
    # If x < 0, d = 0.5 + x. dd/dx = 1. dU/dx = -12 sigma_w^12/d^13.
    # F_x = 12 sigma_w^12/d^13. (Positive, pushes to 0).
    # So F = -12 * sigma_w^12 / d^13 * sign(x)
    
    F_wx = -12.0 * (sigma_w**12) / (d_wall_x**13) * np.sign(centres[:, 0]) * eps_w
    F_wy = -12.0 * (sigma_w**12) / (d_wall_y**13) * np.sign(centres[:, 1]) * eps_w
    F_wz = -12.0 * (sigma_w**12) / (d_wall_z**13) * np.sign(centres[:, 2]) * eps_w
    
    forces_wall = np.column_stack((F_wx, F_wy, F_wz))
    
    total_forces = forces_pair + forces_wall
    total_energy = energy_pair + np.sum((sigma_w**12) / (d_wall_x**12)) + np.sum((sigma_w**12) / (d_wall_y**12)) + np.sum((sigma_w**12) / (d_wall_z**12))
    
    return total_energy, total_forces

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    np.random.seed(42)
    
    # Initial configuration: Random or Grid?
    # Grid is better for initialization to avoid immediate high energy.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    np.random.shuffle(grid)
    centres = grid[:N].copy()
    
    best_centres = centres.copy()
    best_r = _compute_radius(centres)
    
    # Annealing Schedule
    # We will run several phases.
    # Phase 1: High T, large steps.
    # Phase 2: Medium T.
    # Phase 3: Low T, small steps.
    
    # Parameters
    sigma = 0.25
    dt = 0.001
    
    # We need to tune dt and T.
    # Force can be huge. We should normalize forces or use adaptive dt.
    
    def run_md(centres, temp, steps, dt_scale=1.0):
        c = centres.copy()
        for i in range(steps):
            # Check time
            if time.time() > deadline - 5.0:
                break
            
            energy, forces = _lj_potential_and_force(c, sigma=0.25)
            
            # Adaptive dt: limit max displacement
            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force > 1000:
                # Reduce dt or clip forces
                forces = forces / (max_force / 100.0)
                max_force = 100.0
            
            # Velocity Verlet or simple Euler?
            # Simple Euler with damping is more stable for this.
            # v = v * damping + force * dt
            # x = x + v * dt
            # Let's use a simple gradient descent step with noise (Langevin).
            # x_new = x + force * dt + noise * sqrt(temp)
            
            # Normalize forces to prevent explosion
            # The potential is very stiff.
            # Let's use a fixed small step size for the force part.
            step_size = 0.005 * dt_scale
            
            # Random perturbation
            noise = np.random.randn(*c.shape) * np.sqrt(temp) * 0.01
            
            c = c + forces * step_size + noise
            
            # Keep inside cube (hard constraint)
            # If a particle goes out, reflect it or clip it.
            # Clipping might cause sticking. Reflection is better.
            # But for simplicity, clip and rely on wall force to push back.
            # Actually, if it clips to 0.5, wall force is huge, pushing it back.
            c = np.clip(c, -0.499, 0.499)
            
        return c

    # Phase 1: Coarse optimization
    # High temp to escape local minima
    current_r = best_r
    for phase in range(3):
        if time.time() > deadline - 20.0:
            break
            
        # Define schedule
        if phase == 0:
            temps = [0.5, 0.3, 0.1]
            steps_per_temp = 200
            dt_scale = 1.0
        elif phase == 1:
            temps = [0.1, 0.05, 0.01]
            steps_per_temp = 500
            dt_scale = 0.5
        else:
            temps = [0.01, 0.001, 0.0001]
            steps_per_temp = 1000
            dt_scale = 0.1
            
        for T in temps:
            if time.time() > deadline - 10.0:
                break
                
            # Run MD
            new_centres = run_md(best_centres, T, steps_per_temp, dt_scale)
            
            # Check improvement
            new_r = _compute_radius(new_centres)
            if new_r > best_r:
                best_r = new_r
                best_centres = new_centres.copy()
            else:
                # If we didn't improve, maybe we should try to recover from a bad move?
                # Or just keep the best.
                pass
        
        # Between phases, try a random restart if stuck?
        # Let's try to perturb the best solution and run a short high-T phase.
        if phase < 2 and time.time() < deadline - 30.0:
            perturbed = best_centres + np.random.randn(N, 3) * 0.05
            perturbed = np.clip(perturbed, -0.49, 0.49)
            # Short high T run
            recovered = run_md(perturbed, 0.5, 100, 1.0)
            rec_r = _compute_radius(recovered)
            if rec_r > best_r:
                best_r = rec_r
                best_centres = recovered.copy()
    
    # Final Polish: Local Search
    # Use a direct nudge method on the best configuration found by MD.
    # This is the _refine_local from the previous version, which is good for final steps.
    
    def _refine_local(centres, max_iter=2000):
        centres = centres.copy()
        n = len(centres)
        r = _compute_radius(centres)
        
        for _ in range(max_iter):
            if time.time() > deadline - 2.0:
                break
                
            current_r = _compute_radius(centres)
            
            # Find the minimum constraint
            wall_dists = 0.5 - np.max(np.abs(centres), axis=1)
            min_wall_idx = np.argmin(wall_dists)
            min_wall_val = wall_dists[min_wall_idx]
            
            diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
            min_pair_val = dists[min_pair_idx] / 2.0
            
            if min_wall_val < min_pair_val:
                i = min_wall_idx
                c = centres[i]
                abs_c = np.abs(c)
                axis = np.argmax(abs_c)
                
                step = 0.0005
                centres[i, axis] -= np.sign(c[axis]) * step
                centres[i, axis] = np.clip(centres[i, axis], -0.5, 0.5)
                
            else:
                i, j = min_pair_idx
                if i == j: continue
                
                v = centres[i] - centres[j]
                dist = np.linalg.norm(v)
                if dist < 1e-10:
                    centres[i] += np.random.rand(3) * 0.001
                    centres[j] -= np.random.rand(3) * 0.001
                else:
                    u = v / dist
                    step = 0.0005
                    centres[i] += u * step
                    centres[j] -= u * step
                    centres[i] = np.clip(centres[i], -0.5, 0.5)
                    centres[j] = np.clip(centres[j], -0.5, 0.5)
            
            new_r = _compute_radius(centres)
            if new_r > r:
                r = new_r
            else:
                # Stuck? Small random kick
                if np.random.rand() < 0.05:
                    centres[min_wall_idx] += np.random.randn(3) * 0.0002
                    centres[min_pair_idx[0]] += np.random.randn(3) * 0.0002
                    centres[min_pair_idx[1]] += np.random.randn(3) * 0.0002
                    centres = np.clip(centres, -0.5, 0.5)
                    
        return centres

    if time.time() < deadline - 5.0:
        final_centres = _refine_local(best_centres, max_iter=5000)
        final_r = _compute_radius(final_centres)
        if final_r > best_r:
            best_r = final_r
            best_centres = final_centres.copy()
            
    return best_centres

# EVOLVE-BLOCK-END
