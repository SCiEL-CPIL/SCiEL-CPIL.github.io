import numpy as np

N = 50

def construct():
    """
    Constructs the positions of 50 equal spheres in a unit cube [-0.5, 0.5]^3
    using a physics-based repulsion simulation.
    """
    def simulate(x0, steps=20000, dt=0.005, cooling=0.9999):
        """
        Runs the repulsion simulation starting from initial positions x0.
        """
        positions = x0.copy()
        # Velocity for momentum (optional, but helps escape shallow local minima)
        velocities = np.zeros_like(positions)
        
        # We want to maximize the minimum distance. 
        # We can simulate forces that push spheres apart when they are close.
        # A common potential is V(r) = 1/r^alpha for r < r0.
        # Force is -grad V.
        
        # To ensure they stay inside the cube, we add strong repulsion from walls.
        
        for step in range(steps):
            # Compute pairwise distances
            # diff[i, j] = positions[i] - positions[j]
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            
            # Avoid division by zero
            np.fill_diagonal(dist_sq, 1.0)
            dist = np.sqrt(dist_sq)
            
            # Compute forces between spheres
            # We use a force that is very strong at short distances.
            # F_ij = k * (1/dist^2) * (pos_i - pos_j) / dist
            # This is equivalent to gradient of 1/dist
            # Let's use a softer potential to ensure stability: V = 1/(dist^2)
            # F = 2 / dist^3 * diff
            
            # Mask diagonal to zero force
            mask = np.ones((N, N), dtype=bool)
            np.fill_diagonal(mask, False)
            
            # Force magnitude scaling
            # We want the force to be significant when dist is small.
            # If dist is large, force is negligible.
            # Let's define force only for dist < some threshold, or just let 1/dist^3 decay.
            
            force_magnitudes = 2.0 / (dist**3)
            force_magnitudes[~mask] = 0.0
            
            # Vector force
            # F_ij on i is force_magnitudes[i,j] * diff[i,j]
            forces = force_magnitudes[:, :, np.newaxis] * diff
            
            # Sum forces on each sphere
            total_force = np.sum(forces, axis=1)
            
            # Wall repulsion
            # Distance to wall: 0.5 - |pos|
            # If pos is close to wall, push it back towards center.
            # Force direction: -sign(pos)
            # Magnitude: proportional to (0.5 - |pos|)^-2 or similar
            # Let's use a potential V_wall = 1 / (0.5 - |pos|)^2
            # Force = -dV/dpos = - 2 * (0.5 - |pos|)^-3 * d(0.5-|pos|)/dpos
            # d(0.5 - |x|)/dx = -sign(x)
            # So Force = - 2 * (0.5 - |pos|)^-3 * (-sign(pos)) = 2 * sign(pos) / (0.5 - |pos|)^3
            
            # Check bounds to avoid singularity
            # If pos is exactly 0.5, dist to wall is 0. We must handle this.
            # We will clamp positions slightly inside before computing wall force.
            # Actually, better to add a constant offset to distance to wall to avoid inf.
            # Let d_wall = 0.5 - |pos|
            # If d_wall < 0, it's outside. We want to push it back.
            # Let's use a smooth wall potential.
            
            # Simple approach: 
            # If pos > 0.49, apply force -k * (pos - 0.49)^2
            # If pos < -0.49, apply force k * (pos + 0.49)^2
            # This is a harmonic potential outside the soft box [-0.49, 0.49].
            # This is robust.
            
            wall_force = np.zeros_like(positions)
            # Right/Top/Back walls (positive side)
            mask_pos = positions > 0.49
            wall_force[mask_pos] -= 10.0 * (positions[mask_pos] - 0.49)**2
            # Left/Bottom/Front walls (negative side)
            mask_neg = positions < -0.49
            wall_force[mask_neg] += 10.0 * (positions[mask_neg] + 0.49)**2
            
            total_force += wall_force
            
            # Update velocity and position
            # Simple Euler with damping
            # v = v * damping + force * dt
            # x = x + v * dt
            damping = 0.9
            velocities = velocities * damping + total_force * dt
            positions += velocities * dt
            
            # Cool down: reduce step size or force strength?
            # Reducing dt helps stability.
            # Or we can just let the forces settle.
            # For cooling, we can reduce the magnitude of the repulsive force over time?
            # No, we want the final state to be the one with max min dist.
            # The equilibrium of 1/r forces is a local max of min dist.
            # We just need to converge.
            
            # Optional: Add small random noise to escape local minima
            # noise = 0.001 * np.random.normal(0, 1, positions.shape)
            # positions += noise
            
        # Final clipping to ensure strict bounds
        positions = np.clip(positions, -0.5, 0.5)
        return positions

    def score(positions):
        """
        Computes the radius r.
        """
        # Pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_dist = np.sqrt(np.min(dist_sq)) / 2.0
        
        # Wall distances
        # min_i min_k (0.5 - |c_ik|)
        # For each sphere, min distance to wall is 0.5 - max(|x|, |y|, |z|)
        max_abs_coord = np.max(np.abs(positions), axis=1)
        min_wall_dist = np.min(0.5 - max_abs_coord)
        
        return min(min_pair_dist, min_wall_dist)

    best_r = 0.0
    best_positions = None
    
    # Multiple restarts
    rng = np.random.RandomState(42)
    
    for restart in range(5):
        # Initialize with random positions
        # To help convergence, we can start with a grid or random.
        # Random is better for finding global-ish optima with physics.
        x0 = rng.uniform(-0.4, 0.4, (N, 3))
        
        # Run simulation
        pos = simulate(x0, steps=10000, dt=0.005)
        
        r = score(pos)
        
        if r > best_r:
            best_r = r
            best_positions = pos

    return best_positions
