"""
Packing 200 equal circles in a unit square using soft-contact rigid-body dynamics.
"""
import numpy as np

N = 200


def construct():
    """
    Place N equal circles in unit square [-0.5, 0.5]^2 to maximize radius.
    Uses soft-contact rigid-body dynamics with symplectic Euler integrator.
    """
    
    def hexagonal_init(n_circles):
        """Generate a good initial hexagonal-like configuration."""
        # Try various hexagonal grid configurations
        best_points = None
        best_score = 0
        
        for rows in range(12, 22):
            for cols in range(12, 22):
                total = rows * cols
                if total < n_circles:
                    continue
                if total > n_circles + 20:  # Don't use too many extra
                    continue
                
                # Hexagonal grid: horizontal spacing 2r, vertical r*sqrt(3)
                # With offset for alternating rows
                # Fit in [-0.5, 0.5]^2
                # Width: (cols-1)*2r + r = (2*cols-1)*r <= 1
                # Height: (rows-1)*r*sqrt(3) <= 1
                r_max = min(1.0 / (2*cols - 1), 1.0 / ((rows - 1) * np.sqrt(3)))
                
                if r_max <= 0:
                    continue
                
                pts = []
                for j in range(rows):
                    for i in range(cols):
                        x = (i * 2 - (cols - 1)) * r_max
                        if j % 2 == 1:
                            x += r_max
                        y = (j - (rows - 1) / 2.0) * r_max * np.sqrt(3)
                        pts.append([x, y])
                
                pts = np.array(pts[:n_circles])
                
                # Check bounds
                margin = r_max * 0.05
                if np.all(np.abs(pts) <= 0.5 - margin):
                    # Score by minimum distance
                    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                    dists = np.sqrt(np.sum(diff**2, axis=2))
                    np.fill_diagonal(dists, np.inf)
                    min_pair = np.min(dists) / 2
                    
                    bound_dists = np.minimum(
                        np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
                        np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
                    )
                    min_bound = np.min(bound_dists)
                    
                    score = min(min_pair, min_bound)
                    if score > best_score:
                        best_score = score
                        best_points = pts
        
        if best_points is not None:
            return best_points
        
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(n_circles)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n_circles]
    
    def compute_radius(centers):
        """Compute the maximum valid radius for given centers."""
        if len(centers) < 2:
            return 0.5
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2
        
        bound_dists = np.minimum(
            np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
            np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
        )
        min_bound = np.min(bound_dists)
        
        return min(min_pair, min_bound)
    
    def soft_contact_forces(centers, r_target, stiffness=1000.0, boundary_stiffness=5000.0):
        """
        Compute forces from soft-contact potential.
        Potential: V = sum_{i<j} k * max(0, 2r - |c_i - c_j|)^4 / 4
                 + sum_i k_b * max(0, |c_ik| - (0.5 - r))^4 / 4
        Force on i from j: -dV/dc_i
        """
        n = len(centers)
        forces = np.zeros_like(centers)
        
        # Pairwise forces
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    # Random small perturbation to break symmetry
                    diff = np.random.randn(2) * 1e-8
                    dist = np.linalg.norm(diff)
                
                overlap = 2 * r_target - dist
                if overlap > 0:
                    # F = -dV/dc_i = k * overlap^3 * (diff / dist)
                    force_mag = stiffness * overlap**3
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
        
        # Boundary forces
        for i in range(n):
            for k in range(2):
                coord = centers[i, k]
                # Distance to boundary in coordinate k
                # Left boundary: 0.5 - (-0.5) = 1.0 from left edge to right edge
                # Actually, boundary is at +/-0.5
                # If coord > 0.5 - r, push left
                # If coord < -(0.5 - r), push right
                
                limit = 0.5 - r_target
                if coord > limit:
                    overlap = coord - limit
                    force_mag = boundary_stiffness * overlap**3
                    forces[i, k] -= force_mag
                elif coord < -limit:
                    overlap = -limit - coord
                    force_mag = boundary_stiffness * overlap**3
                    forces[i, k] += force_mag
        
        return forces
    
    def symplectic_euler_simulate(centers, velocities, r_target, n_steps=5000, 
                                  dt=0.001, damping=0.995, stiffness=1000.0,
                                  boundary_stiffness=5000.0, adaptive_dt=True):
        """
        Simulate using symplectic Euler method.
        v_{t+1} = v_t + dt * F(x_t)
        x_{t+1} = x_t + dt * v_{t+1}
        With damping: v_{t+1} *= damping
        """
        centers = centers.copy()
        velocities = velocities.copy()
        
        # Adaptive time step parameters
        dt_min = 1e-5
        dt_max = 0.01
        current_dt = dt
        
        for step in range(n_steps):
            # Compute forces
            forces = soft_contact_forces(centers, r_target, stiffness, boundary_stiffness)
            
            # Symplectic Euler update
            velocities = velocities + current_dt * forces
            velocities *= damping  # Damping
            centers = centers + current_dt * velocities
            
            # Adaptive time stepping: reduce dt if forces are large
            if adaptive_dt:
                max_force = np.max(np.linalg.norm(forces, axis=1))
                if max_force > 100:
                    current_dt = max(dt_min, current_dt * 0.5)
                elif max_force < 1:
                    current_dt = min(dt_max, current_dt * 1.1)
            
            # Safety: ensure centers stay within reasonable bounds
            centers = np.clip(centers, -0.6, 0.6)
        
        return centers, velocities
    
    # Initial configuration
    centers = hexagonal_init(N)
    r_init = compute_radius(centers)
    
    # Start with a slightly smaller target radius to allow convergence
    r_target = r_init * 0.95
    
    # Initialize velocities to zero
    velocities = np.zeros_like(centers)
    
    # Run simulation with increasing target radius
    best_centers = centers.copy()
    best_r = r_init
    
    n_phases = 5
    for phase in range(n_phases):
        # Increase target radius gradually
        target_r = r_init * (0.95 + 0.05 * (phase + 1) / n_phases)
        
        # Run simulation
        centers_sim, velocities_sim = symplectic_euler_simulate(
            centers, velocities, target_r,
            n_steps=3000,
            dt=0.002,
            damping=0.99,
            stiffness=500.0,
            boundary_stiffness=2000.0,
            adaptive_dt=True
        )
        
        r_sim = compute_radius(centers_sim)
        
        if r_sim > best_r:
            best_r = r_sim
            best_centers = centers_sim.copy()
        
        # Use simulated centers as starting point for next phase
        centers = centers_sim
        velocities = velocities_sim * 0.1  # Dampen velocities for next phase
    
    # Final fine-tuning: run with the best radius found
    r_final_target = best_r * 1.01
    centers_final, _ = symplectic_euler_simulate(
        best_centers, np.zeros_like(best_centers),
        r_final_target,
        n_steps=2000,
        dt=0.001,
        damping=0.995,
        stiffness=2000.0,
        boundary_stiffness=5000.0,
        adaptive_dt=True
    )
    
    r_final = compute_radius(centers_final)
    
    if r_final > best_r:
        best_r = r_final
        best_centers = centers_final.copy()
    
    return best_centers


# Quick test
if __name__ == "__main__":
    centers = construct()
    print(f"Shape: {centers.shape}")
    print(f"Radius: {compute_radius(centers):.10f}")
