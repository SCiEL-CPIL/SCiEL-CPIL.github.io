"""
Packing 200 equal circles in a unit square using soft-contact rigid-body dynamics.
"""
import numpy as np

N = 200


def construct():
    """
    Construct the centers of 200 equal circles in a unit square [-0.5, 0.5]^2
    using a symplectic Euler integrator with a soft-contact potential.
    """
    
    def hexagonal_init(n_circles):
        """Generate a hexagonal lattice initialization."""
        # Estimate optimal radius to set initial grid spacing
        # Area per circle approx 1/N = 0.005
        # Hex packing density ~0.9069
        # Area of circle = pi * r^2
        # pi * r^2 / 0.9069 ~ 1/N
        # r ~ sqrt(N / (pi * 0.9069)) ~ sqrt(200 / 2.84) ~ sqrt(70) ~ 8.4 ... wait
        # 1/N = 0.005. r^2 ~ 0.005 * 0.9 / pi ~ 0.0014. r ~ 0.037.
        # This matches the known best radius ~0.0366.
        
        r_est = 0.036
        # Hex grid parameters
        # dx = 2 * r_est
        # dy = r_est * sqrt(3)
        dx = 2 * r_est
        dy = r_est * np.sqrt(3)
        
        # Number of rows and cols
        # Width ~ 1, Height ~ 1
        # cols ~ 1 / dx ~ 1 / 0.072 ~ 14
        # rows ~ 1 / dy ~ 1 / 0.062 ~ 16
        # 14 * 16 = 224 > 200.
        
        points = []
        row = 0
        while len(points) < n_circles:
            offset = 0 if row % 2 == 0 else r_est
            col = 0
            while len(points) < n_circles:
                x = -0.5 + r_est + offset + col * dx
                y = -0.5 + r_est + row * dy
                # Only add if within bounds (with some margin)
                if -0.5 < x < 0.5 and -0.5 < y < 0.5:
                    points.append([x, y])
                col += 1
            row += 1
        
        pts = np.array(points[:n_circles])
        return pts

    # Initial positions
    centers = hexagonal_init(N)
    
    # Simulation parameters
    dt = 0.001  # Time step
    damping = 0.98  # Velocity damping factor
    k_repel = 10.0  # Repulsion strength
    k_bound = 100.0  # Boundary strength
    n_steps = 20000
    eps = 1e-6  # Small epsilon to avoid division by zero
    
    # State
    positions = centers.copy()
    velocities = np.zeros_like(positions)
    
    # Precompute indices for pairs? N=200 is small, O(N^2) is fine.
    # 200*200 = 40000 pairs. 20000 steps * 40000 ops = 800M ops.
    # This might be slow in Python. We need to vectorize.
    
    # Vectorized force calculation
    # positions shape (N, 2)
    # diff[i, j] = positions[i] - positions[j]
    
    for step in range(n_steps):
        # Calculate pairwise forces
        # diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        # This creates a 200x200x2 array. 200*200*2 = 80000 floats.
        # Doing this 20000 times is heavy but NumPy should handle it.
        
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq + eps)
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(dists, 1.0)
        
        # Force magnitude: F = k / d^2
        # Direction: diff / d
        # F_vec = (k / d^2) * (diff / d) = k * diff / d^3
        
        # We want to repel. If i is to the right of j, force on i is to the right?
        # diff[i,j] = pos[i] - pos[j]. If pos[i] > pos[j], diff is positive.
        # Force on i should be in direction of diff (away from j).
        # So F_i += k * diff[i,j] / d^3
        
        inv_d3 = 1.0 / (dists**3)
        # Set diagonal to 0
        np.fill_diagonal(inv_d3, 0.0)
        
        forces = k_repel * diff * inv_d3[:, :, np.newaxis]
        
        # Sum forces over j for each i
        # forces shape (N, N, 2). Sum over axis 1.
        net_forces = np.sum(forces, axis=1)
        
        # Boundary forces
        # Push back if outside [-0.5, 0.5]
        # Let's use a soft boundary: force proportional to penetration depth
        # Or just clamp and zero velocity?
        # Soft boundary is better for dynamics.
        # If x > 0.5, F_x = -k_bound * (x - 0.5)
        # If x < -0.5, F_x = -k_bound * (x - (-0.5)) = -k_bound * (x + 0.5)
        
        bound_forces = np.zeros_like(positions)
        
        # Right boundary
        mask_r = positions[:, 0] > 0.5
        bound_forces[mask_r, 0] -= k_bound * (positions[mask_r, 0] - 0.5)
        
        # Left boundary
        mask_l = positions[:, 0] < -0.5
        bound_forces[mask_l, 0] -= k_bound * (positions[mask_l, 0] + 0.5)
        
        # Top boundary
        mask_t = positions[:, 1] > 0.5
        bound_forces[mask_t, 1] -= k_bound * (positions[mask_t, 1] - 0.5)
        
        # Bottom boundary
        mask_b = positions[:, 1] < -0.5
        bound_forces[mask_b, 1] -= k_bound * (positions[mask_b, 1] + 0.5)
        
        # Total force
        total_forces = net_forces + bound_forces
        
        # Symplectic Euler
        # v_new = v_old + a * dt
        # x_new = x_old + v_new * dt
        
        velocities += total_forces * dt
        velocities *= damping  # Damping
        positions += velocities * dt
        
        # Hard clip to ensure they don't fly away too much (safety)
        # This might introduce artifacts, but prevents explosion
        positions = np.clip(positions, -0.51, 0.51)
        
    # Final cleanup: ensure inside bounds
    centers_final = np.clip(positions, -0.5, 0.5)
    
    return centers_final


# Quick test
if __name__ == "__main__":
    centers = construct()
    print(f"Shape: {centers.shape}")
    # Calculate radius
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2
    
    bound_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
        np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
    )
    min_bound = np.min(bound_dists)
    
    r = min(min_pair, min_bound)
    print(f"Radius: {r:.10f}")
