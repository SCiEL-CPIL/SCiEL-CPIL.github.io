"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    # Rigid-body dynamics with smooth repulsive potential, symplectic (velocity Verlet) integration.
    # Objective: maximize r = min(min_{i<j} |ci-cj|/2, min_i (1-|ci|)).
    # We use a soft repulsive potential that penalizes overlaps and boundary violations.
    # The potential is smooth and differentiable, avoiding numerical stiffness of hard constraints.
    
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Initial placement: hexagonal lattice cut to unit disk
    def hex_lattice_init(n):
        spacing = 2.0 / (np.sqrt(n) + 1.0)
        while True:
            k = int(1.0 / spacing) + 3
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= n:
                return points[:n].copy()
            spacing *= 0.97
    
    pos = hex_lattice_init(N)
    vel = np.zeros_like(pos)
    
    # Compute current packing radius
    def compute_radius(positions):
        # Boundary constraint
        dist_to_center = np.hypot(positions[:, 0], positions[:, 1])
        boundary_min = 1.0 - dist_to_center
        
        # Pairwise distances
        # Use vectorized computation for efficiency
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        # Set diagonal to infinity
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
        
        return min(boundary_min.min(), pair_min)
    
    current_r = compute_radius(pos)
    best_pos = pos.copy()
    best_r = current_r
    
    # Symplectic integration parameters
    dt = 0.01
    temperature = 0.01  # Controls strength of repulsive forces
    cooling_rate = 0.999  # Annealing schedule
    
    # Number of steps per iteration
    steps_per_iter = 50
    
    for it in range(1000000):
        if time.time() > deadline:
            break
        
        # Compute forces from smooth repulsive potential
        # Potential: U = sum_{i<j} max(0, d_min - |ci-cj|)^2 + sum_i max(0, |ci| - (1-r_target))^2
        # We want to push circles apart and keep them inside the container
        
        # For efficiency, compute forces in batches
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1.0)  # Avoid division by zero
        
        # Repulsive force between pairs: F = k * (d_min - dist) * (dist_vec / dist) if dist < d_min
        # Use a smooth approximation: F = k * exp(-(dist - d_min)/sigma) * (dist_vec / dist)
        # But for stability, use a simpler quadratic penalty
        
        # Target minimum distance based on current best radius
        target_d = 2.0 * best_r + 0.001
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < target_d:
                    # Quadratic penalty force
                    force_mag = 100.0 * (target_d - d)
                    force_vec = diff[i, j] / d
                    forces[i] += force_mag * force_vec
                    forces[j] -= force_mag * force_vec
        
        # Boundary repulsion: push inward if too close to boundary
        dist_to_center = np.hypot(pos[:, 0], pos[:, 1])
        boundary_limit = 1.0 - best_r - 0.001
        for i in range(N):
            if dist_to_center[i] > boundary_limit:
                # Force toward center
                force_mag = 100.0 * (dist_to_center[i] - boundary_limit)
                force_vec = -pos[i] / dist_to_center[i]
                forces[i] += force_mag * force_vec
        
        # Velocity Verlet integration
        # Update velocity half-step
        vel += 0.5 * dt * forces
        # Update position
        pos += dt * vel
        # Recompute forces (approximate: reuse forces for simplicity, or recompute)
        # For better accuracy, recompute forces at new positions
        # But for speed, use same forces
        vel += 0.5 * dt * forces
        
        # Damping to stabilize
        vel *= 0.95
        
        # Cool down
        temperature *= cooling_rate
        
        # Check current radius
        current_r = compute_radius(pos)
        if current_r > best_r:
            best_r = current_r
            best_pos = pos.copy()
        
        # Occasionally, add small random perturbation to escape local minima
        if it % 1000 == 0:
            pos += 0.001 * np.random.randn(*pos.shape)
            vel *= 0.5
    
    return best_pos
# EVOLVE-BLOCK-END
