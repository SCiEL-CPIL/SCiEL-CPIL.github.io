"""Packing 200 equal circles in a unit square using soft-contact rigid-body dynamics."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree


def construct():
    """
    Construct centres of N equal circles inside the square [-0.5, 0.5]^2
    maximizing the minimum radius.
    
    Approach: Rigid-body dynamics with soft-contact potential.
    - Initialize with a perturbed grid
    - Use symplectic Euler with adaptive time-stepping
    - Penalty for overlap (high-order polynomial) and wall violations
    - Add small random perturbations to escape local minima
    """
    # Parameters
    total_time = 50.0  # Total simulation time
    max_dt = 0.002
    min_dt = 1e-5
    stiffness = 100.0
    damping = 0.5
    repulsion_power = 4  # High-order polynomial for overlap penalty
    
    # Initialize with a grid plus small random perturbation
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    pos = grid.copy().astype(float)
    pos += np.random.randn(N, 2) * 0.001
    
    # Clamp to stay inside the square
    pos = np.clip(pos, -0.49, 0.49)
    
    vel = np.zeros((N, 2))
    
    # Adaptive time-stepping simulation
    t = 0.0
    dt = max_dt * 0.5
    
    while t < total_time:
        # Compute forces
        forces = np.zeros((N, 2))
        
        # Pairwise repulsion (soft contact)
        tree = cKDTree(pos)
        # Find all pairs within a certain distance
        pairs = tree.query_pairs(r=0.15, output_type='ndarray')
        
        for i, j in pairs:
            diff = pos[i] - pos[j]
            dist_sq = np.dot(diff, diff)
            dist = np.sqrt(dist_sq)
            
            if dist < 1e-10:
                # Coincident points - add random kick
                forces[i] += np.random.randn(2) * 0.1
                forces[j] -= np.random.randn(2) * 0.1
                continue
            
            # Overlap penalty: (1 - dist)^power when dist < 1
            # We want the circles to be at least 2r apart, but since r is variable,
            # we use a normalized distance. Let's use a target minimum distance.
            # Actually, for the dynamics, we just want to maximize the minimum distance.
            # Use a soft repulsion that grows rapidly when circles get close.
            
            # Scale: when dist < 0.07 (approx 2*0.035), strong repulsion
            threshold = 0.075
            if dist < threshold:
                overlap = 1.0 - dist / threshold
                # High-order polynomial potential
                force_mag = stiffness * (overlap ** repulsion_power)
                if dist > 1e-10:
                    force_dir = diff / dist
                    forces[i] += force_mag * force_dir
                    forces[j] -= force_mag * force_dir
        
        # Wall repulsion: keep circles inside [-0.5, 0.5]^2
        wall_margin = 0.05
        for i in range(N):
            for axis in range(2):
                coord = pos[i, axis]
                # Left wall
                if coord < -0.5 + wall_margin:
                    overlap = 1.0 - (coord + 0.5) / wall_margin
                    overlap = max(0, overlap)
                    force_mag = stiffness * (overlap ** repulsion_power) * 2
                    forces[i, axis] += force_mag
                # Right wall
                if coord > 0.5 - wall_margin:
                    overlap = 1.0 - (0.5 - coord) / wall_margin
                    overlap = max(0, overlap)
                    force_mag = stiffness * (overlap ** repulsion_power) * 2
                    forces[i, axis] -= force_mag
        
        # Symplectic Euler integration
        vel += forces * dt
        vel *= (1.0 - damping * dt)  # Damping
        pos += vel * dt
        
        # Clamp positions to stay inside the square
        pos = np.clip(pos, -0.499, 0.499)
        
        # Adaptive time-stepping: reduce dt if velocities are large
        max_vel = np.max(np.linalg.norm(vel, axis=1))
        if max_vel > 0.1:
            dt = min(dt, 0.01 / (max_vel + 1e-8))
        else:
            dt = min(max_dt, dt * 1.05)
        dt = max(min_dt, dt)
        
        t += dt
    
    # Final cleanup: ensure no coincident points and all inside square
    pos = np.clip(pos, -0.4999, 0.4999)
    
    # Remove any coincident points by small nudges
    for i in range(N):
        for j in range(i + 1, N):
            diff = pos[i] - pos[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                pos[i] += np.random.randn(2) * 1e-4
                pos[j] -= np.random.randn(2) * 1e-4
    
    pos = np.clip(pos, -0.4999, 0.4999)
    
    return pos
# EVOLVE-BLOCK-END
