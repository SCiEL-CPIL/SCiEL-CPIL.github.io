"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_r(centres):
    """Compute the maximum common radius for given centres."""
    if len(centres) == 0:
        return 0.0
    # Distance to boundary
    boundary = 0.5 - np.abs(centres)
    r_boundary = np.min(boundary)
    # Pairwise distances
    if len(centres) > 1:
        # Use a smarter approach: only compute if needed, but for 100 points it's fine
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
    else:
        r_pair = r_boundary
    return min(r_boundary, r_pair)


def _initial_grid(n):
    """Create a simple cubic grid initialisation."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]


def _repulsion_energy(centres, r):
    """Compute a repulsion energy that is high when constraints are violated."""
    # Boundary violation
    boundary_viol = np.maximum(0, np.abs(centres) + r - 0.5)
    boundary_energy = np.sum(boundary_viol ** 2)
    
    # Pairwise violation
    if len(centres) > 1:
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        # Only count violations where dist < 2r
        violations = np.maximum(0, 2 * r - dists)
        pair_energy = np.sum(violations ** 2)
    else:
        pair_energy = 0.0
    return boundary_energy + pair_energy


def _gradient_ascent_step(centres, r, step_size, max_iter=100):
    """Perform gradient ascent on r while adjusting centres to maintain feasibility."""
    # This is a simplified version; a full implementation would use more sophisticated methods
    pass


def construct():
    """
    Construct centres of 100 equal spheres in a unit cube to maximize common radius.
    Uses a combination of grid initialisation and local optimisation via 
    repulsion-based gradient descent.
    """
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    # Start with a grid
    centres = _initial_grid(N).copy()
    best_centres = centres.copy()
    best_r = _compute_r(centres)
    
    # Local optimisation using repulsion-based gradient ascent
    # We'll use a simple iterative approach:
    # 1. Compute current radius
    # 2. Move points apart using repulsion forces
    # 3. Project back to feasible region
    # 4. Repeat until convergence or time limit
    
    r = best_r
    step_size = 0.01
    learning_rate = 0.005
    
    for iteration in range(10000):
        # Check time
        if time.time() - start_time > deadline - 5.0:
            break
        
        # Compute repulsion forces
        forces = np.zeros_like(centres)
        
        # Boundary repulsion: push points away from boundaries
        for i in range(N):
            for dim in range(3):
                if centres[i, dim] > 0:
                    forces[i, dim] -= 1.0  # push left
                else:
                    forces[i, dim] += 1.0  # push right
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = centres[i] - centres[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random perturbation
                    diff = np.random.randn(3) * 0.01
                    dist = np.linalg.norm(diff)
                # Repulsion strength increases as distance decreases
                repulsion = 1.0 / (dist + 0.1)
                forces[i] += repulsion * diff / dist
                forces[j] -= repulsion * diff / dist
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.maximum(force_norms, 1e-10)
        forces = forces / force_norms
        
        # Update centres
        centres = centres + learning_rate * forces
        
        # Project to feasible region (keep inside cube with margin for radius)
        # For now, just clip to [-0.5, 0.5]
        centres = np.clip(centres, -0.5, 0.5)
        
        # Compute new radius
        new_r = _compute_r(centres)
        
        if new_r > best_r:
            best_r = new_r
            best_centres = centres.copy()
            # Increase learning rate slightly
            learning_rate = min(learning_rate * 1.01, 0.02)
        else:
            # Decrease learning rate
            learning_rate = max(learning_rate * 0.95, 0.001)
    
    return best_centres
# EVOLVE-BLOCK-END
