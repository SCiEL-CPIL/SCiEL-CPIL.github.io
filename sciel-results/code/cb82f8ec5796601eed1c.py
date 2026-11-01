"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50
DEADLINE = 140  # Safety margin before 150s limit

def _compute_radius(c):
    """Compute the common radius for a given set of centres."""
    # Pairwise distances
    d = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _repulsion_step(c, step_size, repulsion_strength=1.0, wall_repulsion=10.0):
    """
    Perform a single step of the repulsion algorithm.
    """
    n = len(c)
    forces = np.zeros_like(c)
    
    # Pairwise repulsion
    # d[i, j] is vector from j to i
    d = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(d * d, axis=-1)
    dists_sq[dist_sq < 1e-12] = 1e-12 # Avoid division by zero, though we check dists later
    
    # We want to push apart. Force is proportional to 1/d^2 or similar.
    # Let's use a smooth potential derivative.
    # F = -grad U. If U ~ 1/d, F ~ 1/d^2.
    # To avoid singularities, we can use a softened distance.
    
    # Calculate magnitude of force for each pair
    # We only want to push if they are close, but actually pushing all pairs helps packing.
    # A common choice is F = 1 / (d^2 + epsilon)
    
    inv_dists_sq = 1.0 / (dists_sq + 1e-6)
    # Force vector from j to i is d_ij * magnitude
    # We sum forces from all j != i
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    inv_dists_sq[~mask] = 0.0
    
    # Force on i is sum over j of (c_i - c_j) / dist^2
    # This is equivalent to (c_i * sum(1/d^2) - sum(c_j / d^2))
    # But simpler to just multiply vector by scalar and sum
    # forces += d * inv_dists_sq[:, :, np.newaxis]
    
    # d has shape (n, n, 3)
    # inv_dists_sq has shape (n, n)
    # We want to sum over axis 1 (j)
    forces += np.sum(d * inv_dists_sq[:, :, np.newaxis], axis=1)
    
    # Wall repulsion
    # Push away from walls if close
    # Distance to wall: 0.5 - |c_k|
    # If dist < threshold, push inwards
    # Let's define a potential that rises sharply near walls
    # Force in x: if c_x > 0, wall is at 0.5. dist = 0.5 - c_x.
    # If dist < r_target, push left.
    # We want to maximize min wall distance.
    # Let's use a similar repulsion from the "virtual" walls outside the box.
    # Or simply: if c_x > 0.5 - margin, push left.
    
    # A smooth approach: 
    # F_x = wall_repulsion * (c_x / (0.5 - c_x)^2) for c_x > 0
    # This is complex. Simpler:
    # If c_x > 0, dist_to_right = 0.5 - c_x.
    # If c_x < 0, dist_to_left = 0.5 + c_x.
    # We want to push towards center if too close to wall.
    
    # Let's use a simple linear penalty for being in the "exclusion zone" near walls
    # Or better, treat walls as infinite mass spheres? No.
    
    # Let's use: F_wall = k * (c / (0.5 - |c|)^2) * sign(c) ... no, that's unstable.
    
    # Standard approach for packing in a box:
    # Add force towards center proportional to how close to wall.
    # F_x = -k * c_x * (1 / (0.5 - |c_x|)^2) ?
    
    # Let's try: 
    # For each coordinate k:
    # d_k = 0.5 - abs(c_ik)
    # if d_k < 0.2: # Threshold
    #    force_k += sign(c_ik) * wall_repulsion / (d_k^2 + 1e-6)
    
    # Vectorized:
    abs_c = np.abs(c)
    dist_to_wall = 0.5 - abs_c
    # We only want to push if close. Let's say if dist < 0.25
    # Actually, to maximize min wall dist, we should always push slightly towards center
    # to balance with pair repulsion.
    
    # Let's use a potential U_wall = 1 / (0.5 - |c|)^2
    # Gradient w.r.t c_k:
    # d/dc ( (0.5 - |c|)^-2 ) = -2 * (0.5 - |c|)^-3 * d/dc(0.5 - |c|)
    # d/dc |c| = sign(c)
    # d/dc (0.5 - |c|) = -sign(c)
    # So grad = -2 * (0.5 - |c|)^-3 * (-sign(c)) = 2 * sign(c) / (0.5 - |c|)^3
    # Force is -grad U = -2 * sign(c) / (0.5 - |c|)^3
    # This pushes towards center (opposite to sign(c)).
    
    # This force becomes very strong near walls, which is good.
    # Let's scale it.
    denom = dist_to_wall**3 + 1e-6
    sign_c = np.sign(c)
    wall_forces = -2.0 * wall_repulsion * sign_c / denom
    
    # Only apply if close? The function 1/d^3 is already very strong.
    # But far from wall, d=0.5, force is small.
    # At d=0.1, force is 1000x larger.
    # This seems reasonable.
    
    forces += wall_forces
    
    # Update positions
    c_new = c + step_size * forces
    
    # Keep inside box (hard constraint)
    c_new = np.clip(c_new, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return c_new

def construct():
    start_time = time.time()
    best_c = None
    best_r = 0.0
    
    # Multiple restarts
    n_restarts = 10
    
    for i in range(n_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Initialization
        if i == 0:
            # Start with a grid
            x = np.linspace(-0.4, 0.4, 5)
            y = np.linspace(-0.4, 0.4, 5)
            z = np.array([-0.25, 0.25])
            points = []
            for zi in z:
                for yi in y:
                    for xi in x:
                        points.append([xi, yi, zi])
            c = np.array(points)
        else:
            # Random initialization
            c = np.random.uniform(-0.45, 0.45, size=(N, 3))
            
        # Simulation parameters
        # Start with high "temperature" (large step) and cool down
        steps = 1000
        initial_step = 0.05
        cooling_rate = 0.99
        
        current_step = initial_step
        best_c_local = c.copy()
        best_r_local = _compute_radius(c)
        
        for step in range(steps):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            c = _repulsion_step(c, current_step, repulsion_strength=1.0, wall_repulsion=1.0)
            current_step *= cooling_rate
            
            # Check improvement
            r_current = _compute_radius(c)
            if r_current > best_r_local:
                best_r_local = r_current
                best_c_local = c.copy()
                
            # If we have a very good radius, we can stop early for this restart
            # The known best is ~0.1359. If we get close, refine.
            if best_r_local > 0.135:
                 # Do a few more steps with very small step size
                 for _ in range(100):
                     if time.time() - start_time > DEADLINE - 5:
                         break
                     c = _repulsion_step(c, current_step * 0.1, repulsion_strength=1.0, wall_repulsion=1.0)
                     r_check = _compute_radius(c)
                     if r_check > best_r_local:
                         best_r_local = r_check
                         best_c_local = c.copy()
                 break

        if best_r_local > best_r:
            best_r = best_r_local
            best_c = best_c_local.copy()
            
        # If we found a very good solution, we might break
        if best_r > 0.1359:
            # Try to polish further
            break

    # Final polish on the best solution found
    if best_c is not None and time.time() - start_time < DEADLINE - 10:
        c = best_c.copy()
        # Very small steps to refine
        for _ in range(500):
            if time.time() - start_time > DEADLINE - 2:
                break
            c = _repulsion_step(c, 1e-4, repulsion_strength=1.0, wall_repulsion=1.0)
            r_check = _compute_radius(c)
            if r_check > best_r:
                best_r = r_check
                best_c = c.copy()
            else:
                # If it got worse, revert and stop? Or just keep going?
                # Let's just keep the best.
                pass

    # Ensure validity
    if best_c is None:
        # Fallback: grid
        x = np.linspace(-0.4, 0.4, 5)
        y = np.linspace(-0.4, 0.4, 5)
        z = np.array([-0.25, 0.25])
        points = []
        for zi in z:
            for yi in y:
                for xi in x:
                    points.append([xi, yi, zi])
        best_c = np.array(points)
        
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)
    return best_c
