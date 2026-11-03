"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    """
    Constructs the centers of 300 equal circles inside a unit circle.
    Uses Simulated Annealing followed by gradient-based refinement.
    """
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    def calc_radius(points):
        """Calculate the maximum radius for the given points."""
        # Pairwise distances
        # For N=300, 300x300 is 90k ops, very fast in numpy
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to boundary
        norm_points = np.linalg.norm(points, axis=1)
        min_bound_dist = np.min(1.0 - norm_points)
        
        r = min(min_pair_dist / 2.0, min_bound_dist)
        return r

    def get_initial_points():
        """Generate a good initial hexagonal grid."""
        # Estimate optimal spacing
        # Area of unit circle = pi. 300 circles.
        # Density ~ 0.9. Area per circle ~ pi/300 * 0.9 ~ 0.0094.
        # r ~ sqrt(0.0094/pi) ~ 0.055.
        # Spacing ~ 2*r ~ 0.11.
        
        # We'll generate a grid with spacing slightly smaller than ideal to ensure we have enough points,
        # then select the best N.
        
        # Try a few different grid densities to find a good start
        best_points = None
        best_r = 0
        
        for spacing_mult in [0.9, 1.0, 1.1]:
            # Base spacing estimate
            base_spacing = 0.11 * spacing_mult
            step = base_spacing
            
            # Generate hex grid
            points = []
            # Range needs to cover [-1, 1]
            num_x = int(2.0 / step) + 5
            num_y = int(2.0 / step) + 5
            
            for i in range(-num_x, num_x + 1):
                x = i * step
                # Hex offset
                if i % 2 == 0:
                    y_offset = 0
                else:
                    y_offset = step * np.sqrt(3) / 2
                
                for j in range(-num_y, num_y + 1):
                    y = j * step + y_offset
                    if x**2 + y**2 <= 1.0:
                        points.append((x, y))
            
            points = np.array(points)
            
            # If we have more than N, select the N closest to center? 
            # Or select to maximize min distance?
            # Simple heuristic: sort by distance to center, take first N.
            # This tends to fill the center, which is good.
            if len(points) > N:
                dists = np.hypot(points[:, 0], points[:, 1])
                indices = np.argsort(dists)
                points = points[indices[:N]]
            elif len(points) < N:
                # If not enough points, add random points inside
                missing = N - len(points)
                new_points = np.random.uniform(-0.9, 0.9, size=(missing, 2))
                # Ensure inside unit circle
                norms = np.hypot(new_points[:, 0], new_points[:, 1])
                mask = norms > 1.0
                while np.any(mask):
                    new_points[mask] = new_points[mask] / norms[mask][:, np.newaxis] * 0.9
                    norms = np.hypot(new_points[:, 0], new_points[:, 1])
                    mask = norms > 1.0
                points = np.vstack([points, new_points])
            
            # Add small noise to break symmetry
            points += np.random.normal(0, 0.001, size=points.shape)
            
            r = calc_radius(points)
            if r > best_r:
                best_r = r
                best_points = points.copy()
        
        return best_points

    def simulated_annealing(points, t_max, t_min, cooling_rate, max_iters):
        """
        Runs Simulated Annealing to improve the packing.
        """
        points = points.copy()
        current_r = calc_radius(points)
        best_points = points.copy()
        best_r = current_r
        
        T = t_max
        
        for i in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random particle
            idx = np.random.randint(N)
            
            # Propose a move: small random displacement
            # Scale step size with temperature
            step_size = 0.05 * (T / t_max) + 0.001
            delta = np.random.normal(0, step_size, size=2)
            
            new_point = points[idx] + delta
            
            # Check if new point is inside unit circle
            if np.linalg.norm(new_point) > 1.0:
                continue
            
            # Calculate new radius
            # We only need to check distances involving idx
            # But calc_radius is fast enough for N=300 (90k ops)
            # To speed up, we could compute only relevant distances, but let's stick to simple for now.
            # Optimization: 
            # old_min_pair = min(dists[idx, :]) excluding idx
            # old_min_bound = 1 - norm(points[idx])
            # old_r = min(old_min_pair/2, old_min_bound, other_min_pair/2, other_min_bound)
            # This is complex. Let's just use calc_radius.
            
            points[idx] = new_point
            new_r = calc_radius(points)
            
            # Acceptance criterion
            if new_r > current_r:
                current_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best_points = points.copy()
            else:
                # Metropolis criterion
                # We want to maximize r. So "energy" E = -r.
                # Delta E = -(new_r - current_r) = current_r - new_r
                # Accept if exp(-Delta E / T) > random
                delta_e = current_r - new_r
                if delta_e > 0:
                    prob = np.exp(-delta_e / T)
                    if np.random.random() < prob:
                        current_r = new_r
                    else:
                        points[idx] = points[idx] - delta # Revert
                # If delta_e <= 0, it was accepted above.
            
            # Cool down
            T *= cooling_rate
            if T < t_min:
                T = t_min
            
            # Occasionally, do a local gradient step to escape plateaus?
            # Or just rely on SA.
        
        return best_points, best_r

    def gradient_refine(points, max_iters=100):
        """
        Refine the configuration using gradient descent on the radius.
        """
        points = points.copy()
        current_r = calc_radius(points)
        
        # We will maximize r.
        # Gradient of min is tricky. We can use the "smallest distance" pair and "boundary" constraint.
        # Simpler: Use the same force-directed method as the original, but with adaptive lr.
        
        lr = 0.05
        for i in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Identify the critical constraint
            min_pair_dist = np.min(dists)
            min_bound_dist = np.min(1.0 - np.linalg.norm(points, axis=1))
            
            if min_pair_dist / 2.0 < min_bound_dist:
                # Pairwise constraint is active
                # Find all pairs with distance close to min
                # Push them apart
                target_dist = min_pair_dist * 1.01 # Try to increase slightly
                overlaps = np.maximum(0, target_dist - dists)
                # Force
                dists_safe = np.where(dists > 1e-10, dists, 1e-10)
                unit_diff = diff / dists_safe[:, :, np.newaxis]
                forces = np.sum(overlaps[:, :, np.newaxis] * unit_diff, axis=1)
            else:
                # Boundary constraint is active
                # Push points inward
                norms = np.linalg.norm(points, axis=1)
                violations = np.maximum(0, norms - (1.0 - min_bound_dist * 0.99))
                unit_inward = -points / np.maximum(norms[:, np.newaxis], 1e-10)
                forces = violations[:, np.newaxis] * unit_inward
            
            points += lr * forces
            
            # Decay lr
            lr *= 0.95
            
            new_r = calc_radius(points)
            if new_r > current_r:
                current_r = new_r
            else:
                # Revert if it got worse? Or just stop?
                # In gradient descent for max, if we step and it gets worse, we should maybe reduce step.
                # For simplicity, we just continue, hoping the next step fixes it.
                # Better: keep best points.
                pass
        
        return points, calc_radius(points)

    # Main Execution
    # 1. Generate initial points
    points = get_initial_points()
    best_points = points.copy()
    best_r = calc_radius(best_points)
    
    # 2. Run Simulated Annealing
    # Parameters:
    # t_max: Initial temperature. Should be large enough to allow big moves.
    # t_min: Final temperature.
    # cooling_rate: How fast to cool.
    # max_iters: Total iterations.
    
    t_max = 0.5
    t_min = 1e-5
    cooling_rate = 0.999
    # Estimate iterations: (ln(t_max) - ln(t_min)) / ln(cooling_rate)
    # ln(0.5) - ln(1e-5) = -0.69 - (-11.5) = 10.8
    # ln(0.999) = -0.001
    # Iters ~ 10800.
    # Each iter is ~100us (calc_radius + overhead). 10800 * 100us = 1s.
    # We have 110s. We can do multiple restarts or more iterations.
    
    # Let's do 5 restarts with SA, keeping the best.
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 20: # Leave time for refinement
            break
            
        # Re-initialize or perturb?
        # For restarts, use a new random initialization or perturb the best.
        if restart == 0:
            sa_points = best_points.copy()
        else:
            # Perturb best points
            sa_points = best_points + np.random.normal(0, 0.05, size=best_points.shape)
            # Ensure inside
            norms = np.hypot(sa_points[:, 0], sa_points[:, 1])
            mask = norms > 1.0
            if np.any(mask):
                sa_points[mask] = sa_points[mask] / norms[mask][:, np.newaxis] * 0.99
        
        # Run SA
        # Adjust iterations based on time
        time_left = time_limit - (time.time() - start_time)
        # Reserve 20s for refinement
        sa_time_budget = max(10, time_left - 20)
        # Assume 1 iter takes ~0.5ms.
        max_sa_iters = int(sa_time_budget / 0.0005)
        
        sa_points, sa_r = simulated_annealing(sa_points, t_max, t_min, cooling_rate, max_sa_iters)
        
        if sa_r > best_r:
            best_r = sa_r
            best_points = sa_points.copy()
    
    # 3. Gradient Refinement
    if time.time() - start_time < time_limit - 5:
        refined_points, refined_r = gradient_refine(best_points, max_iters=500)
        if refined_r > best_r:
            best_r = refined_r
            best_points = refined_points
    
    return best_points

# EVOLVE-BLOCK-END
