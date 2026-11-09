"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Reproducibility for debugging, but we want variety in search
    # random.seed(42)
    
    def get_score(pts):
        """Exact score calculation: min radius."""
        # Pairwise distances
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        # Boundary distance
        min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_pair, min_boundary)

    def compute_forces(pts, r_target):
        """
        Compute repulsive forces between circles and attractive force towards center.
        Force is proportional to overlap or proximity.
        We use a simple repulsion model: F = k / d^2 if d < 2*r_target, else 0.
        Actually, to maximize min distance, we want to push apart if d < 2*r_best.
        But r_best changes. 
        A better approach for SA: 
        Energy E = sum( (2*r_target - d_ij)^2 ) for d_ij < 2*r_target + boundary_penalty
        Or simpler: Move points apart if they are too close.
        
        Let's use a force-based approach where we try to increase the minimum distance.
        Force on i from j: direction (p_i - p_j) * strength.
        Strength is high if d_ij is small.
        
        Let's use a potential: V(d) = 1/d^2 if d < 2*r_current else 0.
        This creates strong repulsion at close range.
        """
        forces = np.zeros_like(pts)
        # Pairwise
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < 1e-6:
                    # Random nudge
                    diff = np.random.randn(2) * 1e-6
                    d = np.linalg.norm(diff)
                
                # We want to separate them. 
                # If we are trying to pack with radius r, we want d >= 2r.
                # The force should be repulsive.
                # Let's use a force that is always repulsive but stronger when close.
                # F = k * (1/d^2) * direction
                # This might be too strong. Let's try F = k / d
                # Or even simpler: just push apart by a fixed amount if overlapping?
                # No, SA needs an energy landscape.
                
                # Let's define energy as sum of (max(0, 2*r_current - d_ij))^2
                # But r_current is what we are optimizing.
                # Alternative: Maximize min distance.
                # In SA, we accept moves that increase the "score" (min dist) or 
                # accept worse moves with probability exp(-delta/T).
                # So we don't necessarily need forces. We can just do random perturbations.
                # However, "smart" perturbations help.
                
                # Let's stick to a simple random walk SA.
                # Move a random point by a small random vector.
                # Calculate new score.
                # Accept if better, or with prob exp((new_score - old_score)/T).
        
        return forces

    def hexagonal_initial():
        """Generate a good initial hexagonal packing."""
        spacing = 0.2
        points = []
        # Generate a large hex grid
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * spacing + 0.5 * spacing * (j % 2)
                y = j * spacing * np.sqrt(3.0) / 2.0
                points.append([x, y])
        points = np.array(points)
        # Filter to inside unit circle
        r = np.linalg.norm(points, axis=1)
        points = points[r < 1.0]
        
        if len(points) > N:
            # Sort by distance to center, take closest N
            r = np.linalg.norm(points, axis=1)
            idx = np.argsort(r)
            points = points[idx[:N]]
        else:
            # Pad if necessary (unlikely for N=50)
            while len(points) < N:
                points = np.vstack([points, np.random.uniform(-0.9, 0.9, 2)])
                
        return points.copy()

    def random_initial():
        pts = np.random.uniform(-1, 1, (N, 2))
        r = np.linalg.norm(pts, axis=1)
        mask = r > 1.0
        pts[mask] /= r[mask, np.newaxis] * 1.05
        return pts

    best_points = hexagonal_initial()
    best_score = get_score(best_points)
    
    # Simulated Annealing
    # We will perform multiple restarts.
    # Each restart: Initialize, then SA loop.
    
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            current_pts = best_points.copy()
        else:
            # Perturb best or start fresh
            if restart % 2 == 0:
                current_pts = best_points + np.random.randn(N, 2) * 0.05
            else:
                current_pts = hexagonal_initial()
                # Add some noise to break symmetry
                current_pts += np.random.randn(N, 2) * 0.01
            
            # Ensure inside
            r = np.linalg.norm(current_pts, axis=1)
            mask = r > 0.99
            if np.any(mask):
                current_pts[mask] *= 0.99 / r[mask, np.newaxis]

        current_score = get_score(current_pts)
        
        # SA Parameters
        # Temperature schedule: T = T0 * decay^step
        T0 = 0.01
        decay = 0.99
        T = T0
        step_size = 0.05
        
        # Run SA for a fixed number of steps or until time limit
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
                
            # Pick a random point
            i = np.random.randint(N)
            
            # Generate a move
            # Step size might need to decrease or stay constant?
            # Let's keep step size roughly constant but scaled by T?
            # Or just fixed step size.
            delta = np.random.randn(2) * step_size
            new_pts = current_pts.copy()
            new_pts[i] += delta
            
            # Check boundary
            r_new = np.linalg.norm(new_pts[i])
            if r_new > 1.0:
                # Reject or push back? Reject is simpler for SA.
                continue
            
            # Calculate new score
            new_score = get_score(new_pts)
            
            # Acceptance criterion
            delta_score = new_score - current_score
            if delta_score > 0:
                # Better, accept
                current_pts = new_pts
                current_score = new_score
            else:
                # Worse, accept with probability exp(delta_score / T)
                # delta_score is negative
                prob = np.exp(delta_score / T)
                if np.random.rand() < prob:
                    current_pts = new_pts
                    current_score = new_score
            
            # Update best
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
            
            # Cooling
            T *= decay
            # If T is too small, we might get stuck. 
            # We could reheat occasionally, but let's just restart.
            
            # Dynamic step size? 
            # If we accept many moves, maybe increase step size?
            # If we accept few, decrease?
            # Let's keep it simple for now.

    # Final local polish using gradient or simple hill climbing on the best point
    # Since SA might leave us in a good basin, a few gradient steps can help.
    # But the prompt asks for a DIFFERENT strategy. SA is the main strategy.
    # We can do a simple local search: try moving each point slightly in random directions.
    
    if time.time() < deadline - 2.0:
        # Simple local refinement
        for _ in range(100):
            if time.time() > deadline - 1.0:
                break
            improved = False
            for i in range(N):
                for _ in range(10):
                    delta = np.random.randn(2) * 0.005
                    new_pts = best_points.copy()
                    new_pts[i] += delta
                    if np.linalg.norm(new_pts[i]) > 1.0:
                        continue
                    new_score = get_score(new_pts)
                    if new_score > best_score:
                        best_points = new_pts
                        best_score = new_score
                        improved = True
            if not improved:
                break

    return best_points
# EVOLVE-BLOCK-END
