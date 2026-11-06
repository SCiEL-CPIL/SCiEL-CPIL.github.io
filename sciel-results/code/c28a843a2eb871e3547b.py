import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def fibonacci_disk(n, radius=0.98):
        """Generate points on a Fibonacci spiral in a disk."""
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        angles = np.arange(n) * golden_angle
        r = radius * np.sqrt((np.arange(n) + 0.5) / n)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        return np.column_stack([x, y])

    def compute_score(pts):
        """Compute the packing score r for given points."""
        if len(pts) != N or not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        # Efficient pairwise distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_pair, min_boundary)

    def energy(pts, r_target):
        """Compute energy: penalize overlaps and boundary violations."""
        radii = np.hypot(pts[:, 0], pts[:, 1])
        boundary_violation = np.maximum(0.0, radii + r_target - 1.0)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        overlap = np.maximum(0.0, 2.0 * r_target - dists)
        
        return np.sum(boundary_violation**2) + 0.5 * np.sum(overlap**2)

    def anneal(pts, r_target, t0, t1, steps, step_scale):
        """Simulated annealing to maximize r."""
        current_energy = energy(pts, r_target)
        best_energy = current_energy
        best_pts = pts.copy()
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Temperature schedule
            t = t0 * (t1 / t0) ** (step / max(steps - 1, 1))
            
            # Generate perturbation
            scale = step_scale * (1.0 + 0.5 * t / t0)
            new_pts = pts + np.random.normal(0, scale, (N, 2))
            
            # Keep points inside container (soft constraint)
            new_radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = new_radii > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / new_radii[mask, np.newaxis] * 0.999
            
            new_energy = energy(new_pts, r_target)
            
            # Acceptance criterion
            delta_e = new_energy - current_energy
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / max(t, 1e-10)):
                pts = new_pts
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = pts.copy()
        
        return best_pts, best_energy

    # Initialize with Fibonacci disk
    points = fibonacci_disk(N, radius=0.95)
    best_points = points.copy()
    best_r = compute_radius(best_points) if 'compute_radius' in dir() else compute_score(best_points)
    
    # Try to increase r_target gradually
    r_target = best_r * 1.01
    t0 = 0.01
    t1 = 0.0001
    step_scale = 0.01
    steps_per_iteration = 300
    
    no_improve_count = 0
    max_no_improve = 10
    
    while time.time() < deadline:
        # Anneal at current r_target
        new_pts, final_energy = anneal(points, r_target, t0, t1, steps_per_iteration, step_scale)
        
        # Check if feasible
        actual_r = compute_score(new_pts)
        
        if actual_r >= r_target * 0.999:
            # Feasible! Update best
            if actual_r > best_r:
                best_r = actual_r
                best_points = new_pts.copy()
                points = new_pts.copy()
                no_improve_count = 0
                # Try to increase r_target
                r_target = best_r * 1.003
            else:
                # Use the annealed configuration even if not better
                points = new_pts.copy()
                no_improve_count += 1
        else:
            # Not feasible. Try smaller increment
            points = new_pts.copy()
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Perturb to escape local minimum
                perturb_scale = 0.005
                points = best_points + np.random.normal(0, perturb_scale, (N, 2))
                # Ensure validity
                radii = np.hypot(points[:, 0], points[:, 1])
                outside = radii > 1.0
                if np.any(outside):
                    points[outside] = points[outside] / radii[outside, np.newaxis] * 0.999
                best_points = points
                no_improve_count = 0
                step_scale = 0.01
                r_target = best_r * 1.001
    
    # Final polish: try to squeeze out more radius with smaller steps
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        r_target = best_r * 1.001
        steps_final = int(remaining_time * 20)  # ~20 steps per second
        new_pts, _ = anneal(best_points, r_target, 0.005, 0.00005, steps_final, 0.005)
        actual_r = compute_score(new_pts)
        if actual_r > best_r:
            best_r = actual_r
            best_points = new_pts.copy()
    
    # Ensure validity
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_points)):
        scale = 0.99 / np.max(radii) if np.max(radii) > 0 else 1.0
        best_points = best_points * scale
    
    return best_points
