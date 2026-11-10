"""Strategy: Rigid-body optimization with multi-restart annealing.
We identify 'active' constraints (pairs closer than a threshold) and push them apart.
We also push boundary violations inward.
This focuses computational effort on the critical points defining the radius.
"""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def compute_score(points):
        """
        Compute the score (radius) for a given set of points.
        """
        n = len(points)
        if n < 2:
            return 0.0
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Get minimum pairwise distance (excluding diagonal)
        mask = ~np.eye(n, dtype=bool)
        min_pair_dist = np.min(dist[mask])
        # Compute minimum distance to boundary
        radii = np.linalg.norm(points, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def generate_hex_lattice(spacing, jitter=0.0):
        """
        Generate a hexagonal lattice of points, keeping those inside the unit circle.
        """
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                # Add jitter if requested
                if jitter > 0:
                    x += random.uniform(-jitter, jitter)
                    y += random.uniform(-jitter, jitter)
                r = np.hypot(x, y)
                # Keep points well inside to allow for optimization
                if r <= 1.0 - spacing * 0.5:
                    pts.append((x, y))
        return np.array(pts)

    def optimize(points, time_limit):
        """
        Optimize the packing using a rigid-body force approach.
        """
        points = points.copy()
        current_score = compute_score(points)
        best_score = current_score
        best_points = points.copy()

        # Initial step size based on current spacing
        # Estimate spacing from score
        est_spacing = max(2.0 * current_score, 0.01)
        step_size = est_spacing * 0.1
        # We will decay step_size over time
        
        # Pre-compute indices for efficiency? No, N=300 is small enough for full matrix ops
        # But we only need to act on active constraints.
        
        while time.time() < time_limit:
            n = len(points)
            
            # Compute distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            
            # Identify active pairs: those with distance < current_score * 2 * (1 + small_epsilon)
            # Actually, we want to push apart pairs that are close to the minimum.
            # Let's define 'active' as dist < 2.0 * best_score * 1.05
            active_thresh = 2.0 * best_score * 1.05
            
            # Forces
            forces = np.zeros_like(points)
            
            # 1. Repulsion from active pairs
            # We only compute forces for active pairs to save time? 
            # With N=300, full matrix is 90k entries. This is fast in numpy.
            # Let's stick to full matrix but only add force if active.
            
            # Mask for active pairs (excluding diagonal)
            mask = ~np.eye(n, dtype=bool)
            active_mask = mask & (dist < active_thresh)
            
            if np.any(active_mask):
                # Force magnitude: proportional to how much they violate the "ideal" distance?
                # Or just constant repulsion for active pairs?
                # Constant repulsion is simpler.
                # Direction: away from neighbor.
                # F_i += sum_{j in active(i)} (p_i - p_j) / dist_ij
                # We want to push them apart.
                
                # Normalize diff by dist
                # dist has zeros on diagonal, but active_mask excludes diagonal.
                # To avoid div by zero, set dist to 1 where not active.
                safe_dist = np.where(active_mask, dist, 1.0)
                inv_dist = np.where(active_mask, 1.0 / safe_dist, 0.0)
                
                # Force vector for each pair (i,j) is (p_i - p_j) * inv_dist
                # Sum over j for each i
                # diff[i,j] = p_i - p_j
                # force_i = sum_j diff[i,j] * inv_dist[i,j]
                
                pair_forces = diff * inv_dist[:, :, np.newaxis]
                forces += np.sum(pair_forces, axis=1)

            # 2. Boundary constraint
            # If a point is too close to boundary (dist < 1.0 - best_score), push inward
            radii = np.linalg.norm(points, axis=1)
            boundary_thresh = 1.0 - best_score
            boundary_mask = radii > boundary_thresh
            
            if np.any(boundary_mask):
                # Push towards center
                # Force direction: -p_i / |p_i|
                # Magnitude: proportional to violation
                violation = radii[boundary_mask] - boundary_thresh
                # Normalize points
                norm_points = points[boundary_mask] / radii[boundary_mask, np.newaxis]
                forces[boundary_mask] -= norm_points * violation[:, np.newaxis]

            # Normalize forces to prevent explosion
            force_norms = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_norms)
            if max_force > 1e-10:
                forces /= max_force

            # Update positions
            points += step_size * forces

            # Project back inside unit circle (hard constraint)
            radii = np.linalg.norm(points, axis=1)
            mask_out = radii > 1.0
            if np.any(mask_out):
                points[mask_out] /= radii[mask_out, np.newaxis]

            # Compute new score
            current_score = compute_score(points)
            
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                # If we improve, we can slightly increase step size? No, keep annealing.
            else:
                # If no improvement, we might be stuck. 
                # Simple annealing: reduce step size.
                step_size *= 0.999

        return best_points, best_score

    # Main loop: Multi-restart
    # Initial spacing estimate
    # Area argument: r ~ sqrt(0.907/N)
    initial_spacing = 2.0 * np.sqrt(0.907 / N)
    
    best_overall_score = 0.0
    best_overall_points = None

    # Try a few different initial spacings and jitters
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Vary spacing slightly
        spacing = initial_spacing * (1.0 + 0.05 * i)
        jitter = 0.0 if i == 0 else 0.01 * i
        
        # Generate points
        pts = generate_hex_lattice(spacing, jitter)
        
        # If we have more than N points, select N closest to center
        if len(pts) > N:
            radii = np.hypot(pts[:, 0], pts[:, 1])
            pts = pts[np.argsort(radii, kind="stable")[:N]].copy()
        else:
            # If we have fewer, we need to add points or adjust spacing
            # This shouldn't happen with the margin in generate_hex_lattice
            # But just in case, reduce spacing
            while len(pts) < N and time.time() < deadline:
                spacing *= 0.98
                pts = generate_hex_lattice(spacing, jitter)
                if len(pts) > N:
                    radii = np.hypot(pts[:, 0], pts[:, 1])
                    pts = pts[np.argsort(radii, kind="stable")[:N]].copy()
                if len(pts) < N:
                    break
            
            if len(pts) < N:
                continue

        # Optimize this configuration
        # Allocate remaining time for this restart, minus a small buffer for next restarts
        remaining_time = deadline - time.time()
        # Use 80% of remaining time for this restart if it's the last, else proportional
        # Simple: give each restart a fixed chunk
        alloc_time = time_per_restart * 0.9
        
        opt_points, opt_score = optimize(pts, time.time() + alloc_time)
        
        if opt_score > best_overall_score:
            best_overall_score = opt_score
            best_overall_points = opt_points

    # If we still have time, do a final polish on the best result
    if best_overall_points is not None and time.time() < deadline - 5:
        # Run one more optimization cycle on the best result with smaller steps
        # This might help refine the local minimum
        final_points, final_score = optimize(best_overall_points, deadline)
        if final_score > best_overall_score:
            best_overall_points = final_points
            best_overall_score = final_score

    if best_overall_points is None:
        # Fallback: return a simple grid if optimization failed completely
        # This should not happen
        return np.zeros((N, 2))

    return best_overall_points
# EVOLVE-BLOCK-END
