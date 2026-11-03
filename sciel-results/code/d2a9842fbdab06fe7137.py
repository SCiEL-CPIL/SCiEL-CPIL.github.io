"""
Strategy: Simulated Annealing on free particle positions.
Unlike the ring-based approach, this allows circles to move freely in 2D space,
breaking symmetry and finding irregular packings.
We minimize a potential energy function that penalizes overlaps and boundary violations.
"""

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_radius(points):
    """Compute the maximum radius for the given points."""
    if len(points) == 0:
        return 0.0
    
    # Check container constraint
    dist_to_center = np.linalg.norm(points, axis=1)
    min_container_dist = 1.0 - np.max(dist_to_center)
    
    # Check pairwise distances
    # Use scipy.spatial.distance.pdist for efficiency
    from scipy.spatial.distance import pdist
    d = pdist(points, 'euclidean')
    if len(d) == 0:
        min_pair_dist = float('inf')
    else:
        min_pair_dist = np.min(d)
    
    # The radius is half the minimum distance between centers, 
    # and also limited by the distance to the boundary
    r_pair = min_pair_dist / 2.0
    r_container = min_container_dist
    
    return min(r_pair, r_container)

def _initial_positions():
    """
    Generate initial positions in a hexagonal lattice pattern, 
    slightly perturbed to break symmetry.
    """
    points = []
    # Approximate spacing for 50 circles in unit circle
    # Area per circle ~ pi/50. r ~ sqrt(1/50) ~ 0.14.
    # Spacing ~ 2*r ~ 0.28.
    spacing = 0.3
    # Generate points in a hex grid
    for row in range(-5, 6):
        for col in range(-5, 6):
            x = col * spacing + (row % 2) * spacing * 0.5
            y = row * spacing * np.sqrt(3) / 2
            if x**2 + y**2 <= (1 - spacing/2)**2:
                points.append([x, y])
    
    points = np.array(points)
    # If we have too many, shuffle and take N
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    # If we have too few, add random points
    while len(points) < N:
        angle = np.random.uniform(0, 2*np.pi)
        rad = np.random.uniform(0, 1)
        points = np.vstack([points, [rad*np.cos(angle), rad*np.sin(angle)]])
        
    # Add small random noise to break symmetry
    points += np.random.normal(0, 0.01, points.shape)
    
    # Clip to inside
    dists = np.linalg.norm(points, axis=1)
    mask = dists > 0.95
    points[mask] *= (0.95 / dists[mask])
    
    return points

def _potential_energy(points):
    """
    Calculate a potential energy score.
    We want to minimize overlaps and boundary violations.
    This function is used for the annealing step.
    """
    # 1. Boundary violations
    dists = np.linalg.norm(points, axis=1)
    # Soft penalty for being too close to boundary (d > 0.95)
    # Actually, we want to maximize min(1-d, min_pair/2).
    # Let's define energy as sum of violations.
    # If d > 1, it's invalid.
    # Let's use a penalty function that pushes points in if d > 0.9
    # and pushes points apart if dist < 0.3
    
    # Boundary energy
    boundary_viol = np.sum(np.maximum(dists - 0.95, 0)**2)
    
    # Pairwise energy
    # We want to maximize min distance.
    # A common approach: sum of (target_dist - actual_dist)^2 for all pairs < target_dist
    # Or simply: -log(min_dist) ? No, non-differentiable.
    # Let's use: sum of max(0, target_dist - dist_ij)^2
    # What is target_dist? We don't know the optimal r yet.
    # Let's assume a target spacing of 0.25 (r=0.125).
    target_dist = 0.25
    from scipy.spatial.distance import pdist
    d = pdist(points, 'euclidean')
    # Only consider pairs that are closer than target
    close_pairs = d < target_dist
    if np.any(close_pairs):
        pair_viol = np.sum((target_dist - d[close_pairs])**2)
    else:
        pair_viol = 0.0
        
    return boundary_viol + pair_viol

def _anneal(points, time_limit):
    """
    Perform Simulated Annealing.
    """
    current_energy = _potential_energy(points)
    best_points = points.copy()
    best_energy = current_energy
    best_radius = _compute_radius(points)
    
    T = 0.1 # Initial temperature
    T_min = 1e-5
    cooling_rate = 0.995
    step_size = 0.05
    
    start_time = time.time()
    
    while T > T_min:
        if time.time() > time_limit:
            break
            
        # Propose a move: pick a random point and move it
        i = np.random.randint(N)
        move = np.random.normal(0, step_size, 2)
        
        new_points = points.copy()
        new_points[i] += move
        
        # Check if it went outside, if so, penalize heavily in energy calc
        new_energy = _potential_energy(new_points)
        
        delta_e = new_energy - current_energy
        
        # Acceptance probability
        if delta_e < 0:
            # Accept if better
            points = new_points
            current_energy = new_energy
        else:
            # Accept if worse with probability exp(-delta_e / T)
            if np.random.rand() < np.exp(-delta_e / T):
                points = new_points
                current_energy = new_energy
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = points.copy()
            # Calculate radius periodically to track best valid packing
            # _compute_radius is expensive (pdist). Only do it if energy improved significantly
            # or every few steps.
            # For now, let's check radius every 100 steps or if energy drops a lot
            if best_energy < best_energy - 1e-4 or np.random.rand() < 0.01:
                r = _compute_radius(points)
                if r > best_radius:
                    best_radius = r
                    best_points = points.copy()
        
        T *= cooling_rate
        
    return best_points, best_radius

def _polish(points, r):
    """
    Deterministic local refinement.
    Push apart overlapping circles and pull in boundary violators.
    """
    points = points.copy()
    for _ in range(100):
        improved = False
        
        # 1. Boundary check
        dists = np.linalg.norm(points, axis=1)
        for i in range(N):
            if dists[i] > 1.0 - r:
                # Push in
                if dists[i] > 1e-9:
                    diff = points[i] / dists[i]
                else:
                    diff = np.array([1.0, 0.0])
                points[i] -= diff * (dists[i] - (1.0 - r))
                improved = True
        
        # 2. Pairwise check
        from scipy.spatial.distance import pdist, squareform
        d = pdist(points, 'euclidean')
        dist_mat = squareform(d)
        
        for i in range(N):
            for j in range(i+1, N):
                dist = dist_mat[i, j]
                if dist < 2.0 * r:
                    # Push apart
                    if dist > 1e-9:
                        diff = (points[i] - points[j]) / dist
                    else:
                        diff = np.array([1.0, 0.0])
                    # Move both by half the overlap
                    overlap = (2.0 * r - dist) / 2.0
                    points[i] += diff * overlap
                    points[j] -= diff * overlap
                    improved = True
                    
        if not improved:
            break
            
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Initial guess
    points = _initial_positions()
    
    # 2. Simulated Annealing
    # We want to maximize r. The annealing minimizes energy.
    # The energy function uses a target distance of 0.25.
    # If the actual optimal r is larger, the energy might not perfectly correlate,
    # but it pushes circles apart generally.
    
    # Run multiple restarts with different initial conditions if time permits
    best_points = points
    best_radius = _compute_radius(points)
    
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for k in range(num_restarts):
        if time.time() > deadline:
            break
            
        # For first restart, use the initial points. For others, regenerate.
        if k == 0:
            pts = points
        else:
            pts = _initial_positions()
            
        # Anneal
        anneal_limit = start_time + (k+1) * time_per_restart
        # Ensure we don't exceed global deadline
        anneal_limit = min(anneal_limit, deadline - 5.0)
        
        if time.time() > anneal_limit:
            continue
            
        polished_pts, radius = _anneal(pts, anneal_limit)
        
        # Polish
        polished_pts = _polish(polished_pts, radius)
        final_radius = _compute_radius(polished_pts)
        
        if final_radius > best_radius:
            best_radius = final_radius
            best_points = polished_pts
            
    # Final polish on the best found
    best_points = _polish(best_points, best_radius)
    final_radius = _compute_radius(best_points)
    
    if final_radius > best_radius:
        best_radius = final_radius
        best_points = best_points # Already polished
    
    # Ensure valid output
    if best_points is None or len(best_points) != N:
        # Fallback
        best_points = np.random.rand(N, 2) * 1.8 - 0.9
        d = np.linalg.norm(best_points, axis=1)
        max_d = np.max(d)
        if max_d > 0.8:
            best_points = best_points * (0.8 / max_d)
            
    return best_points

# EVOLVE-BLOCK-END
