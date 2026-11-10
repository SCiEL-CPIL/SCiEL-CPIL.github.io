# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the raw score: A_min / A_hull.
    Since points are generated to be strictly convex and ordered, 
    the convex hull area is the polygon area.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Calculate hull area (shoelace formula)
    # points are in order around the perimeter
    x = points[:, 0]
    y = points[:, 1]
    # Shift x for cross product
    x_shifted = np.roll(x, -1)
    y_shifted = np.roll(y, -1)
    
    area_sum = np.sum(x * y_shifted - x_shifted * y)
    A_hull = 0.5 * np.abs(area_sum)
    
    if A_hull <= 1e-10:
        return 0.0
        
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 0:
        return 0.0
        
    return min_area / A_hull

def _objective_from_angles(angles: np.ndarray) -> float:
    """
    Objective function: maximize A_min / A_hull.
    Returns negative score for minimization.
    angles: array of 13 angular gaps that sum to 2*pi.
    """
    # Ensure angles are positive and sum to 2*pi
    # We assume the input is already normalized to sum to 2*pi in the search step
    # But for safety, normalize here if slight drift occurs
    total = np.sum(angles)
    if total <= 1e-10:
        return 1e10
    angles = angles / total * (2 * np.pi)
    
    # Cumulative sum to get actual angles from origin (starting at 0)
    # The order of points is determined by the cumulative sum
    theta = np.cumsum(angles)
    # Ensure the last point is exactly 2*pi? No, cumulative sum of gaps = 2*pi.
    # The points are at theta[0], theta[1], ..., theta[12].
    # Note: theta[12] should be close to 2*pi.
    
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Calculate score
    # We can reuse _compute_score logic but inline for speed?
    # _compute_score is vectorized, so it's fast enough.
    return -_compute_score(points)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area.
    
    Strategy:
    1. Parametrize points as vertices of a convex polygon on the unit circle.
    2. The shape is defined by 13 angular gaps (deltas) that sum to 2*pi.
    3. Use Simulated Annealing to optimize these 13 gaps.
    4. This enforces convexity, simplifying the hull area calculation.
    """
    start_time = time.time()
    deadline = 600.0  # seconds
    safety_margin = 10.0
    
    n_points = 13
    
    # Initial state: Regular 13-gon
    # Angles are the gaps between consecutive points
    best_angles = np.full(n_points, 2 * np.pi / n_points)
    best_score = -_objective_from_angles(best_angles)
    
    # Simulated Annealing parameters
    # We need to explore the space of angular distributions.
    # Start with a relatively large temperature to allow big jumps.
    T = 0.5
    T_min = 1e-6
    cooling_rate = 0.9995  # Slow cooling to explore more
    
    # We will run as many iterations as possible until deadline
    # Each iteration involves:
    # 1. Perturb one or a few angles
    # 2. Normalize
    # 3. Compute objective (vectorized, ~0.1ms)
    
    current_angles = best_angles.copy()
    current_score = best_score
    
    rng = np.random.default_rng(12345)
    
    # Track best
    # We might also want to try a few different starting configurations?
    # The regular polygon is a strong baseline. 
    # Let's stick to one robust SA run for simplicity and reliability, 
    # or maybe 2 runs with different seeds if time permits?
    # Given the high iteration count possible, one long run is likely sufficient.
    
    # To allow for some diversity in case SA gets stuck, we can occasionally 
    # "reset" to a random valid configuration if stuck for too long? 
    # Or just rely on the high T.
    
    # Let's implement a simple restart strategy: 
    # If no improvement for X iterations, and T is still high, maybe perturb more aggressively?
    # Standard SA usually handles this.
    
    iter_count = 0
    no_improve_count = 0
    
    while time.time() - start_time < deadline - safety_margin:
        # Perturbation
        # Choose 1 or 2 random angles to perturb
        num_perturb = 1 if rng.random() < 0.7 else 2
        indices = rng.choice(n_points, size=num_perturb, replace=False)
        
        # Perturbation size proportional to T
        # Angles are roughly 2*pi/13 ~ 0.48
        # We want perturbations that can significantly change the shape
        noise = rng.normal(0, T, size=num_perturb)
        
        new_angles = current_angles.copy()
        new_angles[indices] += noise
        
        # Ensure angles are positive (convexity constraint)
        # If an angle becomes <= 0, the points would cross or coincide.
        # We can clamp them to a small epsilon.
        min_angle = 1e-4
        if np.any(new_angles < min_angle):
            # Reject if invalid, or clamp? 
            # Clamping might bias the search. Rejecting is cleaner.
            # But if T is high, many might be rejected.
            # Let's try clamping to min_angle and renormalizing.
            new_angles = np.maximum(new_angles, min_angle)
            
        # Normalize to sum to 2*pi
        total = np.sum(new_angles)
        if total <= 1e-10:
            continue
        new_angles = new_angles / total * (2 * np.pi)
        
        # Evaluate
        new_score = -_objective_from_angles(new_angles)
        
        # Acceptance criterion
        delta_E = new_score - current_score
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_angles = new_angles
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_angles = current_angles.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        # Cooling
        T *= cooling_rate
        
        # If T gets too low and we haven't improved in a while, 
        # maybe we should restart with a higher T? 
        # Or just let it run.
        # Given the deadline, we just keep going.
        
        iter_count += 1
        
        # Safety check for time
        if time.time() - start_time > deadline - safety_margin:
            break

    # Convert best angles to points
    theta = np.cumsum(best_angles)
    # Ensure the last point is handled correctly. 
    # The points are at angles theta[0] ... theta[12].
    # The polygon is closed by the fact that sum of gaps is 2*pi.
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Final sanity check
    if not np.all(np.isfinite(points)):
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return points

# EVOLVE-BLOCK-END
