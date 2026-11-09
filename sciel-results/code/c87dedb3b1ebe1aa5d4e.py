import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools
import sys

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(points):
    """
    Fast calculation of minimum triangle area using vectorized operations.
    Points: (N, 2) array. Returns minimum area (not doubled).
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            v1 = points[j] - pi
            v2s = points[j + 1:] - pi
            cross_vals = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            if len(areas) > 0:
                cur_min = np.min(areas)
                if cur_min < min_area:
                    min_area = cur_min
    return min_area

def _score(points):
    """
    Score = Min Triangle Area / Convex Hull Area.
    """
    try:
        if np.allclose(points, points[0]):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        
        if hull_area < 1e-12:
            return 0.0
            
        min_tri_area = _min_triangle_area_fast(points)
        
        if min_tri_area < 1e-12:
            return 0.0
            
        return min_tri_area / hull_area
    except Exception:
        return 0.0

def _angles_to_points(angles):
    """
    Convert angles to points on the unit circle.
    """
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _angular_score(angles):
    """
    Score based on angles.
    """
    points = _angles_to_points(angles)
    return _score(points)

def _optimize_angles(angles0, max_iter=1000, deadline=None, seed=42):
    """
    Custom gradient-free optimization for angles.
    Uses a hill-climbing approach with adaptive step size and angular separation enforcement.
    """
    rng = np.random.default_rng(seed)
    current_angles = angles0.copy()
    current_score = _angular_score(current_angles)
    
    best_angles = current_angles.copy()
    best_score = current_score
    
    # Adaptive step size
    step_size = 0.1
    min_step_size = 1e-6
    no_improve_count = 0
    max_no_improve = 20
    
    # Minimum angular separation to prevent degeneracy (rough heuristic)
    # For N points, average spacing is 2pi/N. We want to avoid clusters.
    min_sep = (2 * np.pi / N) * 0.3 
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Generate a perturbation
        # We perturb a subset of angles to allow local rearrangements
        num_perturb = rng.integers(1, 4)
        indices = rng.choice(N, size=num_perturb, replace=False)
        
        new_angles = current_angles.copy()
        for idx in indices:
            new_angles[idx] += rng.normal(0, step_size)
            
        # Normalize angles to [0, 2pi)
        new_angles = new_angles % (2 * np.pi)
        
        # Enforce angular separation: if two angles are too close, push them apart
        # This is a simple constraint projection
        # Sort to find neighbors
        sorted_idx = np.argsort(new_angles)
        sorted_angles = new_angles[sorted_idx]
        
        # Check circular distance
        # We'll do a simple pass to enforce minimum separation
        for _ in range(5): # few iterations of separation
            for k in range(N):
                idx1 = sorted_idx[k]
                idx2 = sorted_idx[(k+1)%N]
                
                a1 = new_angles[idx1]
                a2 = new_angles[idx2]
                
                # Calculate forward distance from a1 to a2
                dist = (a2 - a1) % (2 * np.pi)
                
                if dist < min_sep:
                    # Push them apart
                    # Move each by half the deficit in opposite directions
                    deficit = (min_sep - dist) / 2
                    new_angles[idx1] = (a1 - deficit) % (2 * np.pi)
                    new_angles[idx2] = (a2 + deficit) % (2 * np.pi)
                    
        new_score = _angular_score(new_angles)
        
        if new_score > current_score:
            current_angles = new_angles
            current_score = new_score
            no_improve_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_angles = current_angles.copy()
        else:
            no_improve_count += 1
            # Reduce step size if stuck
            step_size *= 0.95
            
        if step_size < min_step_size or no_improve_count > max_no_improve:
            break
            
    return best_angles, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = 0.0
    
    def update_best(points, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_points = points.copy()
            best_score = score
            
    # Strategy: Optimize angles for points on the unit circle.
    # The hull area is fixed to that of the regular 14-gon if all points are on the circle.
    # Actually, the hull area depends on the specific angles.
    # Regular polygon has max area for a given circumradius.
    # If we keep points on the circle, the hull is the polygon formed by the points in angular order.
    
    # Initializations
    initializations = []
    
    # 1. Regular polygon angles
    regular_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initializations.append(regular_angles)
    
    # 2. Jittered regular polygon angles
    for seed in [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]:
        rng = np.random.default_rng(seed)
        jitter = rng.normal(0, 0.05, N)
        angles = (regular_angles + jitter) % (2 * np.pi)
        angles.sort() # Keep them sorted for consistency, though _angles_to_points doesn't strictly need it, 
                      # but the separation logic assumes some order or handles it. 
                      # Actually _angular_score uses _score which uses ConvexHull, so order doesn't matter for scoring.
                      # But my separation logic sorts internally.
        initializations.append(angles)
        
    # 3. Random angles
    for seed in [42, 123, 456, 789, 1011]:
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0, 2 * np.pi, N)
        angles.sort()
        initializations.append(angles)
        
    # First pass: Quick evaluation
    for init in initializations:
        if time.time() > deadline - 10:
            break
        points = _angles_to_points(init)
        score = _score(points)
        update_best(points, score)
        
    # Second pass: Local optimization from best initializations
    # We will optimize the angles directly.
    scored_inits = []
    for init in initializations:
        points = _angles_to_points(init)
        score = _score(points)
        scored_inits.append((score, init))
        
    scored_inits.sort(key=lambda x: -x[0])
    
    top_k = min(10, len(scored_inits))
    for i, (score, init_angles) in enumerate(scored_inits[:top_k]):
        if time.time() > deadline - 15:
            break
            
        # Run custom angular optimization
        try:
            opt_angles, opt_score = _optimize_angles(
                init_angles, 
                max_iter=2000, 
                deadline=deadline - 5, 
                seed=42 + i
            )
            opt_points = _angles_to_points(opt_angles)
            # Re-verify score with the standard scorer to be safe
            final_score = _score(opt_points)
            update_best(opt_points, final_score)
        except Exception:
            pass
            
    # Third pass: Refine the best solution
    if best_points is not None and time.time() < deadline - 20:
        # Convert best_points back to angles? 
        # It's better to keep track of the best angles directly if we had them.
        # But update_best stores points. 
        # Let's just re-optimize from the current best points by extracting angles.
        # This is slightly inefficient but robust.
        best_angles = np.arctan2(best_points[:, 1], best_points[:, 0])
        best_angles = best_angles % (2 * np.pi)
        
        try:
            opt_angles, opt_score = _optimize_angles(
                best_angles, 
                max_iter=5000, 
                deadline=deadline - 5, 
                seed=999
            )
            opt_points = _angles_to_points(opt_angles)
            final_score = _score(opt_points)
            update_best(opt_points, final_score)
        except Exception:
            pass
            
    if best_points is None:
        # Fallback: regular polygon
        best_points = _angles_to_points(np.linspace(0, 2 * np.pi, N, endpoint=False))
        
    return best_points

# EVOLVE-BLOCK-END
