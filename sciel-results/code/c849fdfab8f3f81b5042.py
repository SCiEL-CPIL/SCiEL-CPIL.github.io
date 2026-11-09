# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_vertex_score(v, points):
    """
    Compute the score for a single vertex v given the set of all points.
    """
    # Calculate distances from v to all other points
    diffs = points - v
    # Exclude v itself (distance 0)
    # Since v is in points, we need to be careful.
    # The task says: "sort the 19 distances from v to the other vertices"
    # So we compute distances to all, then remove the 0.
    
    dists = np.linalg.norm(diffs, axis=1)
    # The distance to v is 0. We need the other 19.
    # Since floating point might make it slightly non-zero, we just sort and take the non-zeros?
    # Actually, v is exactly one of the points.
    # Let's just sort all 20 and take indices 1:20 assuming 0 is the smallest.
    # But to be safe, we can mask out the exact index if we knew it. 
    # However, the input `points` is the whole array.
    # Let's find the index of v.
    # Actually, simpler: sort all 20, the first is 0 (or close to it).
    # The task says "19 distances from v to the other vertices".
    
    # Sort distances
    sorted_dists = np.sort(dists)
    # The first one is the distance to v itself (0).
    # The next 19 are the distances to others.
    d = sorted_dists[1:]
    
    # Check for very small distances
    if np.any(d < 0.05):
        return 0.0
        
    # Find k: first index minimizing d_{k+1} - d_k
    # d has indices 0 to 18
    # Gaps are d[1]-d[0], d[2]-d[1], ..., d[18]-d[17]
    # So we look at gaps between d[i] and d[i+1] for i in 0..17
    gaps = d[1:] - d[:-1]
    k = np.argmin(gaps)
    
    # D = (d_k + d_{k+1}) / 2
    D = (d[k] + d[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    # Determine the 4 distances to use
    # If k = 0: d_0..d_3
    # If k = 17: d_15..d_18
    # Else: d_{k-1}..d_{k+2}
    
    if k == 0:
        indices = [0, 1, 2, 3]
    elif k == 17:
        indices = [15, 16, 17, 18]
    else:
        indices = [k-1, k, k+1, k+2]
        
    selected_d = d[indices]
    
    # raw = mean over these four d of max(d/D, D/d)
    # Note: d here refers to the selected distances
    ratios = np.maximum(selected_d / D, D / selected_d)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score

def _polygon_min_score(points):
    """
    Compute the minimum vertex score for a given polygon.
    """
    scores = []
    for i in range(len(points)):
        s = _compute_vertex_score(points[i], points)
        scores.append(s)
    return min(scores)

def _generate_polygon(params):
    """
    Generate a polygon from parameters.
    params: [A, phi]
    r_i = 1 + A * cos(5 * theta_i + phi)
    """
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    A = params[0]
    phi = params[1]
    radii = 1.0 + A * np.cos(5.0 * angles + phi)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    # Normalize as per task
    centroid = np.mean(polygon, axis=0)
    polygon = polygon - centroid
    dists = np.linalg.norm(polygon, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        # Return a valid but low score polygon (regular)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    polygon = polygon / mean_dist
    return polygon

def _neg_min_score(params):
    """
    Objective function for minimization (negate the score).
    """
    polygon = _generate_polygon(params)
    score = _polygon_min_score(polygon)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 600.0 # Safety margin
    
    # Initial guess: Regular polygon (A=0)
    # Try a few initial guesses for A and phi
    best_score = -np.inf
    best_polygon = None
    
    initial_params_list = [
        [0.0, 0.0],
        [0.1, 0.0],
        [0.2, 0.0],
        [0.3, 0.0],
        [0.1, np.pi/5],
        [0.2, np.pi/5],
        [0.1, np.pi/2],
        [0.2, np.pi/2],
    ]
    
    for init_params in initial_params_list:
        if time.time() > deadline:
            break
            
        try:
            # Optimize A and phi
            res = minimize(_neg_min_score, init_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if time.time() > deadline:
                break
                
            if res.success or res.fun < -best_score:
                # Evaluate the best point found
                best_params = res.x
                polygon = _generate_polygon(best_params)
                score = _polygon_min_score(polygon)
                
                if score > best_score:
                    best_score = score
                    best_polygon = polygon
                    
        except Exception:
            continue
            
    # If we have a good polygon, do local refinement
    if best_polygon is not None:
        # Coordinate descent on radii
        # We need to reconstruct angles? No, we can just optimize x,y directly or radii.
        # Optimizing radii is safer for convexity if we keep angles fixed.
        # Let's refine the radii of the best polygon found.
        
        # Reconstruct angles from best_polygon? 
        # It's better to keep the parameterization consistent.
        # Let's extract radii from best_polygon relative to centroid (which is 0).
        # r_i = norm(polygon[i])
        # But we need to be careful about the phase.
        # Actually, let's just do a local search on the 20 radii directly, 
        # keeping the angles fixed to the regular polygon angles.
        
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        # Get current radii
        current_radii = np.linalg.norm(best_polygon, axis=1)
        
        def _score_from_radii(radii):
            # Check convexity? 
            # If radii are too different, it might not be convex.
            # The task requires strictly convex.
            # We'll check convexity in the score function or return 0 if not.
            
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            polygon = np.stack([x, y], axis=1)
            
            # Normalize
            centroid = np.mean(polygon, axis=0)
            polygon = polygon - centroid
            dists = np.linalg.norm(polygon, axis=1)
            mean_dist = np.mean(dists)
            if mean_dist <= 0.01:
                return 0.0
            polygon = polygon / mean_dist
            
            # Check convexity
            # Calculate cross products of consecutive edges
            # For CCW order, cross product z-component should be > 0
            # v1 = p[1]-p[0], v2 = p[2]-p[1]
            # cross = v1_x * v2_y - v1_y * v2_x
            # If any cross <= 0, it's not strictly convex (or is CW).
            # We assume CCW.
            p = polygon
            n_pts = len(p)
            for i in range(n_pts):
                p0 = p[i]
                p1 = p[(i+1)%n_pts]
                p2 = p[(i+2)%n_pts]
                v1 = p1 - p0
                v2 = p2 - p1
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                if cross <= 1e-9: # Small tolerance
                    return 0.0
                    
            # Check no repeated points
            for i in range(n_pts):
                for j in range(i+1, n_pts):
                    if np.linalg.norm(p[i] - p[j]) < 1e-9:
                        return 0.0
                            
            score = _polygon_min_score(polygon)
            return score
            
        # Simple coordinate descent
        improved = True
        step = 0.05
        max_iter = 100
        current_score = _score_from_radii(current_radii)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if not improved:
                step *= 0.5
                if step < 1e-4:
                    break
                improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                # Try increasing
                old_r = current_radii[i]
                current_radii[i] = old_r + step
                s = _score_from_radii(current_radii)
                if s > current_score:
                    current_score = s
                    improved = True
                else:
                    current_radii[i] = old_r - step
                    s = _score_from_radii(current_radii)
                    if s > current_score:
                        current_score = s
                        improved = True
                    else:
                        current_radii[i] = old_r
                        
        # Final check
        final_score = _score_from_radii(current_radii)
        if final_score > best_score:
            best_score = final_score
            x = current_radii * np.cos(angles)
            y = current_radii * np.sin(angles)
            best_polygon = np.stack([x, y], axis=1)
            # Normalize final polygon
            centroid = np.mean(best_polygon, axis=0)
            best_polygon = best_polygon - centroid
            dists = np.linalg.norm(best_polygon, axis=1)
            mean_dist = np.mean(dists)
            if mean_dist > 0.01:
                best_polygon = best_polygon / mean_dist
            else:
                # Fallback to regular
                best_polygon = np.column_stack([np.cos(angles), np.sin(angles)])

    # If no valid polygon was found or score is 0, return regular polygon
    if best_polygon is None or best_score <= 0:
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_polygon = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_polygon

# EVOLVE-BLOCK-END
