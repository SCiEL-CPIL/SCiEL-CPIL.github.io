import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Solves the Heilbronn problem for 13 points in a convex region.
    
    Strategy:
    1. Use a structured initial guess based on Halton sequences or regular polygons.
    2. Perform a local search using a deterministic "push apart" heuristic specifically
       targeting the minimum area triangle.
    3. Crucially, alternate between moving points to increase the min triangle area
       and normalizing/scaling to keep the hull area fixed (or simply maximizing the ratio).
    
    The current program uses Nelder-Mead which is slow and can get stuck.
    This program uses a direct, geometry-aware local optimization loop:
    - Find the min-area triangle (i, j, k).
    - Move i, j, k slightly apart to increase their area.
    - Check for collisions or hull violations? No, hull area is in denominator.
    - Actually, moving points to increase min area might decrease hull area if they move inward.
    - Better: Maximize min_area / hull_area.
    - Gradient of min_area w.r.t points is non-zero only for the min triangle.
    - Gradient of hull_area is complex.
    
    Let's try a different approach: **Coordinate Descent with Adaptive Steps**.
    For each point, try moving it in a grid of directions/steps.
    Accept the move if the score improves.
    This is robust and simple.
    
    To make it faster:
    - Vectorized score calculation.
    - Pre-computed indices for triangles.
    - Early stopping.
    
    Let's implement a fast local search.
    """
    start_time = time.time()
    deadline = start_time + 590

    # Precompute triangle indices
    indices = np.array(list(itertools.combinations(range(13), 3)))
    i_idx, j_idx, k_idx = indices[:, 0], indices[:, 1], indices[:, 2]

    def score(points):
        """
        Compute the score: min triangle area / convex hull area.
        """
        if points.shape != (13, 2):
            return -np.inf
        if not np.all(np.isfinite(points)):
            return -np.inf
        
        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        if min_area < 1e-12:
            return 0.0

        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 0.0

        if hull_area < 1e-12:
            return 0.0

        return min_area / hull_area

    def fast_score(points):
        """
        Faster score calculation, skipping ConvexHull if possible?
        No, ConvexHull is needed for the denominator.
        But we can cache it? No, points change.
        """
        return score(points)

    # Initial configuration: Halton sequence in a square, scaled.
    def halton(index, base):
        f = 1
        r = 0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    halton_points = np.zeros((13, 2))
    for i in range(13):
        halton_points[i, 0] = halton(i + 1, 2)
        halton_points[i, 1] = halton(i + 1, 3)
    
    # Center and scale
    halton_points = (halton_points - 0.5) * 2.0
    
    # Also try regular polygon
    theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    poly_points = np.column_stack((np.cos(theta), np.sin(theta)))
    
    inits = [halton_points, poly_points]
    
    best_score = -np.inf
    best_points = None
    
    for init in inits:
        if time.time() > deadline:
            break
            
        current_points = init.copy()
        current_score = score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
        
        # Local search loop
        step_size = 0.1
        no_improve_count = 0
        
        while time.time() < deadline:
            improved = False
            
            # Try moving each point
            for idx in range(13):
                if time.time() > deadline:
                    break
                
                original_pos = current_points[idx].copy()
                
                # Directions: 8 neighbors + radial out/in
                # Simple grid search around current position
                # To save time, only try a few directions
                directions = [
                    [1, 0], [-1, 0], [0, 1], [0, -1],
                    [1, 1], [-1, -1], [1, -1], [-1, 1]
                ]
                
                # Normalize directions
                for d in directions:
                    d = np.array(d, dtype=float)
                    d = d / np.linalg.norm(d)
                    
                    # Try a few step sizes
                    for ss in [step_size, step_size * 0.5, step_size * 2]:
                        new_pos = original_pos + d * ss
                        new_points = current_points.copy()
                        new_points[idx] = new_pos
                        
                        new_score = score(new_points)
                        
                        if new_score > current_score:
                            current_points = new_points
                            current_score = new_score
                            improved = True
                            # Reset no_improve count? Or keep it?
                            # If we improved, we can try to increase step size slightly?
                            # Or just keep it.
                            break
                    else:
                        continue
                    break
                
                # If any move improved, restart the loop over points?
                # Or continue to next point?
                # Continuing is faster.
            
            if improved:
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Reduce step size
                step_size *= 0.9
                if step_size < 1e-5 or no_improve_count > 20:
                    break

    # Final check
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = inits[0]
        
    return best_points

if __name__ == "__main__":
    pts = heilbronn_convex13()
    print("Shape:", pts.shape)
    # Recalculate score for print
    indices = np.array(list(itertools.combinations(range(13), 3)))
    i_idx, j_idx, k_idx = indices[:, 0], indices[:, 1], indices[:, 2]
    p1 = pts[i_idx]
    p2 = pts[j_idx]
    p3 = pts[k_idx]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        print("Score:", min_area / hull_area)
    except Exception as e:
        print("Error:", e)
