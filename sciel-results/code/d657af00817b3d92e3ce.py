"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

def _convex_hull_area(points):
    """Compute convex hull area of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        pts = np.array(points)
        if len(pts) < 3:
            return 0.0
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        hull = lower[:-1] + upper[:-1]
        if len(hull) < 3:
            return 0.0
        x = hull[:, 0]
        y = hull[:, 1]
        return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def _all_triangle_areas(points):
    """Compute all triangle areas among all triples."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * np.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    return np.array(areas)

def _score(points):
    """Compute the minimum triangle area divided by the convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-15:
        return 0.0
    min_area = np.min(_all_triangle_areas(points))
    return min_area / hull_area

def _greedy_lattice_construction(grid_size=4, start_time=None, deadline=None):
    """
    Construct a set of 14 points by greedily selecting from a grid.
    We start with a regular polygon or a few good points, and then add points
    from a candidate grid (or slightly perturbed grid) that maximize the minimum
    triangle area with the existing set.
    """
    if start_time is None:
        start_time = time.time()
    if deadline is None:
        deadline = start_time + 140.0
        
    # Candidate pool: A 4x4 grid of points, plus maybe some intermediate points
    # Actually, a 5x5 grid gives 25 candidates, which is small enough for greedy.
    # Let's use a 5x5 grid centered at 0.
    coords = np.linspace(-1, 1, grid_size + 1)
    candidates = np.array([[x, y] for x in coords for y in coords])
    
    # We need 14 points.
    # Initial set: The 4 corners of the square (or a regular 4-gon)
    # Let's start with the 4 corners of the bounding box to maximize hull area initially?
    # No, Heilbronn is about min area. Starting with a regular polygon is better for symmetry.
    # But we are restricted to grid points.
    # Let's start with the 4 corners of the largest square in the grid.
    # If grid is 5x5 from -1 to 1, corners are (-1,-1), (1,-1), (1,1), (-1,1).
    # Area of these 4 is 4.
    
    # Alternative: Start with a regular 14-gon? No, we must pick from grid.
    # Let's try starting with a small set that is well-distributed.
    # Maybe start with the center and 3 points forming an equilateral-ish triangle?
    
    # Let's just start with the 4 corners and add points one by one.
    # Current set: [(-1,-1), (1,-1), (1,1), (-1,1)]
    current_set = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    remaining_candidates = candidates[~np.isin(candidates, current_set, assume_unique=False, invert=False).any(axis=1)]
    # np.isin on 2D arrays is tricky. Let's do manual filtering.
    remaining_candidates = []
    for c in candidates:
        if not any(np.allclose(c, p) for p in current_set):
            remaining_candidates.append(c)
    remaining_candidates = np.array(remaining_candidates)
    
    while len(current_set) < N and len(remaining_candidates) > 0:
        if time.time() > deadline - 20:
            break
        best_candidate = None
        best_min_area = -1
        
        for c in remaining_candidates:
            # Tentatively add c
            temp_set = np.vstack([current_set, [c]])
            # Calculate min triangle area involving c and any two existing points
            # And existing min area
            # For efficiency, we can just calculate all new triangles involving c
            new_areas = []
            for i in range(len(current_set)):
                for j in range(i + 1, len(current_set)):
                    p1 = current_set[i]
                    p2 = current_set[j]
                    area = 0.5 * np.abs((p2[0] - p1[0]) * (c[1] - p1[1]) - (c[0] - p1[0]) * (p2[1] - p1[1]))
                    new_areas.append(area)
            
            # The new min area is min(old_min_area, min(new_areas))
            # We need to track old_min_area.
            # For simplicity in this greedy step, let's just compute the full min area of temp_set
            # This is O(N^3) per candidate, N=14, candidates ~ 20. 14*14*14/6 * 20 * 10 steps ~ 200k ops. Very fast.
            areas = _all_triangle_areas(temp_set)
            min_area = np.min(areas)
            
            if min_area > best_min_area:
                best_min_area = min_area
                best_candidate = c
        
        if best_candidate is not None:
            current_set = np.vstack([current_set, [best_candidate]])
            # Remove best_candidate from remaining
            mask = ~np.isin(remaining_candidates, [best_candidate], assume_unique=False).all(axis=1)
            remaining_candidates = remaining_candidates[mask]
        else:
            break
            
    return current_set

def _refine_points(points, start_time=None, deadline=None):
    """
    Refine the points using L-BFGS-B to maximize the soft-min area.
    """
    if start_time is None:
        start_time = time.time()
    if deadline is None:
        deadline = start_time + 140.0
        
    def objective(params):
        pts = params.reshape(-1, 2)
        areas = _all_triangle_areas(pts)
        if len(areas) == 0:
            return 1e10
        min_val = np.min(areas)
        # Use a soft min to make it differentiable
        # beta controls sharpness
        beta = 100.0
        shifted = areas - min_val
        log_sum = np.log(np.sum(np.exp(-beta * shifted)))
        soft_min = min_val - log_sum / beta
        
        hull_area = _convex_hull_area(pts)
        if hull_area < 1e-15:
            return 1e10
        return -soft_min / hull_area

    try:
        result = minimize(
            objective,
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        if result.fun < 0:
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    return points

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Strategy:
    1. Greedy construction on a 5x5 grid.
    2. Local refinement with L-BFGS-B.
    3. Try multiple grid scales/orientations if time permits.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_points = None
    best_score = -1
    
    # Strategy 1: Greedy on 5x5 grid
    try:
        greedy_pts = _greedy_lattice_construction(grid_size=5, start_time=start_time, deadline=deadline)
        if len(greedy_pts) >= 3:
            # Normalize to unit area
            hull_area = _convex_hull_area(greedy_pts)
            if hull_area > 1e-15:
                scale = np.sqrt(1.0 / hull_area)
                greedy_pts = greedy_pts * scale
            score = _score(greedy_pts)
            if score > best_score:
                best_score = score
                best_points = greedy_pts.copy()
    except Exception:
        pass

    # Strategy 2: Refine the greedy result
    if best_points is not None and time.time() < deadline - 20:
        refined_pts = _refine_points(best_points, start_time=start_time, deadline=deadline)
        hull_area = _convex_hull_area(refined_pts)
        if hull_area > 1e-15:
            scale = np.sqrt(1.0 / hull_area)
            refined_pts = refined_pts * scale
        score = _score(refined_pts)
        if score > best_score:
            best_score = score
            best_points = refined_pts.copy()

    # Strategy 3: Try a different grid size (e.g., 4x4 is too small for 14 pts? 4x4=16, so it's possible)
    if time.time() < deadline - 20:
        try:
            greedy_pts2 = _greedy_lattice_construction(grid_size=4, start_time=start_time, deadline=deadline)
            if len(greedy_pts2) >= 3:
                hull_area = _convex_hull_area(greedy_pts2)
                if hull_area > 1e-15:
                    scale = np.sqrt(1.0 / hull_area)
                    greedy_pts2 = greedy_pts2 * scale
                score = _score(greedy_pts2)
                if score > best_score:
                    best_score = score
                    best_points = greedy_pts2.copy()
                    
                    # Refine this one too
                    if time.time() < deadline - 10:
                        refined_pts2 = _refine_points(best_points, start_time=start_time, deadline=deadline)
                        hull_area2 = _convex_hull_area(refined_pts2)
                        if hull_area2 > 1e-15:
                            scale2 = np.sqrt(1.0 / hull_area2)
                            refined_pts2 = refined_pts2 * scale2
                        score2 = _score(refined_pts2)
                        if score2 > best_score:
                            best_score = score2
                            best_points = refined_pts2.copy()
        except Exception:
            pass
            
    # Fallback: Regular 14-gon
    if best_points is None or time.time() < deadline - 5:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        fallback_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        hull_area = _convex_hull_area(fallback_pts)
        if hull_area > 1e-15:
            scale = np.sqrt(1.0 / hull_area)
            fallback_pts = fallback_pts * scale
        fallback_score = _score(fallback_pts)
        if best_points is None or fallback_score > best_score:
            best_points = fallback_pts

    return best_points
# EVOLVE-BLOCK-END
