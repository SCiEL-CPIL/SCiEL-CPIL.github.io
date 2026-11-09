# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, ConvexHull


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    
    # Start with a reasonable initial configuration
    # Use a grid-like pattern perturbed slightly to avoid degeneracy
    # For 13 points, we can start with a 4x4 grid minus one corner, or similar
    # Let's try a more structured start: points on a regular polygon + center + inner ring
    
    # Actually, let's use a simpler approach: start with random points but then optimize
    # using the physics-based relaxation described in the directive
    
    # Initial configuration: use a grid pattern
    # 13 points can be arranged as 4x4 grid minus 3 corners, or 3x5 minus 2
    # Let's use a 4x4 grid (16 points) and remove 3 to get 13
    # Or better: start with points roughly evenly spaced
    
    # Let's use a circular arrangement with some internal points
    # Outer ring: 8 points on a circle
    # Inner: 5 points arranged in a smaller circle or grid
    
    # Actually, let me start with a simple grid-like arrangement
    # 13 points: rows of 4, 4, 4, 1? No, let's do 3-4-4-2 or similar
    
    # Simpler: start with random points in [0,1]^2 and optimize
    rng = np.random.default_rng(seed=123)
    points = rng.random((n, 2))
    
    # Optimization: physics-based relaxation
    # We want to maximize the minimum triangle area
    # This is equivalent to maximizing the minimum circumradius of the Delaunay triangles
    # (since area = r * s where s is semiperimeter, but for fixed point set, 
    #  maximizing min area is related to maximizing min circumradius in some sense)
    
    # Actually, the area of a triangle is (1/2) * base * height, or using circumradius:
    # Area = (a*b*c) / (4*R) where R is circumradius
    # So for fixed side lengths, larger R means smaller area. Wait, that's the opposite.
    # Actually: Area = abc/(4R), so for fixed sides, larger R means smaller area.
    # But we want to maximize the minimum area, so we want to avoid small areas.
    
    # Let's directly optimize the minimum triangle area using gradient-free methods
    # with the physics-based relaxation idea
    
    # Use a simple iterative improvement:
    # 1. Compute all 286 triangle areas
    # 2. Find the minimum area triangle
    # 3. Move the vertices of that triangle slightly apart to increase its area
    # 4. Repeat
    
    # But this is slow. Let's use a better approach.
    
    # Use scipy's optimization or implement a simple gradient ascent on the min area
    
    # Alternative: use the fact that optimal Heilbronn configurations have points
    # that are roughly evenly spaced. Start with a good initial guess and refine.
    
    # Let me try a different initial configuration: 
    # Place points on a regular 13-gon? No, that's not convex hull friendly.
    # Better: place some on the boundary of a square and some inside.
    
    # Let's try: 8 points on the perimeter of a square (2 on each side), 
    # 5 points inside in a cross pattern
    
    # Square corners: (0,0), (1,0), (1,1), (0,1)
    # Midpoints of sides: (0.5,0), (1,0.5), (0.5,1), (0,0.5)
    # That's 8 points on the perimeter.
    # Inside: center (0.5,0.5), and 4 points at (0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)
    # That's 5 inside. Total 13.
    
    points = np.array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 0.5],
        [1.0, 1.0], [0.5, 1.0], [0.0, 1.0], [0.0, 0.5],
        [0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75],
        [0.5, 0.5]
    ])
    
    # Now optimize using a simple iterative method
    # Compute the objective: min triangle area / hull area
    # Use coordinate ascent or simple gradient steps
    
    def compute_min_area(pts):
        """Compute the minimum triangle area among all C(n,3) triangles."""
        n_pts = len(pts)
        min_area = float('inf')
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                for k in range(j+1, n_pts):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                    if area < min_area:
                        min_area = area
        return min_area
    
    def compute_hull_area(pts):
        """Compute the area of the convex hull."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except:
            return 0.0
    
    def objective(pts):
        min_area = compute_min_area(pts)
        hull_area = compute_hull_area(pts)
        if hull_area <= 0:
            return 0.0
        return min_area / hull_area
    
    # Simple iterative optimization: perturb each point and keep if it improves
    # This is slow but should work within 600s for n=13
    
    # Use a more efficient approach: for each point, try small moves in x and y
    # and keep the best move
    
    best_points = points.copy()
    best_obj = objective(best_points)
    
    # Number of iterations - keep it modest to stay within time limit
    n_iter = 200
    step_size = 0.05
    
    for it in range(n_iter):
        improved = False
        for i in range(n):
            # Try small perturbations in x and y
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_points = best_points.copy()
                    new_points[i, 0] += dx
                    new_points[i, 1] += dy
                    # Check if all points are still in [0,1]^2 (optional, but keeps things bounded)
                    if np.any(new_points < 0) or np.any(new_points > 1):
                        continue
                    new_obj = objective(new_points)
                    if new_obj > best_obj:
                        best_points = new_points
                        best_obj = new_obj
                        improved = True
        
        if not improved:
            break
    
    # Final cleanup: ensure finite values
    if not np.all(np.isfinite(best_points)):
        # Fallback to the initial structured configuration
        best_points = np.array([
            [0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 0.5],
            [1.0, 1.0], [0.5, 1.0], [0.0, 1.0], [0.0, 0.5],
            [0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75],
            [0.5, 0.5]
        ])
    
    return best_points


# EVOLVE-BLOCK-END
