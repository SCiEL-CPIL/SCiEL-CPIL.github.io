# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n=13 points on or inside a convex region to maximize the area of the
    smallest triangle formed by these points.
    
    Strategy: Use a structured grid-based initial guess (e.g., a 4x4 grid minus one corner, or a 
    hexagonal packing arrangement) and then refine via local search / optimization to maximize 
    the minimum triangle area relative to the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    
    def min_triangle_area(points):
        """Compute the minimum area among all triangles formed by triples of points."""
        n_pts = len(points)
        min_area = float('inf')
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                for k in range(j+1, n_pts):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0
                    if area < min_area:
                        min_area = area
        return min_area
    
    def hull_area(points):
        """Compute the area of the convex hull of the points."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0
    
    def score(points):
        """Compute the score: A_min / A_hull."""
        a_min = min_triangle_area(points)
        a_hull = hull_area(points)
        if a_hull <= 1e-12:
            return 0.0
        return a_min / a_hull
    
    # Initial guess: Use a structured arrangement
    # Option 1: A 4x4 grid with one point removed (15 points -> remove 2 to get 13)
    # Let's try a hexagonal-like packing for 13 points.
    # A common good arrangement for Heilbronn is based on triangular grids.
    
    # Start with a triangular grid arrangement.
    # For 13 points, we can use a 4-row triangular arrangement:
    # Row 0: 1 point
    # Row 1: 2 points
    # Row 2: 3 points
    # Row 3: 4 points
    # Row 4: 3 points
    # Total: 1+2+3+4+3 = 13 points
    
    # Let's construct this triangular arrangement
    points = []
    spacing = 1.0
    # Row 0: 1 point at (0, 0)
    points.append([0, 0])
    # Row 1: 2 points
    points.append([-0.5*spacing, spacing*np.sqrt(3)/2])
    points.append([0.5*spacing, spacing*np.sqrt(3)/2])
    # Row 2: 3 points
    for i in range(3):
        points.append([(i-1)*spacing, spacing*np.sqrt(3)])
    # Row 3: 4 points
    for i in range(4):
        points.append([(i-1.5)*spacing, spacing*np.sqrt(3)*1.5])
    # Row 4: 3 points
    for i in range(3):
        points.append([(i-1)*spacing, spacing*np.sqrt(3)*2])
    
    points = np.array(points, dtype=float)
    
    # Normalize to unit square for better conditioning
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    points = (points - min_coords) / (max_coords - min_coords)
    
    # Local search to improve the arrangement
    best_points = points.copy()
    best_score = score(best_points)
    
    # Simple local search with random perturbations
    rng = np.random.default_rng(seed=123)
    step_size = 0.05
    n_iterations = 5000
    
    for it in range(n_iterations):
        # Pick a random point and perturb it
        idx = rng.integers(0, n)
        new_points = best_points.copy()
        perturbation = rng.normal(0, step_size, size=2)
        new_points[idx] += perturbation
        
        # Keep points within bounds [0, 1] x [0, 1]
        new_points = np.clip(new_points, 0, 1)
        
        new_score = score(new_points)
        
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
    
    return best_points


# EVOLVE-BLOCK-END
