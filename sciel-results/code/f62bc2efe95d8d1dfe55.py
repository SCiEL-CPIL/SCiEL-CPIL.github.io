# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, ConvexHull


def _triangle_area(p1, p2, p3):
    """Compute the area of a triangle given three points."""
    return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0


def _compute_min_triangle_area(points):
    """Compute the minimum area among all triangles formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def _compute_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _compute_score(points):
    """Compute the score: A_min / A_hull."""
    a_min = _compute_min_triangle_area(points)
    a_hull = _compute_hull_area(points)
    if a_hull <= 0 or not np.isfinite(a_min):
        return 0.0
    return a_min / a_hull


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Uses a Delaunay-based local optimization strategy:
    1. Initialize points on a regular 13-gon.
    2. Iteratively identify minimum-area triangles and move their vertices toward the centroid
       of adjacent triangles to increase the minimum area while preserving the convex hull structure.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    
    # Initialize on a regular 13-gon (unit circle)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # Local optimization iterations
    max_iterations = 2000
    step_size = 0.05
    patience = 50
    no_improve_count = 0
    
    for iteration in range(max_iterations):
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(points)
        except Exception:
            break
        
        # Find all triangles in the Delaunay triangulation and their areas
        tri_areas = []
        tri_indices = []
        for simplex in tri.simplices:
            i, j, k = simplex
            area = _triangle_area(points[i], points[j], points[k])
            tri_areas.append(area)
            tri_indices.append(simplex)
        
        if not tri_areas:
            break
        
        tri_areas = np.array(tri_areas)
        min_tri_idx = np.argmin(tri_areas)
        min_area_tri = tri_indices[min_tri_idx]
        
        # For each vertex in the minimum-area triangle, try moving it toward the centroid
        # of adjacent triangles
        improved = False
        for vertex_idx in min_area_tri:
            # Find all triangles containing this vertex
            adjacent_triangles = []
            for t_idx, simplex in enumerate(tri_indices):
                if vertex_idx in simplex:
                    adjacent_triangles.append(simplex)
            
            if not adjacent_triangles:
                continue
            
            # Compute centroid of adjacent triangles
            centroid = np.zeros(2)
            count = 0
            for simplex in adjacent_triangles:
                for idx in simplex:
                    centroid += points[idx]
                    count += 1
            centroid /= count
            
            # Try moving the vertex toward the centroid
            direction = centroid - points[vertex_idx]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction = direction / norm
            
            # Try several step sizes
            for step in [step_size, step_size * 0.5, step_size * 0.1, step_size * 2]:
                new_points = points.copy()
                new_points[vertex_idx] = points[vertex_idx] + step * direction
                
                # Ensure points remain in reasonable bounds (within [-2, 2])
                if not np.all(np.abs(new_points) <= 2):
                    continue
                
                # Ensure convex hull doesn't collapse
                try:
                    hull_area = _compute_hull_area(new_points)
                    if hull_area <= 0:
                        continue
                except Exception:
                    continue
                
                new_score = _compute_score(new_points)
                if new_score > best_score * (1 + 1e-8):
                    best_points = new_points.copy()
                    best_score = new_score
                    improved = True
                    no_improve_count = 0
                    break
            else:
                # No improvement for this vertex
                continue
            
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            # Decay step size
            step_size *= 0.95
            if no_improve_count >= patience:
                break
        else:
            # Keep current points as the working set
            points = best_points.copy()
    
    return best_points


# EVOLVE-BLOCK-END
