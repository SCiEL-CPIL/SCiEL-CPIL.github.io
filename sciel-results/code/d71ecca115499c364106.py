"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def _triangle_area(p1, p2, p3):
    """Compute the area of the triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by the given points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def _inside_triangle(p, vertices):
    """Check if point p is inside the triangle defined by vertices."""
    # Barycentric coordinate method
    (x1, y1), (x2, y2), (x3, y3) = vertices
    px, py = p
    
    # Compute the sign of cross products
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign((px, py), (x1, y1), (x2, y2))
    d2 = sign((px, py), (x2, y2), (x3, y3))
    d3 = sign((px, py), (x3, y3), (x1, y1))
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)


def construct():
    # Vertices of the equilateral triangle
    h = np.sqrt(3) / 2
    vertices = [(0, 0), (1, 0), (0.5, h)]
    
    # Total area of the equilateral triangle
    total_area = 0.5 * 1 * h  # base=1, height=sqrt(3)/2
    
    # We need to place 11 points. Let's use differential evolution to optimize.
    # The objective is to maximize the minimum triangle area.
    # We'll parameterize the 11 points by 22 coordinates.
    
    # Define bounds for each coordinate
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, h))
    
    def objective(flat_params):
        """Objective function for differential evolution: minimize negative min area."""
        points = []
        for i in range(N):
            x = flat_params[2 * i]
            y = flat_params[2 * i + 1]
            # Check if point is inside the triangle
            if not _inside_triangle((x, y), vertices):
                # Penalize points outside the triangle
                # Find the closest point on the triangle boundary or just return a large penalty
                # Simple penalty: return a large negative value (since we're minimizing -min_area)
                return 1000.0
            points.append((x, y))
        
        # Compute the minimum triangle area
        min_area = _min_triangle_area(points)
        
        # Return negative of the normalized area (we want to maximize, so minimize negative)
        # Normalize by total area to make it scale-invariant
        return -min_area / total_area
    
    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=42,
        maxiter=200,
        popsize=20,
        tol=1e-6,
        polish=False
    )
    
    # Extract the optimal points
    optimal_params = result.x
    points = []
    for i in range(N):
        x = optimal_params[2 * i]
        y = optimal_params[2 * i + 1]
        points.append([x, y])
    
    return np.array(points)
# EVOLVE-BLOCK-END

