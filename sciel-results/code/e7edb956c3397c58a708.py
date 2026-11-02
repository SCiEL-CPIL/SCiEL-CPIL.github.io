"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def _in_triangle(p, A, B, C):
    """Check if point p is inside triangle ABC (including boundary)."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, A, B)
    d2 = sign(p, B, C)
    d3 = sign(p, C, A)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def _penalty(points, A, B, C):
    """Penalty if any point is outside the triangle."""
    penalty = 0.0
    for p in points:
        if not _in_triangle(p, A, B, C):
            # Distance to nearest edge
            penalty += 10.0
    return penalty

def construct():
    """
    Construct 11 points in the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    maximizing the minimum triangle area.
    
    Uses a 3-4-4 grid topology with spring model and symmetry.
    """
    sqrt3 = np.sqrt(3.0)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, sqrt3 / 2.0])
    
    # Define 11 points in a quasi-lattice: rows of 4, 3, 4 or similar
    # Let's use: bottom row 4, middle row 3, top row 4 = 11
    # But that's not symmetric. Let's use: 3, 4, 4 or 4, 3, 4
    
    # Better: use a hexagonal-like pattern
    # Row 0 (y=0): 4 points
    # Row 1 (y=h): 3 points  
    # Row 2 (y=2h): 4 points
    # Total: 11
    
    # Actually, let's think about this differently.
    # The equilateral triangle has vertices at (0,0), (1,0), (0.5, sqrt(3)/2)
    # Height = sqrt(3)/2
    
    # Let me define a base configuration with symmetry about x=0.5
    
    # Configuration: 
    # Bottom row (y=0): points at x = 0.1, 0.3, 0.7, 0.9 (4 points)
    # Middle row (y=h/2): points at x = 0.2, 0.5, 0.8 (3 points) 
    # Top row (y=h): points at x = 0.3, 0.5, 0.7 (3 points)
    # That's 10. Need 11.
    
    # Let's do:
    # Bottom row (y=0): 4 points
    # Middle row (y=h/2): 4 points
    # Top row (y=h): 3 points
    # Total: 11
    
    h = sqrt3 / 2.0
    
    # Base positions (before optimization)
    # We'll parameterize with global affine transform + local perturbations
    
    # Global affine: x' = a*x + b*y + c, y' = d*x + e*y + f
    # But to maintain triangle shape, we can use: scale, rotate, translate
    # Simpler: just use a linear transform that maps the unit triangle to itself
    
    # Let's use a different approach: define points in a normalized coordinate
    # system and then apply an affine transform
    
    # Base lattice in a unit square-like space, then map to triangle
    
    # Actually, let's just directly optimize 11*2 = 22 variables with constraints
    
    # Spring model: connect nearby points with springs
    # Topology: 
    # Points indexed 0-3: bottom row
    # Points 4-7: middle row
    # Points 8-10: top row
    
    # Base positions (normalized to fit in triangle)
    base = np.array([
        [0.15, 0.05],   # 0
        [0.35, 0.05],   # 1
        [0.65, 0.05],   # 2
        [0.85, 0.05],   # 3
        [0.25, 0.25],   # 4
        [0.45, 0.25],   # 5
        [0.55, 0.25],   # 6
        [0.75, 0.25],   # 7
        [0.35, 0.42],   # 8
        [0.50, 0.42],   # 9
        [0.65, 0.42],   # 10
    ])
    
    # Spring connections (edges in the graph)
    edges = [
        (0, 1), (1, 2), (2, 3),  # bottom row
        (4, 5), (5, 6), (6, 7),  # middle row
        (8, 9), (9, 10),         # top row
        (0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (3, 7),  # bottom to middle
        (4, 8), (5, 8), (5, 9), (6, 9), (6, 10), (7, 10),        # middle to top
    ]
    
    # Rest lengths (ideal distances)
    rest_lengths = {}
    for i, j in edges:
        rest_lengths[(i, j)] = np.linalg.norm(base[i] - base[j])
    
    def objective(params):
        """
        params: 22 values = 11*2 point coordinates
        Returns negative min triangle area + penalty (to minimize)
        """
        pts = params.reshape(11, 2)
        
        # Penalty for being outside triangle
        pen = 0.0
        for p in pts:
            if not _in_triangle(p, A, B, C):
                # Compute distance to outside
                # Use barycentric-like penalty
                pen += 50.0
        
        # Spring potential energy
        spring_energy = 0.0
        for i, j in edges:
            d = np.linalg.norm(pts[i] - pts[j])
            rest = rest_lengths[(i, j)]
            spring_energy += 0.5 * (d - rest) ** 2
        
        # Min triangle area
        min_area = _min_triangle_area(pts)
        
        # We want to maximize min_area, so minimize -min_area
        # But also penalize springs and out-of-bounds
        return -min_area + 0.1 * spring_energy + pen
    
    # Initial guess
    x0 = base.flatten()
    
    # Try multiple initial configurations and pick the best
    best_points = None
    best_score = -float('inf')
    
    # Initial: base configuration
    candidates = [base]
    
    # Add some perturbed versions
    np.random.seed(42)
    for _ in range(5):
        perturbed = base + 0.05 * np.random.randn(11, 2)
        candidates.append(perturbed)
    
    # Also try a more symmetric hexagonal arrangement
    hex_base = np.array([
        [0.1, 0.05],
        [0.3, 0.05],
        [0.5, 0.05],
        [0.7, 0.05],
        [0.2, 0.2],
        [0.4, 0.2],
        [0.6, 0.2],
        [0.8, 0.2],
        [0.3, 0.35],
        [0.5, 0.35],
        [0.7, 0.35],
    ])
    candidates.append(hex_base)
    
    for cand in candidates:
        x0 = cand.flatten()
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-8})
        
        pts = res.x.reshape(11, 2)
        
        # Check if all points are inside triangle
        valid = True
        for p in pts:
            if not _in_triangle(p, A, B, C):
                valid = False
                break
        
        if valid:
            score = _min_triangle_area(pts)
            if score > best_score:
                best_score = score
                best_points = pts
    
    # If no valid points found, use base
    if best_points is None:
        best_points = base
    
    return np.array(best_points)
# EVOLVE-BLOCK-END
