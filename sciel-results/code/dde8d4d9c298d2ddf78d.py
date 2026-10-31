"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def is_inside_triangle(p):
    """Check if point p is inside the triangle (V0, V1, V2)."""
    # Using barycentric coordinates
    # p = b0*V0 + b1*V1 + b2*V2 with b0+b1+b2=1 and bi >= 0
    # We can solve for b1, b2 from:
    # p_x = b1*1 + b2*0.5
    # p_y = b2*sqrt(3)/2
    # b0 = 1 - b1 - b2
    b2 = p[1] / (np.sqrt(3) / 2)
    b1 = p[0] - 0.5 * b2
    b0 = 1 - b1 - b2
    eps = 1e-9
    return (b0 >= -eps) and (b1 >= -eps) and (b2 >= -eps)

def project_to_triangle(p):
    """Project point p onto the triangle if it's outside, else return p."""
    if is_inside_triangle(p):
        return p
    # Simple projection: find closest point on boundary
    # This is a simplification; for optimization, we can use penalties or bounds
    # For now, let's just clip to the bounding box and hope DE handles it, 
    # or better, use a transformation that maps a square to the triangle.
    # Let's use the standard mapping:
    # u, v in [0,1] with u+v <= 1
    # p = u*V1 + v*V2
    # This maps the square [0,1]x[0,1] with u+v<=1 to the triangle.
    # But DE works in a hypercube. We can use the transformation:
    # Let x, y be the DE variables in [0,1].
    # If x+y > 1, we need to map back.
    # Alternatively, we can just use the barycentric coordinates directly as variables.
    # Let b1, b2 be in [0,1] with b1+b2 <= 1.
    # This is not a hypercube.
    # Let's use the transformation:
    # u = x * (1 - y)
    # v = y * (1 - x)  ... no, that's not right.
    
    # Standard method: use 2 variables u, v in [0,1].
    # If u + v > 1, set u = 1 - v, v = v (or similar).
    # Actually, a common trick is:
    # u = x * (1 - y)
    # v = y * (1 - x)
    # Then u + v = x - xy + y - xy = x + y - 2xy.
    # This doesn't guarantee u+v <= 1.
    
    # Let's use the barycentric coordinates with a penalty or just rely on DE's bounds.
    # We can define the variables as b1, b2 and use bounds [0,1] for each, 
    # and add a penalty if b1+b2 > 1.
    # Or, we can transform:
    # Let s, t be in [0,1].
    # b1 = s * (1 - t)
    # b2 = t * (1 - s)
    # Then b1 + b2 = s - st + t - st = s + t - 2st.
    # Max of s+t-2st on [0,1]x[0,1] is 0.5 (at s=t=0.5).
    # So this maps to a smaller triangle. Not ideal.
    
    # Let's stick to the simple approach: use x, y as coordinates in [0,1]x[0,1],
    # and if the point is outside, project it.
    # For DE, we can just use bounds [0,1] for x and [0, sqrt(3)/2] for y,
    # and check if it's inside. If not, we can use a penalty.
    
    # Actually, the best approach is to use the barycentric coordinates directly.
    # Let's define the variables as (b1, b2) with bounds [0,1] for each.
    # Then p = b1*V1 + b2*V2.
    # We need b1 + b2 <= 1.
    # We can add a penalty term to the objective function if b1 + b2 > 1.
    
    return p

def triangle_area(p1, p2, p3):
    """Calculate the area of a triangle given three points."""
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))

def min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    """Objective function for optimization. We want to maximize the min area,
    so we minimize the negative min area."""
    # x is a vector of length 2*N, containing [x0, y0, x1, y1, ..., x10, y10]
    points = x.reshape(N, 2)
    
    # Check if all points are inside the triangle
    for p in points:
        if not is_inside_triangle(p):
            # Return a large penalty
            return 1e6
    
    # Calculate the min area
    min_area = min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 seconds margin
    
    # Initial guess: random points inside the triangle
    rng = np.random.RandomState(42)
    
    # Use barycentric coordinates for initialization
    # b1, b2 in [0,1] with b1+b2 <= 1
    points = np.zeros((N, 2))
    for i in range(N):
        while True:
            b1 = rng.random()
            b2 = rng.random()
            if b1 + b2 <= 1:
                points[i] = b1 * V1 + b2 * V2
                break
    
    # Convert to Cartesian coordinates for DE
    # DE will work in [0,1]x[0,sqrt(3)/2] space
    # But we need to ensure points stay inside.
    # Let's use the barycentric coordinates as the variables for DE.
    # Variables: [b1_0, b2_0, b1_1, b2_1, ..., b1_10, b2_10]
    # Bounds: [0,1] for each
    # Penalty if b1+b2 > 1
    
    def objective_bary(x):
        """Objective function using barycentric coordinates."""
        b = x.reshape(N, 2)
        
        # Check if all points are inside
        for i in range(N):
            if b[i, 0] + b[i, 1] > 1 + 1e-9:
                return 1e6
        
        # Convert to Cartesian
        points = np.zeros((N, 2))
        for i in range(N):
            points[i] = b[i, 0] * V1 + b[i, 1] * V2
        
        # Calculate min area
        min_area = min_triangle_area(points)
        return -min_area
    
    # Initial point in barycentric space
    bary_init = np.zeros(N * 2)
    for i in range(N):
        while True:
            b1 = rng.random()
            b2 = rng.random()
            if b1 + b2 <= 1:
                bary_init[2*i] = b1
                bary_init[2*i+1] = b2
                break
    
    # Run DE
    best_result = None
    best_score = float('-inf')
    
    try:
        result = differential_evolution(
            objective_bary,
            bounds=[(0, 1)] * (2 * N),
            seed=42,
            maxiter=50,
            popsize=10,
            tol=1e-6,
            atol=1e-8,
            init='latinhypercube',
            polish=True
        )
        
        if time.time() < deadline:
            bary_best = result.x.reshape(N, 2)
            points_best = np.zeros((N, 2))
            for i in range(N):
                points_best[i] = bary_best[i, 0] * V1 + bary_best[i, 1] * V2
            
            score = min_triangle_area(points_best)
            if score > best_score:
                best_score = score
                best_result = points_best.copy()
                
    except Exception as e:
        pass
    
    # If we have time, try a few local optimizations
    if best_result is not None and time.time() < deadline:
        # Try to improve with local search
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            
            # Use Nelder-Mead on the current best
            try:
                local_result = minimize(
                    objective_bary,
                    best_result.reshape(-1),  # This is wrong, we need barycentric
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                if local_result.success:
                    bary_local = local_result.x.reshape(N, 2)
                    points_local = np.zeros((N, 2))
                    for i in range(N):
                        points_local[i] = bary_local[i, 0] * V1 + bary_local[i, 1] * V2
                    
                    score = min_triangle_area(points_local)
                    if score > best_score:
                        best_score = score
                        best_result = points_local.copy()
                        
            except Exception as e:
                pass
    
    # Return the best result found
    if best_result is not None:
        return best_result
    else:
        # Fallback: return the initial random points
        return points

# EVOLVE-BLOCK-END
