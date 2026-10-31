"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 11
S3 = 3 ** 0.5
H = S3 / 2.0

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

def point_in_triangle(p):
    """Check if point p is inside or on the boundary of the triangle ABC."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, A, B)
    d2 = sign(p, B, C)
    d3 = sign(p, C, A)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def objective(x, n=N):
    """Negative of the minimum triangle area (to be minimized)."""
    pts = x.reshape(n, 2)
    # Check if all points are inside the triangle
    # We use a penalty approach: if any point is outside, return a large penalty
    # But better to use constraints. Here we just compute area.
    return -min_triangle_area(pts)

def constraints_def(n=N):
    """Define constraints for SLSQP."""
    constraints = []
    # For each point, it must be inside the triangle
    # Triangle inequalities:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x  (below line AC? No, line AC is from (0,0) to (0.5, H). Slope is H/0.5 = sqrt(3). So y <= sqrt(3)*x for x in [0, 0.5]
    #    Actually, the lines are:
    #    AB: y = 0
    #    BC: y = -sqrt(3)*(x - 1) = sqrt(3)*(1-x)
    #    CA: y = sqrt(3)*x
    #    Inside means:
    #    y >= 0
    #    y <= sqrt(3)*x  (for the left side, but this is only valid for x <= 0.5)
    #    y <= sqrt(3)*(1-x) (for the right side)
    #    A unified way: 
    #    y >= 0
    #    sqrt(3)*x - y >= 0  -> y <= sqrt(3)*x
    #    sqrt(3)*(1-x) - y >= 0 -> y <= sqrt(3)*(1-x)
    
    for i in range(n):
        # y_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        # y_i <= sqrt(3)*x_i
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: S3 * x[2*i] - x[2*i+1]})
        # y_i <= sqrt(3)*(1-x_i)
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: S3 * (1 - x[2*i]) - x[2*i+1]})
    return constraints

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # Initial guess: vertices and center, plus some random points
    # A good starting configuration is often a grid or symmetric arrangement
    # Let's try a few different initializations and pick the best one
    
    best_points = None
    best_area = 0.0
    
    # Strategy 1: Symmetric initialization
    # Place points at vertices, midpoints, and center
    # For 11 points, we can use a hexagonal-like pattern
    
    # Let's try a simple grid-like approach first
    # Or use the known optimal configurations for small N
    
    # For N=11, a good starting point might be:
    # 3 vertices, 3 midpoints, 1 center, and 4 more points
    
    # Let's try multiple random starts and a structured start
    
    # Structured start:
    # Use a hexagonal lattice clipped to the triangle
    # Or just use random points with good seeding
    
    rng = np.random.default_rng(42)
    
    # Try a few initial configurations
    configs = []
    
    # Config 1: Random uniform
    while True:
        pts = []
        for _ in range(N):
            u, v = rng.random(), rng.random()
            if u + v <= 1.0:
                p = [u + 0.5 * v, H * v]
                pts.append(p)
        if len(pts) == N:
            configs.append(np.array(pts))
            break
    
    # Config 2: Vertices + midpoints + center + 4 random
    base_points = [A, B, C, 
                   (A + B) / 2, (B + C) / 2, (C + A) / 2,
                   (A + B + C) / 3]
    # Add 4 more random points
    while len(base_points) < N:
        u, v = rng.random(), rng.random()
        if u + v <= 1.0:
            p = [u + 0.5 * v, H * v]
            base_points.append(p)
    configs.append(np.array(base_points[:N]))
    
    # Config 3: More structured - use a 3x3 grid in barycentric coordinates
    # Barycentric coordinates (a, b, c) with a+b+c=1, a,b,c >= 0
    # Point = a*A + b*B + c*C
    grid_pts = []
    for a in np.linspace(0, 1, 4):
        for b in np.linspace(0, 1-a, 4):
            c = 1 - a - b
            if c < 0:
                continue
            p = a * A + b * B + c * C
            grid_pts.append(p)
    if len(grid_pts) >= N:
        configs.append(np.array(grid_pts[:N]))
    
    # For each config, run SLSQP optimization
    for init_pts in configs:
        if time.time() > deadline - 30:
            break
            
        x0 = init_pts.flatten()
        cons = constraints_def(N)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=[(0, 1)] * (2 * N),
                          constraints=cons, options={'maxiter': 500, 'ftol': 1e-10})
            
            if res.success:
                pts = res.x.reshape(N, 2)
                # Verify points are inside triangle
                if all(point_in_triangle(p) for p in pts):
                    area = min_triangle_area(pts)
                    if area > best_area:
                        best_area = area
                        best_points = pts
        except Exception:
            pass
    
    # If no optimization succeeded, use the best initial config
    if best_points is None:
        for init_pts in configs:
            if all(point_in_triangle(p) for p in init_pts):
                area = min_triangle_area(init_pts)
                if area > best_area:
                    best_area = area
                    best_points = init_pts
    
    # Final polish: run a few more iterations with tighter tolerance if time permits
    if best_points is not None and time.time() < deadline - 10:
        x0 = best_points.flatten()
        cons = constraints_def(N)
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=[(0, 1)] * (2 * N),
                          constraints=cons, options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success:
                pts = res.x.reshape(N, 2)
                if all(point_in_triangle(p) for p in pts):
                    area = min_triangle_area(pts)
                    if area > best_area:
                        best_area = area
                        best_points = pts
        except Exception:
            pass
    
    return best_points if best_points is not None else configs[0]

# EVOLVE-BLOCK-END
