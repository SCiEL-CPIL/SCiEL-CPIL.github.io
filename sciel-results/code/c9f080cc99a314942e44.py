import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def bary_to_cartesian(bary_coords):
    """
    Convert barycentric coordinates (u, v) to Cartesian coordinates.
    w = 1 - u - v
    P = u*V0 + v*V1 + w*V2
    """
    u = bary_coords[:, 0]
    v = bary_coords[:, 1]
    w = 1.0 - u - v
    # P = u*V0 + v*V1 + w*V2
    # V0 is (0,0), so u*V0 is (0,0)
    x = v * V1[0] + w * V2[0]
    y = v * V1[1] + w * V2[1]
    return np.column_stack((x, y))

def calculate_areas(points):
    """
    Calculate areas of all triangles formed by N points.
    Returns a 1D array of areas.
    """
    n = len(points)
    areas = []
    # Vectorized approach might be faster, but N=11 is small.
    # Number of triangles is C(11,3) = 165.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                areas.append(area)
    return np.array(areas)

def smooth_min_area(areas, beta):
    """
    Smooth approximation of the minimum area.
    We want to maximize min(areas), so we minimize -smooth_min.
    smooth_min = -1/beta * log(sum(exp(-beta * areas)))
    """
    # To avoid overflow, we can subtract max, but areas are small (< 1).
    # exp(-beta * areas) is safe for beta=50, areas ~ 0.01 -> exp(-0.5) ~ 0.6
    # If area is 0, exp(0)=1.
    # Sum is at most 165.
    # log is fine.
    # However, if areas are very small, -beta * areas is close to 0.
    # The function is: -1/beta * log(sum(exp(-beta * A)))
    # This is a smooth lower bound on min(A).
    # Wait, log-sum-exp is a smooth upper bound on max.
    # -log-sum-exp is a smooth lower bound on min?
    # Let f(x) = -1/beta * log(sum exp(-beta x_i)).
    # If x_i = min, others > min.
    # sum exp(-beta x_i) ~ exp(-beta min) * (1 + small).
    # log(sum) ~ -beta min + log(1+small) ~ -beta min.
    # f(x) ~ -1/beta * (-beta min) = min.
    # Yes, it approximates min.
    return -1.0 / beta * np.log(np.sum(np.exp(-beta * areas)))

def objective(x, beta):
    """
    x: flattened array of barycentric coordinates [u1, v1, u2, v2, ...]
    Returns scalar to minimize.
    """
    bary = x.reshape((N, 2))
    points = bary_to_cartesian(bary)
    areas = calculate_areas(points)
    min_area_smooth = smooth_min_area(areas, beta)
    # We want to maximize min_area, so minimize -min_area_smooth
    return -min_area_smooth

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe within 150s
    
    best_points = None
    best_score = -float('inf')
    
    # Define constraints for barycentric coordinates
    # u >= 0, v >= 0, u + v <= 1
    # Bounds for L-BFGS-B: 0 <= u <= 1, 0 <= v <= 1
    # But u+v <= 1 is a linear constraint. L-BFGS-B doesn't support linear inequalities directly, only bounds.
    # So we must use SLSQP or project.
    # Let's use SLSQP.
    
    # Constraints: u_i + v_i <= 1 for all i
    # Inequality constraints in scipy: fun(x) >= 0
    # So: 1 - u_i - v_i >= 0
    
    constraints = []
    for i in range(N):
        def make_constraint(i):
            def constraint(x):
                u = x[2*i]
                v = x[2*i+1]
                return 1.0 - u - v
            return constraint
        constraints.append({'type': 'ineq', 'fun': make_constraint(i)})
        
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        
    # Initializations
    initializations = []
    
    # 1. Symmetric/Structured Initialization
    # Place points in a grid-like pattern within the triangle
    # Or use known good configurations for small N and extend?
    # Let's try a few random ones and one structured.
    
    # Structured: Vertices, Midpoints, Center
    # V0, V1, V2
    # Midpoints: (V0+V1)/2, (V1+V2)/2, (V2+V0)/2
    # Center: (V0+V1+V2)/3
    # We need 11 points.
    # Let's just generate random points for simplicity and robustness, 
    # but ensure they are well spread.
    
    rng = np.random.RandomState(42)
    
    # Generate 5 random initializations
    for _ in range(5):
        # Generate random points in barycentric coords
        # u, v >= 0, u+v <= 1
        # Sample u, v from uniform [0,1] and reject if u+v > 1
        bary_pts = []
        while len(bary_pts) < N:
            u, v = rng.random(2)
            if u + v <= 1.0:
                bary_pts.append([u, v])
        initializations.append(np.array(bary_pts).flatten())
        
    # One more structured init: 
    # Try to place points on a hexagonal lattice clipped to the triangle?
    # Or just use the vertices and midpoints as a base and jitter.
    base_bary = [
        [0.0, 0.0], # V0
        [1.0, 0.0], # V1
        [0.0, 1.0], # V2
        [0.5, 0.0], # Mid V0-V1
        [0.5, 0.5], # Mid V1-V2? No, Mid V1-V2 is (0.5, 0.5) in bary?
        # V1=(1,0), V2=(0,1). Mid = (0.5, 0.5). u+v=1. OK.
        [0.0, 0.5], # Mid V2-V0
        [1/3.0, 1/3.0], # Center
        # Add 4 more randomly but near center?
    ]
    # Add 4 random points
    for _ in range(4):
        u, v = rng.random(2)
        if u + v <= 1.0:
            base_bary.append([u, v])
    initializations.append(np.array(base_bary).flatten())
    
    # Optimization loop
    for i, x0 in enumerate(initializations):
        if time.time() > deadline:
            break
            
        # Use SLSQP
        # Beta annealing: start with low beta, increase?
        # For simplicity, use a fixed beta that balances smoothness and accuracy.
        # Beta=50 is usually good.
        beta = 50.0
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-9}
            )
            
            if res.success or res.fun < 0: # res.fun is -area, so negative is good
                x_opt = res.x
                bary_opt = x_opt.reshape((N, 2))
                
                # Check validity
                valid = True
                for u, v in bary_opt:
                    if u < -1e-6 or v < -1e-6 or u + v > 1.0 + 1e-6:
                        valid = False
                        break
                
                if valid:
                    points_opt = bary_to_cartesian(bary_opt)
                    score = calculate_min_area(points_opt)
                    
                    if score > best_score:
                        best_score = score
                        best_points = points_opt
                        
        except Exception as e:
            pass
            
    # If no valid point found, fallback
    if best_points is None:
        # Return random valid points
        rng = np.random.RandomState(7)
        bary_pts = []
        while len(bary_pts) < N:
            u, v = rng.random(2)
            if u + v <= 1.0:
                bary_pts.append([u, v])
        best_points = bary_to_cartesian(np.array(bary_pts))
        
    return best_points
