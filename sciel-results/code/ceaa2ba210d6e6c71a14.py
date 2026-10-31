"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3.0)
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])

# EVOLVE-BLOCK-START

def _area(a, b, c):
    """Compute area of triangle formed by points a, b, c."""
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))

def _min_area(points):
    """Compute the minimum area among all triangles formed by 3 points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def _soft_min_area(points, temp=0.05):
    """
    Smooth surrogate for the minimum area.
    Uses a log-sum-exp based soft-min on the log of areas (or just areas) to handle non-smoothness.
    Actually, min area is better approximated by a soft-min of the areas directly if we want to maximize it.
    Let A be the vector of areas. We want to maximize min(A).
    Soft-min: -log(sum(exp(-A_i/temp))) * temp.
    Maximize this value.
    """
    n = len(points)
    areas = np.zeros((n * (n-1) * (n-2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = points[i], points[j], points[k]
                areas[idx] = _area(a, b, c)
                idx += 1
    
    # Avoid log(0) or exp overflow
    # If any area is very small, the soft-min will be very small (negative log sum exp of negative large numbers)
    # Wait, soft-min of A is -temp * log(sum(exp(-A_i/temp)))
    # As A_i -> 0, exp(-A_i/temp) -> 1.
    # Let's just compute it.
    # To be numerically stable:
    min_val = np.min(areas)
    shifted = areas - min_val
    # sum(exp(-(shifted + min_val)/temp)) = exp(-min_val/temp) * sum(exp(-shifted/temp))
    # log(sum(...)) = -min_val/temp + log(sum(exp(-shifted/temp)))
    # soft_min = -temp * (-min_val/temp + log(sum(...))) = min_val - temp * log(sum(exp(-shifted/temp)))
    
    sum_exp = np.sum(np.exp(-shifted / temp))
    soft_min_val = min_val - temp * np.log(sum_exp)
    
    return soft_min_val

def _objective(x, temp=0.05):
    """
    Objective function to minimize.
    We want to maximize min area, so we minimize -min_area.
    Using soft surrogate: minimize -soft_min_area.
    Also add a penalty if points go outside the triangle.
    """
    points = x.reshape((-1, 2))
    
    # Check if points are inside the triangle
    # Barycentric coordinates
    # For a point p, barycentric coordinates (u, v, w) with u+v+w=1
    # u = area(p, V1, V2) / area(V0, V1, V2)
    # v = area(V0, p, V2) / area(V0, V1, V2)
    # w = area(V0, V1, p) / area(V0, V1, V2)
    # All must be >= 0 (with small tolerance)
    
    total_area = _area(V0, V1, V2)
    
    penalty = 0.0
    for p in points:
        u = _area(p, V1, V2) / total_area
        v = _area(V0, p, V2) / total_area
        w = _area(V0, V1, p) / total_area
        
        # If any coordinate is negative, it's outside.
        # Penalty proportional to how far outside
        if u < 0: penalty += (-u) ** 2
        if v < 0: penalty += (-v) ** 2
        if w < 0: penalty += (-w) ** 2
        
    # The soft min area
    val = -_soft_min_area(points, temp)
    
    return val + 100.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial best solution
    # Use a heuristic initial layout: vertices + center + edge midpoints + some interior points
    # Or just random points inside.
    
    best_x = None
    best_val = float('inf')
    
    # Generate diverse initial points
    # 1. Regular grid inside triangle
    # 2. Random points
    # 3. Voronoi-like (farthest point sampling)
    
    inits = []
    
    # 1. Grid
    # Sample points in a grid and keep those inside
    grid_pts = []
    for i in range(10):
        for j in range(10):
            u = i / 9.0
            v = j / 9.0
            if u + v <= 1.0:
                p = (1 - u - v) * V0 + u * V1 + v * V2
                grid_pts.append(p)
    
    # Randomly pick 11 from grid? Or just use a subset.
    # Let's try to place 11 points roughly evenly.
    # A known good configuration for 11 points might be a hexagonal packing or similar.
    # Let's just use random restarts with a good initial guess.
    
    # Initial guess: 
    # 3 vertices, 3 edge midpoints, 1 center, 4 more?
    # Let's try a specific pattern.
    
    # Pattern 1: Vertices + Midpoints + Center
    pts1 = [
        V0, V1, V2,
        (V0 + V1) / 2.0, (V1 + V2) / 2.0, (V2 + V0) / 2.0,
        (V0 + V1 + V2) / 3.0
    ]
    # Need 4 more. Let's place them at centroids of the 4 sub-triangles formed by center and edges?
    # Sub-triangles: (V0, V1, C), (V1, V2, C), (V2, V0, C)
    # Centroids:
    c1 = (V0 + V1 + (V0+V1+V2)/3.0) / 3.0
    c2 = (V1 + V2 + (V0+V1+V2)/3.0) / 3.0
    c3 = (V2 + V0 + (V0+V1+V2)/3.0) / 3.0
    # We have 3+3+1+3 = 10 points. Need 1 more.
    # Maybe split one of the sub-triangles? Or just randomize the last one.
    # Let's just add a random point inside for the 11th.
    
    # Instead of hardcoding, let's use a few random restarts and one structured restart.
    
    rng = np.random.default_rng(42)
    
    # Structured init
    x0_struct = np.array(pts1 + [rng.uniform(0, 0.5, 2)]) # Last point random
    # Ensure last point is inside
    while not np.all(x0_struct[-1] >= 0) or _area(x0_struct[-1], V1, V2) < 0 or _area(V0, x0_struct[-1], V2) < 0 or _area(V0, V1, x0_struct[-1]) < 0:
        x0_struct[-1] = rng.uniform(0, 1, 2)
        # Simple rejection sampling for triangle
        u, v = x0_struct[-1]
        if u + v > 1.0:
            continue
        x0_struct[-1] = u * V1 + v * V2 # Wait, this is wrong mapping.
        # p = (1-u-v)V0 + u V1 + v V2
        # x = u + 0.5v, y = sqrt(3)/2 v
        # Let's just use barycentric to cartesian properly.
        break
    
    # Proper barycentric to cartesian
    def bary_to_cart(u, v, w):
        return u * V0 + v * V1 + w * V2
    
    # Re-generate structured init properly
    # 3 vertices, 3 midpoints, 1 center, 3 sub-centroids
    # 10 points.
    # 11th point: maybe midpoint of one of the sub-centroids to a vertex?
    # Let's just use random points for the rest of the restarts.
    
    inits.append(x0_struct.flatten())
    
    # Random inits
    for _ in range(5):
        pts = []
        while len(pts) < 11:
            u, v = rng.random(2)
            if u + v < 1.0:
                p = (1 - u - v) * V0 + u * V1 + v * V2
                pts.append(p)
        inits.append(np.array(pts).flatten())
    
    # Optimization loop
    temp = 0.05
    
    for x0 in inits:
        if time.time() > deadline:
            break
            
        # Use L-BFGS-B
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', args=(temp,),
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < best_val:
                # Check validity
                x_opt = res.x
                points = x_opt.reshape((-1, 2))
                
                # Verify inside
                valid = True
                for p in points:
                    u = _area(p, V1, V2) / _area(V0, V1, V2)
                    v = _area(V0, p, V2) / _area(V0, V1, V2)
                    w = _area(V0, V1, p) / _area(V0, V1, V2)
                    if u < -1e-6 or v < -1e-6 or w < -1e-6:
                        valid = False
                        break
                
                if valid:
                    # Calculate true min area
                    true_min = _min_area(points)
                    # The objective was -soft_min. 
                    # We want to compare based on true min area.
                    # But we track best_val as objective.
                    # Let's track best_true_min separately.
                    if res.fun < best_val:
                        best_val = res.fun
                        best_x = x_opt.copy()
                        
        except Exception as e:
            pass
            
    # Polish with lower temperature
    if best_x is not None:
        temp = 0.01
        try:
            res = minimize(_objective, best_x, method='L-BFGS-B', args=(temp,),
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                x_opt = res.x
                points = x_opt.reshape((-1, 2))
                valid = True
                for p in points:
                    u = _area(p, V1, V2) / _area(V0, V1, V2)
                    v = _area(V0, p, V2) / _area(V0, V1, V2)
                    w = _area(V0, V1, p) / _area(V0, V1, V2)
                    if u < -1e-6 or v < -1e-6 or w < -1e-6:
                        valid = False
                        break
                if valid:
                    best_val = res.fun
                    best_x = x_opt.copy()
        except Exception:
            pass
            
    if best_x is None:
        # Fallback to random
        pts = []
        while len(pts) < 11:
            u, v = rng.random(2)
            if u + v < 1.0:
                p = (1 - u - v) * V0 + u * V1 + v * V2
                pts.append(p)
        return np.array(pts)
        
    return best_x.reshape((-1, 2))

# EVOLVE-BLOCK-END
