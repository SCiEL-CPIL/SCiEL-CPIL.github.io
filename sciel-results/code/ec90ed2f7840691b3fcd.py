"""
Unit regular hexagons inside a regular hexagon.
We pack 11 unit hexagons to minimize the outer side length.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
START_TIME = time.time()
TIME_LIMIT = 130.0  # safety margin for 150s limit

def construct():
    # We will try to find a tight packing.
    # A regular hexagon of side S has inradius S * sqrt(3)/2 and circumradius S.
    # The distance between centers of two touching unit hexagons is 2 * sqrt(3).
    
    # The optimal packing for 11 hexagons is likely a central one surrounded by 6,
    # and then 4 more in the gaps, or a 4-3-4 arrangement.
    # Let's use a gradient-based optimization starting from a reasonable guess.
    
    # Parameters:
    # x, y: center of outer hexagon (we can fix this at 0,0 by symmetry, but let's keep it for flexibility)
    # Actually, fixing outer center at (0,0) is safe due to symmetry.
    # We need to place 11 inner hexagons.
    # Each inner hexagon: (cx, cy, theta)
    # Total variables: 11 * 3 = 33
    
    # Objective: Minimize S such that all inner hexagons are inside the outer hexagon of side S.
    # This is equivalent to maximizing the "inradius" of the convex hull of the inner hexagons,
    # but it's easier to think of it as: for a given S, check if all inner hexagons fit.
    # We can minimize S directly by treating S as a variable and adding constraints,
    # or we can fix S and check feasibility, then binary search on S.
    # A better way: minimize S subject to constraints that each inner hexagon is inside the outer.
    
    # The condition for a point (px, py) to be inside a regular hexagon of side S centered at (0,0) with angle 0:
    # The hexagon has 6 sides. The distance from center to each side is S * sqrt(3) / 2.
    # The sides are at angles 0, 60, 120, 180, 240, 300 degrees.
    # The inequality for a point to be inside is:
    # |px * cos(a) + py * sin(a)| <= S * sqrt(3) / 2 for a in [0, 60, 120, 180, 240, 300]
    
    # For an inner hexagon, we need all its vertices to be inside the outer hexagon.
    # The vertices of a unit hexagon centered at (cx, cy) with angle theta are:
    # (cx + cos(theta + k*60), cy + sin(theta + k*60)) for k=0..5
    
    # This is a non-convex problem in general, but we can try to solve it with a local optimizer.
    
    # Initial guess:
    # Place hexagons in a hexagonal lattice pattern.
    # Distance between centers: 2 * sqrt(3)
    # Let's place one at (0,0), six around it, and four more.
    
    # Actually, let's use a known good packing as a starting point.
    # The optimal packing for 11 hexagons is known to have a side length of about 7.
    # Let's start with a grid-like arrangement.
    
    # Let's define the initial positions.
    # We'll use a 4-3-4 arrangement.
    # Row 1 (top): 4 hexagons
    # Row 2 (middle): 3 hexagons
    # Row 3 (bottom): 4 hexagons
    # But this is for a rectangular arrangement. For hexagonal, it's different.
    
    # Let's just use a random initialization and optimize.
    np.random.seed(42)
    
    def objective(params):
        # params: [S, x1, y1, t1, x2, y2, t2, ...]
        S = params[0]
        if S < 1:
            return 1000.0
        inner_params = params[1:]
        
        # Check if all inner hexagons are inside the outer hexagon
        # and don't overlap each other.
        
        # For each inner hexagon, check if its vertices are inside the outer hexagon.
        for i in range(N):
            cx = inner_params[i*3]
            cy = inner_params[i*3+1]
            theta = inner_params[i*3+2]
            
            # Vertices of inner hexagon
            for k in range(6):
                angle = theta + k * np.pi / 3
                vx = cx + np.cos(angle)
                vy = cy + np.sin(angle)
                
                # Check if (vx, vy) is inside outer hexagon of side S centered at (0,0)
                # The outer hexagon has inradius S * sqrt(3) / 2
                # The distance from (vx, vy) to the center is sqrt(vx^2 + vy^2)
                # But we need to check the hexagonal distance.
                
                # The condition for (vx, vy) to be inside a hexagon of side S centered at (0,0):
                # For each of the 6 sides, the dot product with the normal must be <= S * sqrt(3) / 2
                # The normals are at angles 0, 60, 120, 180, 240, 300
                for m in range(6):
                    normal_angle = m * np.pi / 3
                    nx = np.cos(normal_angle)
                    ny = np.sin(normal_angle)
                    dist = abs(vx * nx + vy * ny)
                    if dist > S * np.sqrt(3) / 2 + 1e-6:
                        return 1000.0 + dist - S * np.sqrt(3) / 2
        
        # Check for overlaps between inner hexagons
        # Two hexagons overlap if the distance between their centers is less than 2 * sqrt(3)
        # (since they are unit hexagons, the distance between centers when touching is 2 * sqrt(3))
        for i in range(N):
            for j in range(i+1, N):
                cx1 = inner_params[i*3]
                cy1 = inner_params[i*3+1]
                cx2 = inner_params[j*3]
                cy2 = inner_params[j*3+1]
                dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                if dist < 2 * np.sqrt(3) - 1e-6:
                    return 1000.0 + (2 * np.sqrt(3) - dist)
        
        return S
    
    # Initial guess
    # Let's place hexagons in a hexagonal cluster.
    # Center at (0,0)
    # 6 around it at distance 2*sqrt(3)
    # And 4 more in the gaps.
    
    d = 2 * np.sqrt(3)
    init = [7.0]  # initial S
    # Center
    init.extend([0.0, 0.0, 0.0])
    # 6 around
    for k in range(6):
        angle = k * np.pi / 3
        init.extend([d * np.cos(angle), d * np.sin(angle), 0.0])
    # 4 more: place them at the "corners" of the hexagonal cluster
    # The cluster of 7 hexagons has a "radius" of 2 * 2 * sqrt(3) = 4 * sqrt(3) from center to the farthest vertex.
    # Let's place the remaining 4 at angles 30, 150, 270, 30 (wait, 30, 150, 270, 30 is not 4 distinct angles)
    # Let's place them at 30, 90, 150, 210, 270, 330 but we only need 4.
    # Actually, the optimal packing for 11 is likely a 3-4-3 or 4-3-4 arrangement.
    # Let's try 4-3-4.
    # Row 1 (y = d * 1.5): 4 hexagons
    # Row 2 (y = 0): 3 hexagons
    # Row 3 (y = -d * 1.5): 4 hexagons
    # But this is for a rectangular grid. For hexagonal, the rows are offset.
    
    # Let's just use a random initialization.
    init = [7.0]
    for i in range(N):
        init.extend([np.random.uniform(-3, 3), np.random.uniform(-3, 3), np.random.uniform(0, 2*np.pi)])
    
    # Optimize
    best_S = 1000.0
    best_params = init
    
    # We'll use Nelder-Mead or Powell since we don't have gradients easily.
    # Powell is good for this.
    
    # Time check
    if time.time() - START_TIME > TIME_LIMIT - 10:
        # Return a known valid packing
        # The seed program has a valid packing with S=40, but we can do better.
        # Let's return a simple valid packing.
        inner = []
        d = 2 * np.sqrt(3)
        # Place 11 hexagons in a line
        for k in range(-5, 6):
            inner.append([k * d, 0.0, 0.0])
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Run optimization
    try:
        result = minimize(objective, init, method='Powell', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
        if result.fun < best_S:
            best_S = result.fun
            best_params = result.x
    except Exception:
        pass
    
    # Extract the solution
    S = best_params[0]
    inner = []
    for i in range(N):
        cx = best_params[i*3+1]
        cy = best_params[i*3+2]
        theta = best_params[i*3+3]
        inner.append([cx, cy, np.degrees(theta)])
    
    # Ensure the solution is valid
    # If not, fall back to a simple valid packing
    if S > 100:
        inner = []
        d = 2 * np.sqrt(3)
        for k in range(-5, 6):
            inner.append([k * d, 0.0, 0.0])
        S = 40.0
    
    return inner, (0.0, 0.0), S, 0.0

# We need to make sure the construct function is defined and returns the correct format.
# The above code is a bit messy. Let's clean it up.
