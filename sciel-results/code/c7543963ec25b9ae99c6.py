import math
import numpy as np
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0

# Define a set of 11 hexagons in axial coordinates.
# This is a compact arrangement. 
# Center (0,0), neighbors (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 7 hexes.
# Add 4 more: (2,-1), (-2,1), (1,2), (-1,-2) ? 
# Let's use a known good shape: a 3x3 block minus corners? 
# Or just a linear chain? No, compact is better.
# Let's use the "flower" + 4 more.
# Axial coords for a 3x3 hex grid (side 2 hexagon) has 19 hexes.
# We need 11.
# Let's take a 3x3 grid and remove 8.
# Center (0,0).
# Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1) -> 6 hexes. Total 7.
# Ring 2 candidates: (2,-1), (2,-2), (1,-2), (-1,2), (-2,2), (-2,1).
# Let's pick (2,-1), (-2,1), (1,2) -> wait (1,2) is not in ring 2? (1,2) is dist 2.
# Let's just define points in Cartesian for simplicity and optimize.

def axial_to_cartesian(q, r):
    x = math.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

# Hardcoded 11 hexes (axial)
# A compact shape
cells = [
    (0, 0),
    (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
    (2, -1), (-2, 1), (1, 1), (-1, -1) # Adding 4 more to make 11
]

# Convert to Cartesian
pts = np.array([axial_to_cartesian(q, r) for q, r in cells])

def calc_min_side(pts, beta):
    """
    Calculate minimal side length S for a regular hexagon of rotation beta
    containing pts. We optimize the center (cx, cy).
    """
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    # The optimal center is the center of the minimum enclosing hexagon.
    # We can approximate this by checking the center of the bounding box in each of the 6 directions?
    # Actually, for a regular hexagon, the optimal center is the point that minimizes the max distance to the 6 sides.
    # This is a convex problem. We can solve it using the fact that the optimal center lies at the intersection of the "midlines" of the critical pairs.
    # A simple and fast method: Use the centroid as a starting point and do a few iterations of gradient descent or just a small grid search around the centroid.
    
    cx0, cy0 = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    
    best_S = float('inf')
    best_cx, best_cy = cx0, cy0
    
    # Grid search for center in a small area around centroid
    # The hexagon size is roughly 2-3 units. Center should be within 1 unit of centroid.
    for dx in np.linspace(-1.5, 1.5, 15):
        for dy in np.linspace(-1.5, 1.5, 15):
            cx = cx0 + dx
            cy = cy0 + dy
            
            max_proj = 0
            for nx, ny in normals:
                projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
                max_proj = max(max_proj, np.max(projs), -np.min(projs))
            
            S = (max_proj + APOTHEM) / APOTHEM
            if S < best_S:
                best_S = S
                best_cx = cx
                best_cy = cy
                
    return best_S, best_cx, best_cy

def construct():
    if time.time() - START_TIME > TIME_LIMIT - 5.0:
        # Fallback: return a valid but large hexagon
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        S = 10.0
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [float(cx), float(cy)], S, 0.0

    best_S_global = float('inf')
    best_beta_global = 0.0
    best_cx_global = 0.0
    best_cy_global = 0.0
    
    # Scan rotation beta from 0 to 30 degrees (pi/6)
    # Step size 1 degree (pi/180) is ~30 steps.
    # For each step, we do a 15x15 grid search for center. 30 * 225 = 6750 evaluations.
    # Each evaluation is fast (vectorized numpy).
    
    for beta in np.linspace(0, math.pi / 6, 31, endpoint=False):
        if time.time() - START_TIME > TIME_LIMIT - 10.0:
            break
            
        S, cx, cy = calc_min_side(pts, beta)
        if S < best_S_global:
            best_S_global = S
            best_beta_global = beta
            best_cx_global = cx
            best_cy_global = cy
            
    # Refine with smaller steps around best beta
    if best_beta_global > 0 and best_beta_global < math.pi / 6:
        for beta in np.linspace(best_beta_global - math.pi/180, best_beta_global + math.pi/180, 21):
            if time.time() - START_TIME > TIME_LIMIT - 5.0:
                break
            if beta < 0 or beta >= math.pi / 6:
                continue
            S, cx, cy = calc_min_side(pts, beta)
            if S < best_S_global:
                best_S_global = S
                best_beta_global = beta
                best_cx_global = cx
                best_cy_global = cy

    # Prepare output
    angle_deg = math.degrees(best_beta_global)
    inner = [[float(x), float(y), float(angle_deg)] for x, y in pts]
    center = [float(best_cx_global), float(best_cy_global)]
    side = float(best_S_global)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation

if __name__ == "__main__":
    start = time.time()
    res = construct()
    print(f"Time taken: {time.time() - start:.2f}s")
    print(f"Side length: {res[2]:.6f}")
