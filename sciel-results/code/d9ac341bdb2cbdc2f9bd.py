"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Pointy-top unit hexagons
    # Hex grid coordinates for 11 hexagons in a compact arrangement
    # Using axial coordinates: (q, r) -> pixel
    # x = 3/2 * q
    # y = sqrt(3) * (r + q/2)
    
    sqrt3 = np.sqrt(3)
    
    # 11 hexagons: try a 4-4-3 arrangement or similar
    # Let's use axial coords for a hexagonal cluster
    # Center at (0,0), then ring of 6, then 4 more
    axial_coords = [
        (0, 0),      # center
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),  # ring of 6
        (2, 0), (2, -1), (1, -2), (-2, 1),  # 4 more to make 11
    ]
    
    # Convert axial to pixel (pointy-top)
    inner = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        inner.append([x, y, 0.0])
    
    inner = np.array(inner)
    
    # Find minimum enclosing regular hexagon
    # A regular hexagon with center (cx, cy), side length S, rotation angle theta
    # contains point (px, py) if for all 6 sides, the point is on the correct side
    
    def hex_contains(points, cx, cy, S, theta):
        # The 6 vertices of the outer hexagon
        # For a regular hexagon with side S, circumradius = S
        # Vertices at angles theta + k*60 degrees, k=0..5
        # A point is inside if its distance from center projected onto each normal
        # is <= S * cos(30 deg) = S * sqrt(3)/2 (the apothem)
        
        # Normals to the 6 sides point outward at angles theta + k*60 + 30 degrees
        # Actually, for a regular hexagon, the 6 face normals are at angles
        # theta + k*60 degrees for k=0..5 (if we define theta as the angle of one vertex)
        
        # Let me define: outer hexagon has vertices at angles theta + k*60, k=0..5
        # The inward normals to the sides are at angles theta + k*60 + 30, k=0..5
        # A point (px,py) is inside if for each k:
        #   (px-cx)*cos(nk) + (py-cy)*sin(nk) <= S * sqrt(3)/2
        # where nk = theta + k*60 + 30 (in radians)
        
        apothem = S * sqrt3 / 2
        for k in range(6):
            nk = theta + np.radians(60 * k + 30)
            proj = (points[:, 0] - cx) * np.cos(nk) + (points[:, 1] - cy) * np.sin(nk)
            if np.max(proj) > apothem + 1e-9:
                return False
        return True
    
    def min_side_for_rotation(theta, cx, cy, points):
        # Binary search on S
        lo, hi = 0.0, 100.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if hex_contains(points, cx, cy, mid, theta):
                hi = mid
            else:
                lo = mid
        return hi
    
    # Optimize center and rotation
    # First, try center at centroid of inner points
    cx0 = np.mean(inner[:, 0])
    cy0 = np.mean(inner[:, 1])
    
    best_S = float('inf')
    best_theta = 0.0
    best_cx = cx0
    best_cy = cy0
    
    # Try various rotations
    for theta_deg in range(0, 180, 5):
        theta = np.radians(theta_deg)
        S = min_side_for_rotation(theta, cx0, cy0, inner)
        if S < best_S:
            best_S = S
            best_theta = theta
            best_cx = cx0
            best_cy = cy0
    
    # Refine around best
    from scipy.optimize import minimize
    
    def objective(params):
        cx, cy, theta = params
        return min_side_for_rotation(theta, cx, cy, inner)
    
    res = minimize(objective, [best_cx, best_cy, best_theta], method='Nelder-Mead',
                   options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if res.fun < best_S:
        best_S = res.fun
        best_cx, best_cy, best_theta = res.x
    
    inner_list = [[float(x), float(y), 0.0] for x, y in inner]
    return inner_list, (float(best_cx), float(best_cy)), float(best_S), float(np.degrees(best_theta))
# EVOLVE-BLOCK-END
