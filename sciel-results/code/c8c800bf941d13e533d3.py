import math
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate centers for N=11 in a hexagonal lattice
    points = []
    for q in range(-5, 6):
        for r in range(-5, 6):
            norm_sq = q*q + q*r + r*r
            points.append((norm_sq, q, r))
    
    points.sort()
    selected = points[:N]
    
    centers = []
    for _, q, r in selected:
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        centers.append([x, y])
        
    # Apply a safety factor to ensure strict separation
    # The lattice spacing is sqrt(3). Touching distance is sqrt(3).
    # Scale by 1.01 to ensure they don't overlap due to floating point or strict checks.
    safety_factor = 1.01
    for i in range(len(centers)):
        centers[i][0] *= safety_factor
        centers[i][1] *= safety_factor

    # Function to calculate the required side length of an enclosing hexagon
    # given a set of points and an outer hexagon angle
    def calc_outer_side(centers, angle_deg):
        theta = math.radians(angle_deg)
        normals = [theta + math.radians(60*i) for i in range(6)]
        
        max_required_A = 0.0
        for cx, cy in centers:
            # We need to check all vertices of the inner hexagons?
            # No, the inner hexagons are fixed orientation (0 deg).
            # We can just check the "radius" of the union in the direction of each normal.
            # The maximum extent in direction n is max( dot(p, n) ) for all vertices p.
            # Since inner hexagons are regular and same orientation, we can precompute their vertices.
            pass
            
        # Actually, let's just compute vertices for all inner hexagons once.
        # This is static for the positions, only outer angle changes.
        return 0.0 # Placeholder

    # Precompute vertices for inner hexagons
    # Inner hexagons are at 0 degrees (pointy top)
    inner_angles = [math.radians(30 + 60*i) for i in range(6)]
    all_vertices = []
    for cx, cy in centers:
        for ang in inner_angles:
            vx = cx + math.cos(ang)
            vy = cy + math.sin(ang)
            all_vertices.append((vx, vy))
            
    def get_min_side(angle_deg):
        theta = math.radians(angle_deg)
        normals = [theta + math.radians(60*i) for i in range(6)]
        
        max_A = 0.0
        for vx, vy in all_vertices:
            # Project onto all normals
            # The constraint is that the hexagon contains the point.
            # For a hexagon centered at origin with apothem A and angle theta,
            # a point (x,y) is inside if for all normal directions n_k:
            # dot((x,y), n_k) <= A
            # So A >= max_k dot((x,y), n_k)
            # We need A >= max over all points and all normals.
            for n in normals:
                nx = math.cos(n)
                ny = math.sin(n)
                proj = vx*nx + vy*ny
                if proj > max_A:
                    max_A = proj
        S = 2.0 * max_A / math.sqrt(3)
        return S

    # Optimize the outer hexagon angle to minimize side length
    # Due to symmetry, 0-60 is enough, but 0-30 covers unique cases for symmetric clusters.
    # Let's use a fine scan or Nelder-Mead.
    # Scan is fast and robust.
    best_S = float('inf')
    best_angle = 0.0
    
    # Scan 0 to 30 degrees with step 0.1
    for deg in np.arange(0, 30.1, 0.1):
        S = get_min_side(deg)
        if S < best_S:
            best_S = S
            best_angle = deg
            
    # Refine with Nelder-Mead if needed? Scan is likely sufficient for 0.1 deg resolution.
    # Let's do a finer scan around the best found.
    if best_S < float('inf'):
        for deg in np.arange(best_angle - 1, best_angle + 1, 0.01):
            S = get_min_side(deg)
            if S < best_S:
                best_S = S
                best_angle = deg

    inner = [[x, y, 0.0] for x, y in centers]
    center = (0.0, 0.0)
    side = best_S
    angle = float(best_angle)
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
