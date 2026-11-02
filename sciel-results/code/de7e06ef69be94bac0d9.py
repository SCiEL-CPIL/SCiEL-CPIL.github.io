import numpy as np
from scipy.optimize import minimize_scalar

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the layout: 3 rows of 4 hexagons in a staggered (honeycomb) arrangement
    # Side length of inner hexagons is 1.
    # Distance between centers in a row: sqrt(3)
    # Vertical distance between rows: 1.5
    # Horizontal offset for alternating rows: sqrt(3)/2
    
    centers = []
    sqrt3 = np.sqrt(3)
    
    # Generate 3 rows, 4 columns
    for r in range(3):
        for c in range(4):
            y = r * 1.5
            x = c * sqrt3
            if r % 2 == 1:
                x += sqrt3 / 2.0
            centers.append([x, y])
            
    centers = np.array(centers)
    
    # Center the configuration at the origin
    centroid = np.mean(centers, axis=0)
    centers -= centroid
    
    # Function to calculate the minimal outer hexagon side length S for a given rotation theta
    def calc_S(theta):
        # The outer hexagon is regular.
        # We check the 6 normal directions.
        # Due to symmetry, we only need to check 3 directions (0, 60, 120) and their opposites
        # but the max projection handles both.
        # The distance from center to side of outer hexagon is R_in = S * sqrt(3) / 2
        # So S = 2 * R_in / sqrt(3)
        
        # Normals of the outer hexagon sides
        # A regular hexagon has sides perpendicular to angles theta, theta+60, theta+120, etc.
        # Let's define the angles of the normals.
        angles = np.array([theta, theta + np.pi/3, theta + 2*np.pi/3])
        
        max_proj = 0
        for angle in angles:
            n = np.array([np.cos(angle), np.sin(angle)])
            # Project all centers onto the normal
            projs = np.dot(centers, n)
            # We need to cover the entire hexagons, not just centers.
            # The "radius" of a unit hexagon in any direction is not constant, but the max distance from center to boundary is 1 (to vertex).
            # However, the support function of a regular hexagon of side 1 in direction n is:
            # h(n) = max_{v in hex} v dot n
            # For a regular hexagon, h(n) depends on the angle of n relative to the hexagon's axes.
            # Since the inner hexagons are all aligned (angle 0), their shape is fixed.
            # The max projection of a unit hexagon (flat-top or pointy-top?)
            # Let's assume standard orientation: vertices at angles 0, 60, 120... (Pointy top? No, vertex at 0 is pointy right).
            # Let's use the fact that the inner hexagons are regular.
            # The max distance from center to any point in the inner hexagon in direction n is the support function.
            # For a unit regular hexagon, the support function h(n) is periodic with period 60 deg.
            # h(n) = 1 / cos(delta) where delta is the angle from the nearest vertex direction?
            # Actually, it's easier to just compute the max projection of the vertices of the inner hexagons.
            # But that's expensive.
            # Approximation: The max radius is 1 (vertex). The min radius is sqrt(3)/2 (side).
            # A safe upper bound for the projection of the inner hexagon in any direction is 1.
            # But we can be tighter.
            # Let's calculate the exact max projection of a unit hexagon in direction n.
            # Vertices of unit hexagon (side 1) at angles 0, 60, 120, 180, 240, 300.
            # Wait, if side is 1, distance to vertex is 1.
            # Vertices: (1,0), (0.5, sqrt(3)/2), ...
            # Let's just use the vertices of the inner hexagons to be precise.
            
            # To optimize speed, we can pre-compute the vertices of one unit hexagon
            # and rotate them? No, inner hexagons are fixed at angle 0.
            # So we just need the max projection of the shape "unit hexagon" in direction n.
            # This is a constant for a given n.
            
            # Let's compute h_inner(n)
            # Vertices of unit hexagon:
            v_angles = np.arange(0, 2*np.pi, np.pi/3)
            v_coords = np.array([np.cos(v_angles), np.sin(v_angles)]).T
            # Project vertices onto n
            v_projs = np.dot(v_coords, n)
            h_inner = np.max(v_projs)
            
            # The max projection of the set of hexagons in direction n is:
            # max_p (p dot n + h_inner(n))
            # Wait, h_inner(n) is the max projection of the hexagon *centered at origin*.
            # So for a hexagon at center c, max proj is c dot n + h_inner(n).
            
            max_p_proj = np.max(projs) + h_inner
            if max_p_proj > max_proj:
                max_proj = max_p_proj
                
        # S * sqrt(3) / 2 >= max_proj
        S = max_proj * 2.0 / sqrt3
        return S

    # Minimize S over theta in [0, pi/3]
    # The function is periodic and symmetric.
    res = minimize_scalar(calc_S, bounds=(0, np.pi/3), method='bounded')
    theta_opt = res.x
    S_opt = res.fun
    
    # The outer hexagon center is (0,0) because we centered the inner cluster.
    # The outer hexagon side length is S_opt.
    # The outer hexagon rotation is theta_opt.
    
    # Construct the return values
    # inner: list of [x, y, angle_degrees]
    inner_list = []
    for c in centers:
        inner_list.append([float(c[0]), float(c[1]), 0.0])
        
    center = (0.0, 0.0)
    side = float(S_opt)
    angle_deg = float(np.degrees(theta_opt))
    
    return inner_list, center, side, angle_deg
# EVOLVE-BLOCK-END
