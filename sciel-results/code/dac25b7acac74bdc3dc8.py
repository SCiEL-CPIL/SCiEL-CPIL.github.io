import numpy as np
import math

def construct():
    # Define 11 hexagon centers in a compact "triangle + 1" configuration
    # Axial coordinates (q, r)
    axial_coords = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3),
        (1, -1) # Added to make 11
    ]
    
    centers = []
    for q, r in axial_coords:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append([x, y])
    
    centers = np.array(centers)
    
    # Find the minimal enclosing regular hexagon
    # We can optimize the rotation angle of the outer hexagon
    # The side length S required for a given rotation alpha is:
    # S = max over all vertices v of all inner hexagons of (distance from center to boundary in direction v)
    
    # Vertices of a unit hexagon aligned with x-axis (flat top/bottom? or pointy?)
    # Standard hexagon with flat top: vertices at angles 0, 60, 120...
    # Let's assume inner hexagons are aligned with angle 0.
    # Vertices of one hexagon at origin:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    
    # Generate all vertices for all 11 hexagons
    all_verts = []
    for c in centers:
        all_verts.extend(c + hex_verts)
    all_verts = np.array(all_verts)
    
    # Function to calculate required side length S for a given outer rotation alpha
    def get_required_S(alpha):
        # Outer hexagon has side S.
        # Distance from center to boundary in direction theta is:
        # d(theta) = (S * sqrt(3) / 2) / cos((theta - alpha) mod 60 deg - 30 deg)
        # Wait, the "radius" of a regular hexagon varies between R_circ = S and R_in = S*sqrt(3)/2.
        # The formula for the distance from center to edge at angle theta (relative to normal of a side) is R_in / cos(theta).
        # Let's define the outer hexagon orientation such that its sides are at angles alpha, alpha+60, ...
        # The normal to the side at angle alpha is alpha.
        # The distance to the boundary in direction theta is:
        # d(theta) = R_in / cos(delta), where delta is the angle between theta and the nearest normal.
        # R_in = S * sqrt(3) / 2.
        
        # We need d(theta) >= |v| for all vertices v.
        # So S >= |v| * cos(delta) / (sqrt(3)/2)
        
        max_S = 0
        for v in all_verts:
            theta_v = math.atan2(v[1], v[0])
            # Find the nearest normal angle
            # Normals are at alpha + k * 60 deg
            # delta = (theta_v - alpha) % (pi/3)
            # We want the smallest angle to the nearest normal.
            delta = (theta_v - alpha) % (math.pi / 3)
            # The angle to the nearest normal is min(delta, pi/3 - delta)
            delta_min = min(delta, math.pi / 3 - delta)
            
            # cos(delta_min) is the projection factor
            # S >= |v| * cos(delta_min) / (sqrt(3)/2)
            S_req = np.linalg.norm(v) * math.cos(delta_min) / (math.sqrt(3) / 2)
            if S_req > max_S:
                max_S = S_req
        return max_S

    # Optimize alpha
    # The function is periodic with period 60 deg.
    # We can sample alpha or use a simple optimization.
    # Since the shape is not perfectly symmetric, we scan alpha.
    
    best_S = float('inf')
    best_alpha = 0
    # Scan alpha from 0 to 60 degrees
    for i in range(100):
        alpha = (i / 100.0) * (math.pi / 3)
        S = get_required_S(alpha)
        if S < best_S:
            best_S = S
            best_alpha = alpha

    # Return the result
    # Inner: list of [x, y, angle_degrees]
    inner = [[c[0], c[1], 0.0] for c in centers]
    center = [0.0, 0.0]
    side = best_S
    angle = math.degrees(best_alpha)
    
    return inner, center, side, angle

