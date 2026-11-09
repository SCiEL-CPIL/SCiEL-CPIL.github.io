import math
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers in the hexagonal lattice (side 3 cluster)
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert axial to cartesian (flat-top hex grid)
                # x = sqrt(3) * (q + r/2)
                # y = 1.5 * r
                x = sqrt3 * (q + r / 2.0)
                y = 1.5 * r
                centers.append((x, y))
                
    # Sort by distance from origin to pick the most central 12
    centers.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first 12
    selected_centers = centers[:12]
    
    # Compute the maximum distance from origin
    max_dist = max(math.sqrt(x*x + y*y) for x, y in selected_centers)
    
    # We need to ensure:
    # 1. Inner hexagons are disjoint: distance between centers >= 2 * apothem = sqrt(3)
    #    In the lattice, adjacent centers are exactly sqrt(3) apart.
    #    To be strictly disjoint (if evaluator uses >), we need a tiny epsilon.
    #    However, rotating the set might change distances.
    #    The set of 12 centers is a subset of the lattice, so pairwise distances are lattice distances.
    #    The minimum distance in the lattice is sqrt(3).
    #    So we need to scale up by a tiny factor to ensure strict inequality if required.
    #    Let's use scale = 1.0001 to be safe but minimal.
    
    # 2. Inner hexagons are inside the outer hexagon.
    #    The outer hexagon has side L.
    #    The distance from center to vertex is L.
    #    The distance from center to edge (apothem) is L * sqrt(3) / 2.
    #    An inner hexagon with center at distance d from origin and orientation angle theta.
    #    The condition for containment is complex, but a sufficient condition is:
    #    The inner hexagon's vertices must be inside the outer hexagon.
    #    Alternatively, a simpler bound: 
    #    If we align the inner hexagons such that their "flat" sides face the outer edges,
    #    and rotate the whole configuration, we can minimize L.
    
    # The current strategy rotates the centers by 30 degrees.
    # This aligns the "radial" directions of the outer 5 centers (which are at dist 3)
    # with the vertices of the outer hexagon (if outer hex is axis-aligned).
    # Wait, let's re-evaluate the geometry.
    
    # Outer hexagon: side L, center (0,0), angle 0 (vertex at (L, 0)).
    # Vertices at angles 0, 60, 120, 180, 240, 300.
    # Edges are between vertices.
    
    # Inner hexagon: side 1, center (x,y), angle alpha.
    # For the inner hexagon to fit, its vertices must be inside the outer hex.
    # The inner hexagon's apothem is sqrt(3)/2.
    # The inner hexagon's circumradius is 1.
    
    # Let's stick to the working solution but tighten the parameters.
    # The previous solution used scale=1.001 and side=4.01.
    # Max dist of centers is 3.0.
    # With scale=1.001, max dist is 3.003.
    # If we align the farthest centers with the vertices of the outer hex,
    # the required L is at least the distance from origin to the farthest vertex of the inner hex.
    # If an inner hex center is at distance D along a vertex direction, and it is oriented such that a vertex points outward,
    # then the outer vertex is at D + 1.
    # So L >= D + 1.
    # With D = 3.003, L >= 4.003.
    # The previous side=4.01 was safe.
    
    # Can we do better?
    # What if we orient the inner hexagons differently?
    # If the inner hexagon is rotated 30 degrees relative to the radial direction,
    # its extent in the radial direction is its apothem + 0.5 * sqrt(3) ? No.
    # The radial extent of a hexagon with circumradius R and angle phi relative to the radial vector is:
    # R * max(cos(phi), cos(60-phi), ...)?
    # Actually, the support function of a regular hexagon.
    # For a unit hexagon, the maximum distance from center to a point in the direction theta is:
    # h(theta) = cos(theta mod 60 deg) / cos(30 deg) ? No.
    # h(theta) = R * cos(delta) / cos(30) where delta is angle to nearest vertex.
    # If aligned with vertex (delta=0), h = R = 1.
    # If aligned with edge midpoint (delta=30), h = R * cos(30)/cos(30) = 1? No.
    # The apothem is R * cos(30) = sqrt(3)/2.
    # So if the radial direction points to an edge midpoint, the extent is apothem = 0.866.
    # If it points to a vertex, the extent is 1.0.
    
    # So, to minimize L, we should align the radial direction of each inner hexagon
    # to point towards the nearest edge midpoint of the outer hexagon, not the vertex.
    # This reduces the required L from D + 1 to D + sqrt(3)/2.
    
    # Let's check if this is feasible.
    # The outer hexagon edges are at angles 30, 90, 150, 210, 270, 330 (midpoints).
    # The centers of the 5 outer hexagons in our selection are at angles:
    # The 12 centers are:
    # Dist 0: (0,0)
    # Dist sqrt(3): 6 centers at angles 0, 60, 120, 180, 240, 300 (in cartesian, these are vertices of the inner hex grid ring 1)
    # Wait, the lattice points at dist sqrt(3) are (sqrt(3), 0), (sqrt(3)/2, 1.5), etc.
    # These are at angles 0, 60, 120...
    # The centers at dist 3 are:
    # (3, 0) -> angle 0
    # (1.5, 2.598) -> angle 60
    # etc.
    # So the 5 selected centers at dist 3 are at angles 0, 60, 120, 180, 240 (or similar).
    # These angles coincide with the OUTER HEXAGONS' VERTEX DIRECTIONS (0, 60, ...).
    # So if we keep the outer hexagon at angle 0, the radial directions of the outer inner-hexagons point to the outer vertices.
    # This forces L >= D + 1.
    
    # To use the apothem benefit, we need the radial directions to point to the outer EDGE MIDPOINTS.
    # The outer edge midpoints are at angles 30, 90, ...
    # So we need to rotate the entire configuration by 30 degrees.
    # If we rotate the inner centers by 30 degrees, the dist-3 centers will be at angles 30, 90, ...
    # Then, if we orient each inner hexagon such that its EDGE MIDPOINT points radially outward,
    # the radial extent is apothem = sqrt(3)/2.
    # So L >= D + sqrt(3)/2.
    # D = 3 * scale.
    # L >= 3 * scale + 0.866.
    # With scale = 1.0001, L >= 3.0003 + 0.866 = 3.8663.
    # This is significantly better than 4.003!
    
    # Let's verify this.
    # 1. Rotate centers by 30 degrees.
    # 2. Orient each inner hexagon such that its flat side faces the center (or vertex faces center?).
    #    We want the radial extent to be minimal.
    #    The radial extent is minimal when the radial direction aligns with an edge midpoint.
    #    So the inner hexagon should have an edge midpoint pointing radially outward.
    #    This means the inner hexagon's "flat" side is perpendicular to the radius.
    #    In terms of angle_degrees for the inner hexagon:
    #    A regular hexagon with angle 0 has a vertex at (1,0).
    #    An edge midpoint is at 30 degrees.
    #    So if we want an edge midpoint at angle theta (radial direction), we set the hex angle to theta - 30.
    
    # Let's implement this.
    
    angle_rad = math.pi / 6  # 30 degrees rotation
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    rotated_centers = []
    for x, y in selected_centers:
        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a
        rotated_centers.append((x_new, y_new))
        
    # Calculate max distance
    max_dist_rot = max(math.sqrt(x*x + y*y) for x, y in rotated_centers)
    
    # Scale to ensure disjointness
    # The minimum distance in the lattice is sqrt(3).
    # We need distance > sqrt(3).
    # scale = 1.0001 should be enough.
    scale = 1.0001
    
    inner = []
    for x, y in rotated_centers:
        x_scaled = x * scale
        y_scaled = y * scale
        # Calculate radial angle
        theta = math.atan2(y_scaled, x_scaled)
        # We want an edge midpoint to point in direction theta.
        # Edge midpoints are at angles 30, 90, ... relative to hex orientation.
        # So hex orientation = theta - 30 degrees.
        # Convert theta to degrees
        theta_deg = math.degrees(theta)
        hex_angle = theta_deg - 30.0
        inner.append([x_scaled, y_scaled, hex_angle])
        
    # Calculate required L
    # L >= max_dist_rot * scale + apothem
    # apothem = sqrt(3) / 2
    apothem = sqrt3 / 2.0
    L_min = max_dist_rot * scale + apothem
    
    # Add a tiny epsilon for safety
    L = L_min + 1e-6
    
    outer_center = (0.0, 0.0)
    side = L
    angle = 0.0
    
    return inner, outer_center, side, angle

# EVOLVE-BLOCK-END
