import numpy as np
from scipy.optimize import minimize_scalar

N = 11

def construct():
    # 1. Generate 11 hexagon centers from a hexagonal lattice
    # Axial coordinates (q, r)
    # Distance metric: d = 0.5 * (|q| + |r| + |q+r|)
    
    centers = []
    # Add center
    centers.append((0, 0))
    
    # Add rings
    # Ring 1: 6 hexes
    # Ring 2: 12 hexes (we need 4 more)
    
    # Generate all lattice points up to radius 2
    candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Check if in hexagon of radius 2
            if abs(q + r) <= 2:
                d = 0.5 * (abs(q) + abs(r) + abs(q + r))
                # Convert to Cartesian
                # x = sqrt(3) * (q + r/2)
                # y = 1.5 * r
                x = np.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                candidates.append((d, x, y))
    
    # Sort by distance
    candidates.sort(key=lambda t: t[0])
    
    # Take first 11
    # Note: The 7th and 8th points might have same distance, order might vary.
    # To ensure a symmetric-ish start, we can just take the first 11.
    selected_centers = np.array([(x, y) for d, x, y in candidates[:N]])
    
    # 2. Function to calculate required side length S for a given rotation theta
    # The container is a regular hexagon centered at origin.
    # Its sides are at distance A (apothem) from center.
    # S = 2 * A / sqrt(3)
    # A must be >= max projection of any inner hexagon onto the normal vectors of the container sides.
    
    # Normals of container sides (for theta=0):
    # Angles: 0, 60, 120, 180, 240, 300
    base_angles = np.deg2rad(np.arange(0, 360, 60))
    
    def calc_S(theta_rad):
        # Rotate centers by theta_rad
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        rot_mat = np.array([[c, -s], [s, c]])
        rotated_centers = selected_centers @ rot_mat.T
        
        max_apothem = 0
        for angle in base_angles:
            nx = np.cos(angle)
            ny = np.sin(angle)
            
            # Project centers onto this normal
            projections = rotated_centers @ np.array([nx, ny])
            
            # For each hexagon, add its support in this direction
            # Support of a unit hexagon (angle 0) in direction phi relative to its orientation:
            # Since we rotate the whole cluster, the hexagons are also rotated by theta_rad.
            # The orientation of each hexagon is theta_rad (relative to container).
            # The support function of a unit hexagon (side 1) with orientation alpha in direction beta is:
            # h(beta, alpha) = (sqrt(3)/2) * max( cos(beta - alpha - k*pi/3) for k in 0..5 )
            # Actually, simpler: The hexagon is regular.
            # If we align the hexagon with the container (theta=0), h is simple.
            # Here, the hexagons are rotated by theta_rad.
            # The direction of the normal is 'angle'.
            # The relative angle is angle - theta_rad.
            
            rel_angle = angle - theta_rad
            
            # Support of unit hexagon in direction rel_angle
            # The hexagon has vertices at angles 0, 60, 120...
            # The support is (sqrt(3)/2) * max( cos(rel_angle - k*pi/3) )
            # Wait, the apothem is sqrt(3)/2.
            # h(theta) = apothem * max( cos(theta - vertex_angle) ) ? No.
            # For a regular polygon, h(theta) = R * cos(theta - phi) where phi is the angle to the closest vertex/face normal?
            # For a hexagon, the support function is periodic with 60 deg.
            # h(rel_angle) = (sqrt(3)/2) * max( cos(rel_angle - k*pi/3) for k in range(6) )
            # Since cos is max 1, and we want the max projection.
            
            # Optimization: h(rel_angle) = (sqrt(3)/2) * cos( min( abs( (rel_angle - k*pi/3) mod (2*pi/3) ) - pi/3 ) )?
            # Simpler:
            # The maximum projection of a unit hexagon (apothem a=sqrt(3)/2) in any direction is 'a' if the direction is normal to a face, and 1 if the direction is to a vertex.
            # h(theta) = a * cos( delta ) where delta is the angle to the nearest face normal?
            # Face normals are at 0, 60, 120...
            # So h(rel_angle) = (sqrt(3)/2) * cos( min( abs( (rel_angle - k*pi/3) mod (pi/3) ) - pi/6 )? No.
            
            # Let's just compute it.
            # Angles of face normals: 0, pi/3, 2pi/3...
            # h(rel) = max( (sqrt(3)/2) * cos(rel - k*pi/3) )
            
            # Since we loop over 6 container normals, and for each we need max over 6 hex orientations?
            # No, the hex orientation is fixed (theta_rad).
            # So for a specific container normal 'angle', the relative angle is fixed.
            # We just need h(rel_angle).
            
            # h(rel) = (sqrt(3)/2) * max( cos(rel - k*pi/3) for k in range(6) )
            
            # We can precompute or just use the property that h(rel) = (sqrt(3)/2) * cos( d ) where d is the angular distance to the nearest multiple of pi/3?
            # No, the support is max over vertices.
            # Vertices are at 0, pi/3...
            # Projection of vertex at angle phi is cos(rel - phi).
            # Max cos is 1 when rel=phi.
            # So h(rel) = (sqrt(3)/2) * max( cos(rel - k*pi/3) )
            
            # Let's calculate this max cos.
            # rel_mod = rel_angle % (pi/3)
            # The max cos is cos( min( rel_mod, pi/3 - rel_mod ) )? No.
            # The function cos(x) is max at 0.
            # So we want rel_angle to be as close to k*pi/3 as possible.
            # Let delta = min( abs( (rel_angle - k*pi/3) mod (2*pi) ) for k in 0..5 )
            # Actually, due to symmetry, we can just look at the angle modulo pi/3.
            # Let a = rel_angle % (pi/3)
            # The closest multiple of pi/3 is either 0 or pi/3.
            # Distance to 0 is a. Distance to pi/3 is pi/3 - a.
            # Min distance is min(a, pi/3 - a).
            # Max cos is cos(min_distance).
            
            a = rel_angle % (np.pi / 3.0)
            min_dist = min(a, np.pi / 3.0 - a)
            h_val = (np.sqrt(3) / 2.0) * np.cos(min_dist)
            
            # The max projection for this normal is max(projections) + h_val
            current_max = np.max(projections) + h_val
            if current_max > max_apothem:
                max_apothem = current_max
        
        # Convert apothem to side length
        S = 2.0 * max_apothem / np.sqrt(3.0)
        return S

    # 3. Optimize theta
    # We only need to search 0 to 60 degrees due to symmetry
    result = minimize_scalar(calc_S, bounds=(0, np.pi/3), method='bounded')
    best_theta = result.x
    best_S = result.fun
    
    # 4. Construct the final inner hexagon positions
    c = np.cos(best_theta)
    s = np.sin(best_theta)
    rot_mat = np.array([[c, -s], [s, c]])
    final_centers = selected_centers @ rot_mat.T
    
    # Format output: [[x, y, angle_degrees], ...]
    # Angle is in degrees.
    angle_deg = np.rad2deg(best_theta)
    
    inner = []
    for x, y in final_centers:
        inner.append([float(x), float(y), float(angle_deg)])
        
    return inner, (0.0, 0.0), float(best_S), float(angle_deg)

# EVOLVE-BLOCK-END
