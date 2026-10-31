"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # 1. Generate positions for 11 unit hexagons in a dense hexagonal lattice
    #    Arrangement: Center (0,0) + 6 neighbors + 4 more to form a compact cluster.
    #    Actually, a standard "hex number" shape is 1, 7, 19...
    #    For 11, we take the central 7 (hex number 7) and add 4 more to complete a rhombus or similar compact shape.
    #    Let's use a simple grid generation and pick the 11 closest to origin.
    
    # Unit hexagon side length s=1.
    # Width (flat-to-flat) = sqrt(3). Height (point-to-point) = 2.
    # Lattice vectors for hexagonal packing (aligned with x-axis):
    # v1 = (sqrt(3), 0)  -> horizontal spacing
    # v2 = (sqrt(3)/2, 1.5) -> 60 degree neighbor
    
    s3 = math.sqrt(3)
    v1 = (s3, 0.0)
    v2 = (s3 / 2.0, 1.5)
    
    # Generate a grid of potential centers
    candidates = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * v1[0] + j * v2[0]
            y = i * v1[1] + j * v2[1]
            candidates.append((x, y))
    
    # Sort by distance to origin to pick the most compact cluster of 11
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first 11
    selected = candidates[:N]
    
    # Ensure we have exactly N points
    if len(selected) < N:
        # Fallback if grid too small (unlikely for 11)
        selected = [[0.0, 0.0]] + [[i*s3, 0.0] for i in range(1, N)]
        
    # Format inner hexagons: (x, y, angle)
    # Angle 0 means flat-top orientation?
    # Standard regular hexagon with side 1:
    # If vertices are at (1,0), (0.5, sqrt(3)/2), ... it's pointy-top.
    # If vertices are at (sqrt(3)/2, 0.5), ... it's flat-top.
    # The problem statement says "regular hexagons of side length 1".
    # Usually, "angle" in these interfaces rotates the shape.
    # Let's assume the base shape is standard and we can rotate.
    # For a hexagonal lattice packing, all hexagons should have the same orientation to tessellate perfectly.
    # Let's use angle 0.
    
    inner = []
    for x, y in selected:
        inner.append([x, y, 0.0])
    
    # 2. Calculate the minimum side length of the outer regular hexagon
    #    that contains all these unit hexagons.
    #    
    #    The outer hexagon is also regular. We can assume it has the same orientation (angle 0)
    #    because the packing is symmetric and aligned with the lattice.
    #    
    #    The condition for a point p to be inside a regular hexagon of side S centered at origin
    #    (flat-top orientation, vertices at angles 0, 60, ...) is:
    #    The hexagon is the intersection of 6 half-planes.
    #    For flat-top (vertices at (S,0), (S/2, S*sqrt(3)/2), ...):
    #    The "width" in the direction of the normal to the flat side is S*sqrt(3).
    #    The normals to the 6 sides are at angles 0, 60, 120, 180, 240, 300.
    #    Wait, if vertices are at 0, 60..., the sides are perpendicular to 30, 90, 150...
    #    
    #    Let's define the outer hexagon by its circumradius R_outer = S_outer.
    #    The inradius (apothem) is R_outer * cos(pi/6) = S_outer * sqrt(3)/2.
    #    
    #    A point (x,y) is inside the hexagon if:
    #    For each of the 6 directions k = 0..5:
    #    Let theta_k = k * 60 degrees.
    #    The distance from the center to the side in direction theta_k + 30?
    #    Actually, simpler:
    #    The hexagon is defined by |x| <= S, |y| <= S*sqrt(3)/2, |x/2 + y*sqrt(3)/2| <= S ...
    #    This is for pointy-top?
    #    
    #    Let's use the support function.
    #    For a regular hexagon of side S, the maximum projection of any point (x,y) onto the
    #    6 normal vectors of the sides must be <= the apothem A = S * sqrt(3)/2.
    #    
    #    The normals to the sides of a regular hexagon (pointy-top, vertex at 90 deg) are at 0, 60, 120, 180, 240, 300.
    #    If we rotate the inner hexagons by 0, they are likely pointy-top or flat-top.
    #    Let's assume the standard "flat-top" hexagon where vertices are at 0, 60, 120...
    #    Then the sides are perpendicular to 30, 90, 150...
    #    So the normals are at angles 30, 90, 150, 210, 270, 330.
    #    
    #    Let's check the vertices of a unit hexagon (side 1, flat-top, center 0,0):
    #    v0 = (1, 0) ? No, if side is 1, distance from center to vertex is 1.
    #    Vertices: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2).
    #    This is a pointy-top hexagon (vertex at 0 deg).
    #    Sides connect (1,0) to (0.5, sqrt(3)/2). Midpoint (0.75, sqrt(3)/4).
    #    Normal vector to this side: (sqrt(3)/2, 0.5) -> angle 30 deg.
    #    So yes, normals are at 30, 90, 150, 210, 270, 330.
    #    
    #    The apothem of the outer hexagon (side S_out) is A_out = S_out * sqrt(3)/2.
    #    For each inner hexagon center (cx, cy), we must ensure that the entire hexagon fits.
    #    Since the inner hexagons have the same orientation, the condition is that the
    #    maximum projection of the *center* plus the *apothem of the inner hexagon* in each normal direction
    #    is <= A_out.
    #    
    #    Inner apothem A_in = 1 * sqrt(3)/2.
    #    
    #    So, for each normal direction u_k (unit vector):
    #    max over all centers c of (c . u_k) + A_in <= A_out
    #    => A_out >= max( max(c . u_k) + A_in )
    #    
    #    S_out = A_out / (sqrt(3)/2) = 2 * A_out / sqrt(3)
    
    A_in = math.sqrt(3) / 2.0
    
    # Normals for flat-top hexagon (vertices at 0, 60, ...):
    # Side normals are at 30, 90, 150, 210, 270, 330 degrees.
    angles = [math.radians(30 + 60 * k) for k in range(6)]
    normals = [(math.cos(a), math.sin(a)) for a in angles]
    
    max_projections = [0.0] * 6
    
    for i in range(N):
        cx, cy, _ = inner[i]
        for k in range(6):
            proj = cx * normals[k][0] + cy * normals[k][1]
            if proj > max_projections[k]:
                max_projections[k] = proj
                
    # A_out must be at least max_projections[k] + A_in for all k
    required_A_out = max(max_projections) + A_in
    
    # Side length S_out
    S_out = 2.0 * required_A_out / math.sqrt(3)
    
    # Center is (0,0)
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, S_out, angle
# EVOLVE-BLOCK-END
