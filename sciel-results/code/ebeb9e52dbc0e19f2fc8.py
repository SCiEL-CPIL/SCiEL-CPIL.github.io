import numpy as np
import math

N = 11

def construct():
    # Define centers for 11 hexagons in a compact cluster
    # Using axial coordinates (q, r) for hex grid
    # Convert to Cartesian: x = sqrt(3) * (q + r/2), y = 3/2 * r
    # This is for pointy-top hexagons? 
    # Let's use flat-top hexagons to align with a flat-top outer hexagon.
    # Flat-top hexagon:
    # Width = 2, Height = sqrt(3)
    # Center to center horizontal distance = 1.5
    # Center to center vertical distance = sqrt(3)
    # Offset for odd columns = sqrt(3)/2
    
    # Let's use a simpler coordinate system.
    # Let the outer hexagon be flat-top, centered at (0,0).
    # Normals are at angles 0, 60, 120, 180, 240, 300 degrees.
    # n_j = (cos(j*pi/3), sin(j*pi/3))
    
    # Place 11 hexagons in a cluster.
    # A 3x3 grid is 9. Add 2 more.
    # Or a hexagonal cluster of 7 (side 2) + 4.
    # Let's use a shape that is roughly symmetric.
    # Coordinates (col, row) in a skewed grid.
    # Let's just hardcode a good set of centers.
    
    # Centers in Cartesian coordinates (x, y)
    # Using a lattice where neighbors are at distance sqrt(3)
    # Let's place them in a "flower" pattern (7) + 4 more on the boundary.
    # Center: (0,0)
    # 6 neighbors:
    # (sqrt(3), 0), (-sqrt(3), 0), (0, sqrt(3)), (0, -sqrt(3)), 
    # Wait, for flat-top hexagons, neighbors are at:
    # (1.5, sqrt(3)/2), (1.5, -sqrt(3)/2), (0, sqrt(3)), (-1.5, sqrt(3)/2), (-1.5, -sqrt(3)/2), (0, -sqrt(3))
    
    # Let's use the standard hex grid coordinates.
    # x = sqrt(3) * (col + row/2)
    # y = 1.5 * row
    # This is for pointy-top?
    # Let's stick to Cartesian.
    
    # A compact cluster of 11:
    # Row 0: 3 hexagons at x = -1.5, 0, 1.5 (y=0)
    # Row 1: 3 hexagons at x = -0.75, 0.75, 2.25? No, shifted.
    # Let's use the axial to Cartesian conversion for pointy-top hexagons.
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # This gives vertical columns.
    # Let's try to form a hexagon of side 2 (7 hexagons) and add 4.
    # Axial coords for side 2 hexagon (centered at 0,0):
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # Add 4 more: (2,-1), (2,0), (1,1), (0,2) ? 
    # Let's check if this fits in a smaller outer hexagon.
    
    centers_axial = [
        (0,0),
        (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
        (2,-1), (2,0), (1,1), (0,2)
    ]
    
    # Convert to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    centers_cart = []
    for q, r in centers_axial:
        x = math.sqrt(3) * (q + r/2)
        y = 1.5 * r
        centers_cart.append((x, y))
    
    # Calculate the minimal enclosing hexagon side length
    # Outer hexagon is aligned with the lattice?
    # The lattice normals are at 0, 60, 120... if we rotate the lattice?
    # The above Cartesian conversion is for pointy-top hexagons.
    # The "up" direction is y.
    # The normals of a pointy-top hexagon are at 0, 60, 120... ?
    # No, pointy-top has vertices at 90, 30, ...
    # Normals are at 0, 60, 120...
    # Let's assume the outer hexagon is also pointy-top and aligned.
    
    # Normals for pointy-top outer hexagon:
    # Angles: 0, 60, 120, 180, 240, 300 degrees.
    # n_j = (cos(j*pi/3), sin(j*pi/3))
    
    max_proj = 0.0
    for j in range(6):
        angle = j * math.pi / 3
        nx = math.cos(angle)
        ny = math.sin(angle)
        for cx, cy in centers_cart:
            proj = cx * nx + cy * ny
            if proj > max_proj:
                max_proj = proj
    
    # The outer hexagon must contain the inner hexagons.
    # The inner hexagons are unit size.
    # The support function of a unit pointy-top hexagon in direction n is:
    # h(n) = (sqrt(3)/2) / cos(angle_diff)
    # where angle_diff is the angle between n and the nearest normal of the inner hexagon.
    # Since we align inner and outer, the nearest normal is at 0 diff for the 6 main directions?
    # No, the normals of the inner hexagon are at the same angles as the outer.
    # So for each outer normal n_j, the inner hexagon has a normal n_j.
    # So the support function is the inradius: sqrt(3)/2.
    
    apothem_outer = max_proj + math.sqrt(3) / 2
    side_length = 2 * apothem_outer / math.sqrt(3)
    
    # Return inner hexagons (x, y, angle)
    # Angle 0 means aligned with outer.
    inner = [[x, y, 0.0] for x, y in centers_cart]
    
    return inner, (0.0, 0.0), side_length, 0.0

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to wrap it or ensure it's the entry point.
# The prompt says "Keep the function names and signatures of the seed program."
# Seed: def construct(): ...
# So I will just define construct.
# EVOLVE-BLOCK-END
