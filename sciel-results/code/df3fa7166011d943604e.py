import numpy as np
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate a hexagonal lattice of points
    # We need enough points to pick N closest
    # For N=11, a radius of 3 or 4 in grid units is sufficient
    max_coord = 10
    points = []
    for q in range(-max_coord, max_coord + 1):
        for r in range(-max_coord, max_coord + 1):
            # Convert axial coordinates to Cartesian
            # Flat-top hexagons
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            points.append((x, y))
    
    # Sort by distance from origin
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the N closest points
    selected_points = points[:N]
    
    # Calculate the minimum enclosing hexagon side length for the centers
    # The hexagon is aligned with the lattice
    # Normals are at 30, 90, 150 degrees (for flat-top)
    # Let's define the 3 unique normal directions
    # Angle 0 is x-axis.
    # Flat-top hexagon sides are vertical? No.
    # Standard flat-top: vertices at 0, 60, 120...
    # Normals at 30, 90, 150...
    
    angles = [30, 90, 150]
    max_proj = 0.0
    for angle_deg in angles:
        angle_rad = math.radians(angle_deg)
        nx = math.cos(angle_rad)
        ny = math.sin(angle_rad)
        for x, y in selected_points:
            proj = abs(x * nx + y * ny)
            if proj > max_proj:
                max_proj = proj
    
    # max_proj is the apothem of the center-hexagon
    L_centers = (2.0 / math.sqrt(3.0)) * max_proj
    L_outer = L_centers + 1.0
    
    # Construct inner list
    # (x, y, angle_degrees)
    inner = []
    for x, y in selected_points:
        inner.append([x, y, 0.0])
        
    center = [0.0, 0.0]
    side = L_outer
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
