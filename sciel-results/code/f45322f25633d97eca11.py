"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def construct():
    """
    Construct 12 unit hexagons inside a minimal outer hexagon.
    
    Strategy:
    1. Use a hexagonal lattice for the inner hexagons. This is optimal for packing hexagons.
    2. Select the 12 lattice points closest to the origin to form a compact cluster.
    3. The optimal outer container for a hexagonal cluster aligned with the lattice is a regular hexagon
       with the same orientation (angle 0) and center at the origin (or centroid).
    4. Calculate the minimum side length of this outer hexagon required to contain all inner hexagons.
    """
    sqrt3 = np.sqrt(3)
    
    # Generate hexagonal lattice points
    # Axial coordinates (q, r) to Cartesian (x, y) for pointy-top or flat-top?
    # Standard hexagon side 1, flat-top:
    # Vertices at angles 0, 60, 120, 180, 240, 300.
    # Distance to neighbor in x-direction is 1.5? No.
    # Let's use the standard hex grid spacing.
    # For flat-top hexagons of side s=1:
    # Width = 2*s = 2
    # Height = sqrt(3)*s = sqrt(3)
    # Horizontal distance between columns = 1.5 * s = 1.5
    # Vertical distance between rows = sqrt(3) * s = sqrt(3)
    # Offset for odd columns = sqrt(3)/2
    
    # We want to pack 12 hexagons.
    # A "hexagonal number" cluster of radius 2 has 1+6+12 = 19 hexagons.
    # We only need 12.
    # The most compact arrangement of 12 hexagons on a lattice is a subset of the radius 2 hexagon.
    # Specifically, the center (1), the first ring (6), and 5 from the second ring.
    # Or simply, take the 12 lattice points with the smallest distance from the center.
    
    # Let's generate a sufficient number of lattice points and pick the 12 closest to origin.
    # Lattice generation for flat-top hexagons:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            points.append((x, y))
            
    # Sort by distance from origin
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first 12
    selected_points = points[:12]
    
    # The inner hexagons are placed at these centers.
    # They all have angle 0 (aligned with the lattice).
    inner_positions = [(x, y, 0.0) for x, y in selected_points]
    
    # Now, find the minimum outer hexagon side length.
    # Since the inner hexagons are on a lattice and aligned, the optimal outer hexagon
    # should also be aligned (angle 0) and centered at the centroid of the inner hexagons.
    # Due to symmetry of the selection (closest to origin), the centroid should be close to (0,0).
    # Let's compute the centroid.
    xs = [p[0] for p in inner_positions]
    ys = [p[1] for p in inner_positions]
    cx = sum(xs) / N
    cy = sum(ys) / N
    
    # The outer hexagon must contain all vertices of the inner hexagons.
    # Inner hexagon vertices for a hexagon at (cx_i, cy_i) with side 1 and angle 0:
    # Local vertices: (1, 0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1, 0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
    # Global vertex: (cx_i + vx, cy_i + vy)
    
    # We need the smallest S such that all these points are inside a hexagon
    # centered at (cx, cy) with side S and angle 0.
    
    # A regular hexagon with side S, angle 0, centered at (cx, cy) is defined by:
    # |x - cx| <= S
    # |y - cy| <= S * sqrt(3) / 2
    # |x - cx| + |y - cy| / sqrt(3) <= S  ... wait, this is for the intersection of half planes.
    
    # Let's use the support function approach or simple geometric constraints.
    # For a flat-top hexagon (vertices at 0, 60, ...):
    # The distance from center to each side (apothem) is S * sqrt(3) / 2.
    # The distance from center to each vertex is S.
    # The hexagon is the intersection of 6 half-planes.
    # Normal vectors for the 6 sides (pointing inward/outward):
    # Side 1 (right): normal (1, 0). Constraint: (x-cx)*1 + (y-cy)*0 <= S * sqrt(3)/2 ? No.
    # Let's check the distance from center to the side.
    # For a regular hexagon of side S, the apothem is S * sqrt(3) / 2.
    # The sides are at distances S * sqrt(3) / 2 from the center.
    # The normals to the sides are at angles 30, 90, 150, 210, 270, 330 degrees.
    # Wait, for flat-top (vertex at 0 deg), the sides are at 30, 90, etc?
    # Vertex at 0: (S, 0).
    # Vertex at 60: (S/2, S*sqrt(3)/2).
    # Midpoint of side 0-60: (3S/4, S*sqrt(3)/4).
    # Vector from center to midpoint: (3S/4, S*sqrt(3)/4). Length: sqrt(9S^2/16 + 3S^2/16) = sqrt(12S^2/16) = S*sqrt(3)/2. Correct.
    # Normal direction: angle 30 deg.
    
    # So the constraints for a point (px, py) to be inside the hexagon centered at (cx, cy) with side S and angle 0 are:
    # For k in 0..5:
    #   angle = 30 + k * 60 degrees
    #   nx = cos(angle), ny = sin(angle)
    #   (px - cx) * nx + (py - cy) * ny <= S * sqrt(3) / 2
    
    # We need to find the minimum S such that for all inner hexagon vertices (px, py):
    #   max_k [ (px - cx) * nx_k + (py - cy) * ny_k ] / (sqrt(3)/2) <= S
    
    # So S = max over all vertices and all 6 normals of:
    #   ( (px - cx) * nx_k + (py - cy) * ny_k ) / (sqrt(3) / 2)
    
    # Let's compute the required S.
    max_s = 0.0
    sqrt3_half = sqrt3 / 2.0
    
    # Precompute normals
    normals = []
    for k in range(6):
        angle = np.radians(30 + k * 60)
        normals.append((np.cos(angle), np.sin(angle)))
    
    # Generate all vertices of the 12 inner hexagons
    for x, y, _ in inner_positions:
        # Local vertices of unit hexagon (side 1, angle 0)
        # (1, 0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1, 0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
        local_verts = [
            (1.0, 0.0),
            (0.5, sqrt3/2),
            (-0.5, sqrt3/2),
            (-1.0, 0.0),
            (-0.5, -sqrt3/2),
            (0.5, -sqrt3/2)
        ]
        for vx, vy in local_verts:
            px = x + vx
            py = y + vy
            for nx, ny in normals:
                dist = (px - cx) * nx + (py - cy) * ny
                s_needed = dist / sqrt3_half
                if s_needed > max_s:
                    max_s = s_needed
    
    # The outer side length is max_s
    outer_side = max_s
    outer_center = (cx, cy)
    outer_angle = 0.0
    
    # Return format: inner rows (x, y, angle_degrees), outer center (x, y), side, angle
    inner = [[x, y, 0.0] for x, y, _ in inner_positions]
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
