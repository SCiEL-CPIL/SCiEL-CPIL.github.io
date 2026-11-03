<<<<<<< SEARCH
scale = 1.05
local_pos = [
        (-3.5*scale, -sqrt3*scale), (-1.5*scale, -sqrt3*scale), (0.5*scale, -sqrt3*scale), (2.5*scale, -sqrt3*scale),
        (-2.5*scale, 0.0),          (-0.5*scale, 0.0),          (1.5*scale, 0.0),          (3.5*scale, 0.0),
        (-3.5*scale, sqrt3*scale),  (-1.5*scale, sqrt3*scale),  (0.5*scale, sqrt3*scale),  (2.5*scale, sqrt3*scale)
    ]
=======
import math
import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 115  # Safety margin from 120s limit
    
    sqrt3 = math.sqrt(3)
    
    # Initial guess: 12 unit hexagons in a 3x4 grid arrangement (rotated/aligned)
    # A regular hexagon of side 1 has width 2*sqrt(3) (flat-to-flat) and height 2 (point-to-point) depending on orientation.
    # Let's assume "flat-top" hexagons for the inner ones initially, or just use a lattice.
    # The optimal packing of 12 unit hexagons in a hexagon is likely related to the hexagonal lattice.
    
    # Let's try a configuration based on a 3x4 block of hexagons in a lattice.
    # Lattice vectors for hexagonal tiling of unit hexagons:
    # If hexagons are flat-top:
    # dx = 1.5, dy = sqrt(3)/2 for neighbors in one direction?
    # Actually, let's just place them in a rectangular grid and optimize.
    # Or better, use the known optimal packing for n=12.
    # Reference: Packing 12 unit hexagons in a hexagon.
    # The outer hexagon side length L needs to accommodate the bounding box of the inner hexagons.
    
    # Let's start with a simple grid of 12 hexagons: 3 rows of 4.
    # Inner hexagons are unit side.
    # Let's align them with the outer hexagon.
    
    # Coordinates for 12 hexagons in a 3x4 grid (centered)
    # Spacing:
    # Horizontal distance between centers: 1.5 (for flat-top)
    # Vertical distance between centers: sqrt(3) (for flat-top, row offset)
    # Wait, standard hex grid:
    # Row 0: x = 0, 3, 6, 9 (scaled by 1.5?)
    # Let's use the lattice vectors for flat-top hexagons of side 1.
    # Width = sqrt(3), Height = 2.
    # Distance between adjacent centers in same row: 1.5 * 2? No.
    # Side length 1.
    # Distance between centers of adjacent flat-top hexagons horizontally: 1.5 + 0.5? No.
    # Let's calculate:
    # Hexagon vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2) is pointy-top.
    # Let's use pointy-top for easier y-spacing?
    # Pointy-top: Width = sqrt(3), Height = 2.
    # Horizontal spacing: sqrt(3)
    # Vertical spacing: 1.5
    # Offset every other row by sqrt(3)/2.
    
    # Let's try a 3x4 grid of pointy-top hexagons.
    # Row 0 (y=0): x = 0, sqrt(3), 2*sqrt(3), 3*sqrt(3)
    # Row 1 (y=1.5): x = sqrt(3)/2, 1.5*sqrt(3), 2.5*sqrt(3), 3.5*sqrt(3)
    # Row 2 (y=3.0): x = 0, sqrt(3), 2*sqrt(3), 3*sqrt(3)
    
    # This bounding box needs to fit in a regular hexagon.
    # We can rotate the inner cluster and scale the outer hexagon.
    
    # Let's define the initial positions relative to center (0,0)
    s3 = math.sqrt(3)
    positions = []
    for row in range(3):
        for col in range(4):
            y = row * 1.5
            if row % 2 == 0:
                x = col * s3
            else:
                x = (col + 0.5) * s3
            positions.append((x, y))
            
    # Center the positions
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    
    centered_positions = [(x - cx, y - cy) for x, y in positions]
    
    # Initial outer hexagon: side length 1, center (0,0), rotation 0
    # We need to find the minimum outer side length S and rotation R such that all inner hexagons fit.
    
    # Function to check if a configuration fits in an outer hexagon of side S, rotation R, center (0,0)
    # Inner hexagons have side 1.
    # This is a complex geometric check. 
    # Approximation: Check if all vertices of inner hexagons are inside the outer hexagon.
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return verts
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        # Transform point to hexagon local frame
        dx = px - cx
        dy = py - cy
        # Rotate by -angle
        a = -math.radians(angle_deg)
        lx = dx * math.cos(a) - dy * math.sin(a)
        ly = dx * math.sin(a) + dy * math.cos(a)
        
        # Hexagon is flat-top in local frame? 
        # Standard regular hexagon centered at origin, side S.
        # Vertices at angles 0, 60, ...
        # The hexagon is the intersection of 6 half-planes.
        # Or simpler: max distance from center in 6 directions?
        # Let's use the distance to edges.
        # Edge normal directions: 0, 60, 120, 180, 240, 300 degrees.
        # Distance from center to edge (apothem) = side * sqrt(3) / 2
        apothem = side * math.sqrt(3) / 2
        
        for i in range(6):
            theta = i * math.pi / 3
            # Project point onto normal
            proj = lx * math.cos(theta) + ly * math.sin(theta)
            if proj > apothem:
                return False
        return True

    def fits(inner_pos, inner_angle, outer_side, outer_angle, outer_center=(0,0)):
        for (ix, iy) in inner_pos:
            # Get vertices of this inner hexagon
            verts = get_hex_vertices(ix, iy, 1.0, inner_angle)
            for (vx, vy) in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    return False
        return True

    # Optimization:
    # We want to minimize outer_side.
    # Variables: inner_angle (rotation of the whole cluster), outer_angle (rotation of outer hex), outer_side.
    # Actually, we can fix outer_center at (0,0) and inner cluster center at (0,0).
    # We can rotate the inner cluster relative to the outer hex.
    # Let's search over inner_rotation and outer_rotation.
    # For a given pair of rotations, find the minimal outer_side.
    
    def min_outer_side(inner_rot, outer_rot):
        # Binary search on outer_side
        lo = 1.0
        hi = 20.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            # Check fit
            # Inner positions are fixed relative to each other, but rotated by inner_rot
            # We need to apply rotation to centered_positions
            a = math.radians(inner_rot)
            rotated_pos = []
            for (x, y) in centered_positions:
                rx = x * math.cos(a) - y * math.sin(a)
                ry = x * math.sin(a) + y * math.cos(a)
                rotated_pos.append((rx, ry))
                
            if fits(rotated_pos, 0.0, mid, outer_rot):
                hi = mid
            else:
                lo = mid
        return hi

    # Grid search for rotations
    best_side = 20.0
    best_inner_rot = 0.0
    best_outer_rot = 0.0
    
    # Try a range of rotations
    # Due to symmetry, we only need to check 0 to 60 degrees for both.
    step = 1.0
    for ir in range(0, 60, step):
        if time.time() > start_time + timeout:
            break
        for or_ in range(0, 60, step):
            if time.time() > start_time + timeout:
                break
            side = min_outer_side(ir, or_)
            if side < best_side:
                best_side = side
                best_inner_rot = ir
                best_outer_rot = or_
    
    # Refine around best
    if time.time() < start_time + timeout - 5:
        step = 0.1
        for ir in [best_inner_rot - 1, best_inner_rot, best_inner_rot + 1]:
            for or_ in [best_outer_rot - 1, best_outer_rot, best_outer_rot + 1]:
                if time.time() > start_time + timeout:
                    break
                side = min_outer_side(ir, or_)
                if side < best_side:
                    best_side = side
                    best_inner_rot = ir
                    best_outer_rot = or_
    
    # Construct final result
    # Inner hexagons: (x, y, angle)
    # The angle in the interface is likely the rotation of the hexagon itself.
    # If we rotate the cluster by best_inner_rot, each hexagon is also rotated by best_inner_rot.
    a = math.radians(best_inner_rot)
    
    inner_rows = []
    for (x, y) in centered_positions:
        rx = x * math.cos(a) - y * math.sin(a)
        ry = x * math.sin(a) + y * math.cos(a)
        inner_rows.append((rx, ry, best_inner_rot))
        
    center = (0.0, 0.0)
    side = best_side
    angle = best_outer_rot
    
    return inner_rows, center, side, angle
>>>>>>> REPLACE
