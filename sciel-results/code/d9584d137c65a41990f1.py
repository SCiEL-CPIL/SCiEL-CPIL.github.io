"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses a rigid-body cluster approach with LP for minimum enclosing hexagon.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog

N = 11
DEADLINE_MARGIN = 10.0  # seconds before the 150s limit


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True


def _inner_in_outer(inner_hex, outer_hex):
    """Check if inner hexagon is fully inside outer hexagon."""
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True


def _min_outer_side_lp(points, oangle=0.0):
    """
    Find minimum side length of outer hexagon enclosing a set of points.
    points: list of (x, y)
    oangle: orientation of outer hexagon in degrees
    Returns: (center_x, center_y, side_length)
    """
    # Variables: [cx, cy, t] where t is the apothem of outer hex
    # Objective: minimize t
    # Constraints: For each point (px, py) and each of 6 normals n_k of outer hex:
    #   (px - cx) * nx_k + (py - cy) * ny_k <= t
    #   => -cx*nx_k - cy*ny_k - t <= -px*nx_k - py*ny_k
    
    c = [0, 0, 1]  # minimize t
    
    A_ub = []
    b_ub = []
    
    # Precompute normals for outer hexagon
    normals = []
    for k in range(6):
        theta = math.radians(oangle + 30 + 60 * k)
        normals.append((math.cos(theta), math.sin(theta)))
    
    for px, py in points:
        for nx, ny in normals:
            # -nx*cx - ny*cy - t <= -nx*px - ny*py
            A_ub.append([-nx, -ny, -1])
            b_ub.append(-nx * px - ny * py)
    
    # t >= 0
    A_ub.append([0, 0, -1])
    b_ub.append(0)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
    
    if res.success:
        t = res.x[2]
        side = t * 2.0 / math.sqrt(3)
        return res.x[0], res.x[1], side
    else:
        return 0.0, 0.0, float('inf')


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    sqrt3 = math.sqrt(3)
    
    # Define the rigid cluster of 11 hexagons in axial coordinates
    # Center at (0,0)
    # Ring 1: 6 neighbors
    # Ring 2: Select 4 to form a compact cluster.
    # A common compact shape for 11 is a hexagon of radius 2 minus 2 corners, 
    # or simply taking the first 4 of ring 2. 
    # Let's use the standard "flower" of 7 + 4 more.
    # To be safe and compact, we can use the hexagon of radius 2 (19 hexes) 
    # and pick 11 that are centrally located.
    # Actually, the most compact packing of 11 hexagons is often cited as 
    # a hexagon of side 2 (19 hexes) is too big. 
    # Side 1 (7 hexes) + 4 more.
    # Let's generate axial coords for radius 2 and pick 11.
    
    axial_coords = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                axial_coords.append((q, r))
                
    # We need exactly 11. The center 7 are definitely in.
    # The next 6 are in ring 2. We pick 4 from ring 2.
    # To maintain symmetry/compactness, we can just fix a specific set.
    # Let's use: (0,0) + ring1 (6) + 4 from ring2.
    # Ring 2 coords: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2), (-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,2)
    # Let's pick (2,0), (0,-2), (-2,0), (0,2) ? No, that's not compact.
    # A known optimal packing for 11 hexagons in a hexagon is often related to 
    # fitting into a hexagon of side ~3.
    
    # Let's just use the first 11 from a spiral or sorted list?
    # Better: Use the 7 center + 4 specific ring 2 neighbors to form a shape close to a hexagon.
    # Let's try: (0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1) [Center 7]
    # Add: (2,0), (2,-1), (0,-2), (-2,0) ?
    
    # Actually, let's just define the points in Cartesian for the "base" configuration
    # and then rotate the whole cluster.
    
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)
        
    # Base configuration: 1 center, 6 ring 1, 4 from ring 2.
    # Let's pick 4 from ring 2 that are "closest" to each other to form a compact blob.
    # (2,0), (2,-1), (1,-2), (0,-2) ?
    # Let's try a symmetric set if possible, or just optimize the rotation.
    # The shape itself matters. Let's use the 11 hexagons that form a hexagon of side 2 
    # minus 8 corners? No.
    # Let's use the 11 hexagons centered at (0,0) in a hex grid.
    # The 11 hexagons with minimal bounding box in axial coords are likely:
    # (0,0)
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # (2,0), (2,-1), (1,-2), (0,-2) -- this is asymmetric.
    
    # Let's just try a few fixed configurations of 11 hexagons.
    # Config 1: Center 7 + (2,0), (0,-2), (-2,0), (0,2) [Cross shape + center]
    # Config 2: Center 7 + (2,0), (2,-1), (1,-2), (0,-2) [Compact corner]
    
    configs = []
    
    # Config 1: Symmetric cross
    c1_axial = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
                (2,0), (0,-2), (-2,0), (0,2)]
    configs.append(c1_axial)
    
    # Config 2: Compact blob
    c2_axial = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
                (2,0), (2,-1), (1,-2), (0,-2)]
    configs.append(c2_axial)
    
    # Config 3: Another compact blob
    c3_axial = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
                (1,2), (0,2), (-1,2), (-2,1)]
    configs.append(c3_axial)
    
    best_side = float('inf')
    best_result = None
    
    # Search over cluster rotations and outer orientations
    # Cluster rotation: 0 to 60 degrees (hex symmetry)
    # Outer rotation: 0 to 60 degrees
    
    cluster_rotations = np.linspace(0, 60, 37) # 1.5 deg steps
    outer_rotations = np.linspace(0, 60, 13) # 5 deg steps
    
    for config_axial in configs:
        # Convert base config to Cartesian
        base_points = [axial_to_xy(q, r) for q, r in config_axial]
        
        for c_rot_deg in cluster_rotations:
            if time.time() > deadline:
                break
            
            c_rot_rad = math.radians(c_rot_deg)
            cos_c = math.cos(c_rot_rad)
            sin_c = math.sin(c_rot_rad)
            
            # Rotate base points
            rotated_points = []
            for x, y in base_points:
                rx = x * cos_c - y * sin_c
                ry = x * sin_c + y * cos_c
                rotated_points.append((rx, ry))
                
            for o_rot_deg in outer_rotations:
                if time.time() > deadline:
                    break
                
                cx, cy, side = _min_outer_side_lp(rotated_points, o_rot_deg)
                
                if side < best_side:
                    # Translate points so that the outer hexagon center is at (cx, cy)
                    # The LP finds the center (cx, cy) that minimizes the side length.
                    # The points in the LP are relative to the origin.
                    # So the actual positions are rotated_points + (cx, cy)
                    final_points = [(px + cx, py + cy) for px, py in rotated_points]
                    
                    best_side = side
                    best_result = {
                        'points': final_points,
                        'center': (cx, cy),
                        'side': side,
                        'angle': o_rot_deg,
                        'cluster_rot': c_rot_deg
                    }
    
    if best_result is None:
        # Fallback: simple lattice
        base_points = [axial_to_xy(q, r) for q, r in [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]]
        cx, cy, side = _min_outer_side_lp(base_points, 0.0)
        final_points = [(px + cx, py + cy) for px, py in base_points]
        best_result = {
            'points': final_points,
            'center': (cx, cy),
            'side': side,
            'angle': 0.0,
            'cluster_rot': 0.0
        }
    
    # Format output
    # Inner hexagons have side length 1 and angle 0 (since we rotated the cluster, 
    # the individual hexagons are also rotated by the cluster rotation)
    # Wait, the task says "each inner hexagon may translate and rotate".
    # If we rotate the cluster, the individual hexagons are rotated by c_rot_deg.
    # So the angle of each inner hexagon is c_rot_deg.
    
    inner = []
    for x, y in best_result['points']:
        inner.append([x, y, best_result['cluster_rot']])
        
    center = best_result['center']
    side = best_result['side']
    angle = best_result['angle']
    
    return inner, center, side, angle
