import math
import time
import random
import numpy as np
from scipy.optimize import linprog

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    sqrt3 = math.sqrt(3.0)

    # 1. Generate lattice points for inner hexagon centers
    # Axial to Cartesian for flat-top hexagons
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return x, y

    # 12 closest points to origin in hex lattice
    # Ring 0: 1
    # Ring 1: 6
    # Ring 2: 12 (we need 5)
    
    # Define the 7 points of ring 0 and 1
    pts = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1)
    ]
    
    # Ring 2 points (axial)
    ring2 = [
        (2, -1), (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1),
        (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1), # wait, duplicates in list generation logic?
    ]
    # Let's just generate all points in radius 2 and pick 12 closest
    candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Hex distance
            dist = (abs(q) + abs(r) + abs(q+r)) / 2.0
            if dist <= 2.0:
                candidates.append((q, r, dist))
                
    # Sort by distance, then by angle to ensure symmetry/compactness if ties
    # For ties (distance 2), we want a contiguous block to maintain convexity/compactness
    # Actually, any 5 from ring 2 will work, but contiguous is better for bounding box.
    # Let's just take the first 5 from the sorted list of ring 2 points.
    # To ensure we get a "chunk", we can sort by angle.
    
    def axial_angle(q, r):
        x, y = axial_to_cartesian(q, r)
        return math.atan2(y, x)
        
    ring2_points = [p for p in candidates if p[2] == 2.0]
    ring2_points.sort(key=lambda p: axial_angle(p[0], p[1]))
    
    # Take 5 contiguous points from ring 2
    # Start at index 0 (angle ~ -30 deg? or 0?)
    # Let's just take the first 5 in the sorted angular order
    selected_ring2 = ring2_points[:5]
    
    all_axial = [(p[0], p[1]) for p in candidates if p[2] < 2.0] + [(p[0], p[1]) for p in selected_ring2]
    
    # Convert to Cartesian
    centers = [axial_to_cartesian(q, r) for q, r in all_axial]
    
    # 2. Generate vertices for all 12 inner hexagons (angle 0)
    # Unit hexagon vertices (flat top)
    hex_verts_local = [
        (1.0, 0.0),
        (0.5, sqrt3 / 2.0),
        (-0.5, sqrt3 / 2.0),
        (-1.0, 0.0),
        (-0.5, -sqrt3 / 2.0),
        (0.5, -sqrt3 / 2.0)
    ]
    
    all_vertices = []
    for cx, cy in centers:
        for vx, vy in hex_verts_local:
            all_vertices.append((cx + vx, cy + vy))
            
    # 3. Optimize outer hexagon
    # Variables: Cx, Cy, t
    # t is the apothem of the outer hexagon
    # L = 2 * t / sqrt3
    
    best_L = float('inf')
    best_Cx = 0.0
    best_Cy = 0.0
    best_angle = 0.0
    
    # Coarse search for angle
    # The problem has 60-degree symmetry, so we only need to search 0 to 60
    for angle_deg in np.arange(0, 60, 0.5):
        if time.time() > deadline:
            break
            
        angle_rad = math.radians(angle_deg)
        # Normals of outer hexagon
        # Edge normals are perpendicular to edges.
        # For a regular hexagon with "flat top" at angle 0, the edges are at angles 0, 60, ...?
        # No, let's define the hexagon by its normals.
        # If the hexagon is rotated by angle_deg, the normals are at angle_deg + 30 + k*60?
        # Let's stick to a consistent definition.
        # A regular hexagon with side L and center C.
        # Let's define the 6 supporting lines.
        # The normal angles are $\alpha + k \pi/3$.
        # If we say the hexagon is "rotated by angle_deg", we need to map this to normal angles.
        # Let's assume angle_deg is the rotation of the hexagon shape.
        # If angle=0, flat-top. Normals are at 30, 90, 150, 210, 270, 330?
        # Flat top hexagon: Top edge is horizontal. Normal is vertical (90 deg).
        # So normals are 90, 150, 210, 270, 330, 30.
        # Generally, normals are at $\pi/2 + angle_rad + k \pi/3$.
        
        normals = []
        for k in range(6):
            normal_angle = math.pi / 2 + angle_rad + k * math.pi / 3
            normals.append((math.cos(normal_angle), math.sin(normal_angle)))
            
        # Setup LP
        # Minimize t
        # Subject to:
        # For each v in all_vertices, each n in normals:
        # (v - C) . n <= t
        # -(v - C) . n <= t
        
        # Variables: x, y, t
        # c = [0, 0, 1]
        c = [0.0, 0.0, 1.0]
        
        A_ub = []
        b_ub = []
        
        for vx, vy in all_vertices:
            for nx, ny in normals:
                # (vx - Cx)nx + (vy - Cy)ny <= t
                # -Cx nx - Cy ny - t <= - (vx nx + vy ny)
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(vx * nx + vy * ny))
                
                # -((vx - Cx)nx + (vy - Cy)ny) <= t
                # Cx nx + Cy ny - t <= vx nx + vy ny
                A_ub.append([nx, ny, -1.0])
                b_ub.append(vx * nx + vy * ny)
                
        # t >= 0
        # linprog handles bounds
        bounds = [(None, None), (None, None), (0.0, None)]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0:
            t_opt = res.fun
            L_opt = 2.0 * t_opt / sqrt3
            
            if L_opt < best_L:
                best_L = L_opt
                best_Cx = res.x[0]
                best_Cy = res.x[1]
                best_angle = angle_deg

    # Refine the best angle found
    if best_L < float('inf'):
        # Fine search around best_angle
        start_a = max(0, best_angle - 1.0)
        end_a = min(60, best_angle + 1.0)
        for angle_deg in np.arange(start_a, end_a, 0.05):
             if time.time() > deadline:
                 break
             angle_rad = math.radians(angle_deg)
             normals = []
             for k in range(6):
                 normal_angle = math.pi / 2 + angle_rad + k * math.pi / 3
                 normals.append((math.cos(normal_angle), math.sin(normal_angle)))
             
             c = [0.0, 0.0, 1.0]
             A_ub = []
             b_ub = []
             
             for vx, vy in all_vertices:
                 for nx, ny in normals:
                     A_ub.append([-nx, -ny, -1.0])
                     b_ub.append(-(vx * nx + vy * ny))
                     A_ub.append([nx, ny, -1.0])
                     b_ub.append(vx * nx + vy * ny)
                     
             bounds = [(None, None), (None, None), (0.0, None)]
             res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
             
             if res.status == 0:
                 t_opt = res.fun
                 L_opt = 2.0 * t_opt / sqrt3
                 if L_opt < best_L:
                     best_L = L_opt
                     best_Cx = res.x[0]
                     best_Cy = res.x[1]
                     best_angle = angle_deg

    # Construct return values
    # Inner hexagons: centers and angle 0
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    center = (best_Cx, best_Cy)
    side = best_L
    angle = best_angle
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
