import numpy as np
import math
import time
import itertools

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    if N != 11:
        # Fallback to original logic for other N if needed, 
        # but for now assume N=11 is the primary target or 
        # use a simple grid for others.
        # Given the task context, N is likely fixed.
        pass

    sqrt3 = math.sqrt(3)
    
    # Unit hexagon vertices (flat-top or pointy-top? 
    # The original code used axial_to_cartesian which implies a specific orientation.
    # axial (q,r) -> x = 2q+r, y=sqrt3*r.
    # This corresponds to pointy-top hexagons?
    # Let's check: (0,0)->(0,0). (1,0)->(2,0). (0,1)->(1,sqrt3).
    # Distance (0,0) to (1,0) is 2.
    # Distance (0,0) to (0,1) is sqrt(1+3)=2.
    # This is a standard hex grid.
    # The hexagons themselves: 
    # If we assume the hexagons are "flat-top" (vertices at 0, 60, ...), 
    # then the grid spacing matches.
    # Let's define the vertices of a unit hexagon at origin, angle 0.
    
    def get_hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + math.cos(theta), cy + math.sin(theta)))
        return verts

    # Precompute vertices for unit hexagon at (0,0), angle 0
    v0 = get_hex_vertices(0, 0, 0.0)
    
    # Generate grid positions
    def hex_ring_positions(ring):
        if ring == 0:
            return [(0, 0)]
        positions = []
        q, r = ring, 0
        for side in range(6):
            for step in range(ring):
                positions.append((q, r))
                dq = [1, 0, -1, -1, 0, 1][side]
                dr = [0, -1, -1, 0, 1, 1][side]
                q += dq
                r += dr
        return positions

    def axial_to_cartesian(q, r):
        x = 2.0 * q + r
        y = sqrt3 * r
        return x, y

    # Core: Ring 0 + Ring 1 (7 hexagons)
    core_axial = [(0,0)] + hex_ring_positions(1)
    core_centers = [axial_to_cartesian(q, r) for q, r in core_axial]
    
    # Candidates: Ring 2 (12 hexagons)
    ring2_axial = hex_ring_positions(2)
    ring2_centers = [axial_to_cartesian(q, r) for q, r in ring2_axial]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # We need to choose 4 from 12 ring2 centers
    # To save time, we can prune symmetric solutions or just run all 495.
    # 495 is small.
    
    # Outer angles to try
    outer_angles = [i * 0.5 for i in range(60)] # 0 to 30 deg in 0.5 steps
    
    # Precompute max projection of unit hexagon vertices for each outer angle?
    # No, the normals change.
    
    for comb in itertools.combinations(range(12), 4):
        if time.time() > deadline - 2:
            break
            
        # Construct the 11 centers
        selected_indices = list(comb)
        selected_centers = core_centers[:]
        for idx in selected_indices:
            selected_centers.append(ring2_centers[idx])
            
        # Calculate centroid
        cx = sum(c[0] for c in selected_centers) / 11.0
        cy = sum(c[1] for c in selected_centers) / 11.0
        outer_center = (cx, cy)
        
        # Shift centers relative to outer center for dot products
        # Actually, we can just use absolute coords and subtract outer center in the dot product.
        
        for theta in outer_angles:
            # Normals for outer hexagon
            # Standard hexagon (vertex at 0) has normals at 30, 90, ...
            # If outer hex is rotated by theta, normals are at theta + 30 + k*60
            max_proj = 0.0
            for k in range(6):
                angle_norm = math.radians(theta + 30 + k * 60)
                nx = math.cos(angle_norm)
                ny = math.sin(angle_norm)
                
                # Max projection of inner hexagon vertices
                # proj = n . (C_i + v_j - C_outer)
                # = n . C_i + n . v_j - n . C_outer
                # Max over i, j
                # = max_i(n . C_i) + max_j(n . v_j) - n . C_outer
                
                max_ci = max(nx * c[0] + ny * c[1] for c in selected_centers)
                max_vj = max(nx * v[0] + ny * v[1] for v in v0)
                n_outer = nx * cx + ny * cy
                
                proj = max_ci + max_vj - n_outer
                if proj > max_proj:
                    max_proj = proj
            
            # Required apothem
            apothem = max_proj
            # Side length S = apothem / (sqrt(3)/2)
            side = apothem * 2.0 / sqrt3
            
            if side < best_side:
                best_side = side
                best_outer_center = outer_center
                best_outer_angle = theta
                best_inner = [[c[0], c[1], 0.0] for c in selected_centers]

    if best_inner is None:
        # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    return best_inner, best_outer_center, best_side, best_outer_angle
# EVOLVE-BLOCK-END
