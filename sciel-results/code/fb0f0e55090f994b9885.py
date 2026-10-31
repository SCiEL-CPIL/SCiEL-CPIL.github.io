"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _hex_vertices_array(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon as (6,2) array."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _min_enclosing_hex_side(inner_positions, inner_angles, outer_center, outer_rotation_deg):
    """
    Compute the minimum side length of an outer hexagon centered at outer_center
    with given rotation that contains all inner hexagons.
    inner_positions: list of (x, y)
    inner_angles: list of angles in degrees
    """
    # Get all vertices of all inner hexagons
    all_verts = []
    for i in range(len(inner_positions)):
        verts = _hex_vertices_array(inner_positions[i][0], inner_positions[i][1], 1.0, inner_angles[i])
        all_verts.append(verts)
    
    if not all_verts:
        return 0.0
        
    all_verts = np.vstack(all_verts)
    # Move to outer center frame
    all_verts = all_verts - np.array(outer_center)
    
    phi = np.radians(outer_rotation_deg)
    # The 6 normal directions of the outer hexagon
    thetas = phi + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    
    max_proj = 0.0
    for theta in thetas:
        proj = all_verts[:, 0] * np.cos(theta) + all_verts[:, 1] * np.sin(theta)
        mp = np.max(proj)
        if mp > max_proj:
            max_proj = mp
            
    # The apothem (inradius) of a regular hexagon with side s is s * sqrt(3) / 2
    # max_proj is the maximum distance from center to any vertex in the direction of the normal.
    # This distance must be <= apothem of the outer hexagon.
    # apothem = side * sqrt(3) / 2
    # side = max_proj / (sqrt(3) / 2)
    return max_proj / (math.sqrt(3) / 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Define the 11 hexagons in a standard hexagonal lattice (1 center, 6 ring, 4 outer)
    # Axial coordinates (q, r)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    def axial_to_xy(q, r):
        # Standard hex to pixel conversion for pointy-top or flat-top?
        # Let's use flat-top (angle 0 is flat side? No, standard math angle 0 is right).
        # For a hexagon with side 1, distance between centers of touching hexagons is sqrt(3).
        # Let's use the geometry that matches the vertex generation.
        # _hex_vertices_array uses angles a + pi/6 + ...
        # This implies the "flat" side is at angle 0?
        # Vertex at angle pi/6 is (s*cos(pi/6), s*sin(pi/6)) = (s*sqrt(3)/2, s/2).
        # This is a pointy-top hexagon? No, vertex at 30 deg.
        # Let's just calculate the distance between (0,0) and (1,0) in this lattice.
        # If we use axial (q,r) to cartesian (x,y) for flat-top hexes:
        # x = 3/2 * q
        # y = sqrt(3) * (r + q/2)
        # Distance between (0,0) and (1,0) is 1.5. This is wrong for touching hexes (should be sqrt(3) ~ 1.732).
        # The distance between centers of two touching unit hexagons is sqrt(3).
        # So the lattice spacing must be sqrt(3).
        # Let's define the positions directly in Cartesian to be safe.
        return (1.5 * q, math.sqrt(3) * (r + q / 2.0))
        
    # Wait, the distance between (0,0) and (1,0) in axial_to_xy above is 1.5.
    # The distance between (0,0) and (0,1) is sqrt(3).
    # This is not a uniform lattice for all neighbors.
    # Let's use the property that the 6 neighbors are at distance sqrt(3) from the center.
    # Angles: 0, 60, 120, 180, 240, 300?
    # Let's generate positions explicitly.
    
    center = (0.0, 0.0)
    positions = [center]
    angles = [0.0]
    
    dist = math.sqrt(3)
    
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        x = dist * np.cos(angle)
        y = dist * np.sin(angle)
        positions.append((x, y))
        angles.append(0.0)
        
    # Ring 2 (4 nodes to make 11 total)
    # We need 4 more. Let's pick the ones that form a compact shape.
    # The ring 2 nodes are at distance 2*dist from center? No, they are neighbors of ring 1.
    # A compact packing of 11 is usually the "hexagon of side 2" minus 2 corners?
    # Side 2 hexagon has 19 hexes.
    # Side 1.5? No.
    # Let's just add 4 nodes from the next shell that are close to the center.
    # The next shell nodes are at distance sqrt(3)*sqrt(3) = 3? No.
    # Distance to ring 2 nodes in hex lattice is 2 * sqrt(3) * cos(30) ?
    # Let's use the axial coordinates and scale them.
    # Axial (q,r) -> Cartesian for pointy top:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # Distance between (0,0) and (1,0) is sqrt(3).
    # Distance between (0,0) and (0,1) is sqrt(3).
    # This is a valid lattice for touching hexagons.
    
    def axial_to_xy_v2(q, r):
        return (math.sqrt(3) * (q + r / 2.0), 1.5 * r)
        
    # Let's re-define positions using this lattice
    positions = []
    angles = [0.0] * 11
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    for q, r in axial_coords:
        x, y = axial_to_xy_v2(q, r)
        positions.append((x, y))
        
    # Verify no overlaps (they shouldn't overlap if on this lattice with same orientation)
    # Distance between any two distinct lattice points is >= sqrt(3).
    # Min distance is sqrt(3). So they touch but do not overlap.
    
    # Initial guess for outer hexagon
    outer_center_init = (0.0, 0.0)
    outer_rot_init = 0.0
    
    # We will optimize:
    # 1. Translation of the inner cluster (dx, dy)
    # 2. Rotation of the inner cluster (dtheta)
    # 3. Translation of the outer hexagon (odx, ody)
    # 4. Rotation of the outer hexagon (otheta)
    
    def objective(params):
        if time.time() > deadline:
            return float('inf')
            
        dx, dy, dtheta, odx, ody, otheta = params
        
        # Transform inner positions
        cos_t = np.cos(dtheta)
        sin_t = np.sin(dtheta)
        
        transformed_positions = []
        for i in range(N):
            x, y = positions[i]
            # Rotate
            rx = x * cos_t - y * sin_t
            ry = x * sin_t + y * cos_t
            # Translate
            tx = rx + dx
            ty = ry + dy
            transformed_positions.append((tx, ty))
            
        # Inner angles also rotate
        transformed_angles = [a + math.degrees(dtheta) for a in angles]
        
        outer_c = (odx, ody)
        
        return _min_enclosing_hex_side(transformed_positions, transformed_angles, outer_c, otheta)
        
    # Initial params: 0, 0, 0, 0, 0, 0
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_side = float('inf')
    best_params = x0
    
    try:
        # Use L-BFGS-B for smooth optimization
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_side:
            best_side = result.fun
            best_params = result.x
            
    except Exception:
        pass
        
    # Finalize with best params
    dx, dy, dtheta, odx, ody, otheta = best_params
    
    cos_t = np.cos(dtheta)
    sin_t = np.sin(dtheta)
    
    final_inner = []
    for i in range(N):
        x, y = positions[i]
        rx = x * cos_t - y * sin_t
        ry = x * sin_t + y * cos_t
        tx = rx + dx
        ty = ry + dy
        final_angle = angles[i] + math.degrees(dtheta)
        final_inner.append([tx, ty, final_angle])
        
    final_outer_center = (odx, ody)
    final_outer_side = best_side
    final_outer_rot = otheta
    
    # Add tiny epsilon to side to ensure containment due to floating point
    final_outer_side *= (1.0 + 1e-9)
    
    return final_inner, final_outer_center, final_outer_side, math.degrees(final_outer_rot)

# EVOLVE-BLOCK-END
