import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin before 120s limit
    
    # Unit regular hexagon with flat-top orientation
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # For a unit hexagon (side=1), the vertices are at distance 1 from center
    # Flat-to-flat distance = sqrt(3), point-to-point = 2
    
    # Define unit hexagon vertices (flat-top, side=1)
    # Vertex i is at angle 60*i degrees, radius 1
    hex_angles = np.array([np.radians(60 * i) for i in range(6)])
    unit_hex_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    # The 6 normal directions for the outer hexagon (flat-top)
    # These are perpendicular to the edges, at angles 30, 90, 150, 210, 270, 330
    outer_normals = np.array([
        [np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))]
        for i in range(6)
    ])
    
    # Honeycomb lattice basis vectors for flat-top hexagons
    # Neighbor positions: 
    # E: (1.5, sqrt(3)/2), W: (-1.5, -sqrt(3)/2)
    # NE: (0.75, sqrt(3)/2+sqrt(3)/2)=(0.75, sqrt(3)), etc.
    # Actually for flat-top hexagons, the 6 neighbors are at:
    sqrt3 = np.sqrt(3)
    neighbor_offsets = np.array([
        [1.5, sqrt3/2],     # E
        [0.75, sqrt3],      # NE
        [-0.75, sqrt3],     # NW
        [-1.5, sqrt3/2],    # W
        [-0.75, -sqrt3],    # SW
        [0.75, -sqrt3],     # SE
    ])
    
    # Place 11 hexagons in a compact honeycomb arrangement
    # Use axial coordinates: (q, r) -> cartesian
    # For flat-top: x = 1.5*q, y = sqrt3*(r + q/2)
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return [x, y]
    
    # Compact arrangement of 11 hexagons:
    # A hexagon of "radius" 1 has 7 hexagons. Add 4 more.
    # Let's try a shape that's close to hexagonal
    # Coordinates in axial (q, r):
    # Center: (0,0)
    # Ring 1 (6 hexagons): (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # That's 7 total. Need 4 more.
    # Add: (2,0), (2,-1), (1,-2), (0,-2) - extending in one direction
    # Or better: make it more symmetric
    # Let's try: center + ring1 + 4 from ring2 positioned to keep shape compact
    
    # Alternative: arrange in rows
    # Row y=0: 5 hexagons at x = -3, -1.5, 0, 1.5, 3
    # Row y=sqrt3: 3 hexagons at x = -2.25, -0.75, 0.75 ... hmm
    
    # Let me use a proper honeycomb cluster
    # 11 hexagons: try a "hexagon" shape with some extensions
    # Best compact shape: a hexagon of side 2 would be 19, too many
    # Side 1.5 isn't integer... 
    # 
    # Let's just use a tight cluster:
    # Positions in axial coords for 11 hexagons (compact):
    axial_positions = [
        (0, 0),      # center
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),  # ring 1
        (2, 0), (2, -1), (1, -2), (-1, 2),  # 4 more
    ]
    
    # Convert to cartesian
    inner_positions = []
    for q, r in axial_positions:
        x, y = axial_to_cartesian(q, r)
        inner_positions.append([x, y, 0.0])  # angle 0 for all
    
    # Compute bounding box in the 6 hexagonal directions
    # For each normal direction, find max projection of all hexagon vertices
    all_vertices = []
    for pos in inner_positions:
        cx, cy = pos[0], pos[1]
        verts = unit_hex_verts + np.array([cx, cy])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Find the minimal enclosing hexagon (same orientation as inner hexagons)
    # The outer hexagon has normals at 30, 90, 150, 210, 270, 330 degrees
    # For each normal, compute max and min projection
    projections = all_vertices @ outer_normals.T  # shape (n_verts, 6)
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    
    # Center the outer hexagon
    center_x = (np.sum(max_proj) + np.sum(min_proj)) / 12.0
    # Actually, for a regular hexagon, the center is at the average of opposite face midpoints
    # Let's compute the center as the midpoint of the bounding region
    # For a regular hexagon with normals n_i, the center c satisfies:
    # c . n_i + R = max_proj_i and c . n_{i+3} - R = min_proj_{i+3}
    # where R is the apothem (inradius) of the outer hexagon
    # Side length s = R / cos(30) = R / (sqrt3/2) = 2R/sqrt3
    
    # The constraint is: for each pair of opposite faces (i, i+3):
    # (max_proj_i - min_proj_{i+3}) / 2 <= R
    # Because the distance between face i and face i+3 is 2R
    
    # Wait, let me reconsider. The outer hexagon has 6 faces with outward normals n_0,...,n_5
    # Face i is at distance R from center in direction n_i
    # For the hexagon to contain all vertices:
    # For vertex v: v . n_i <= R + c . n_i for all i
    # Equivalently: (v - c) . n_i <= R for all i, all vertices v
    
    # So R >= max over all vertices v and all i of (v - c) . n_i
    # We want to minimize R, which means minimizing the side length s = 2R/sqrt3
    
    # Optimal c: minimize max_{v,i} (v-c).n_i
    # This is a convex optimization problem!
    
    from scipy.optimize import minimize
    
    def objective(c):
        cx, cy = c
        diffs = all_vertices - np.array([cx, cy])
        projs = diffs @ outer_normals.T
        return np.max(projs)
    
    # Initial guess: centroid of all vertices
    c0 = np.mean(all_vertices, axis=0)
    
    result = minimize(objective, c0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    c_opt = result.x
    R = objective(c_opt)
    side_length = 2 * R / np.sqrt(3)
    
    # Check time
    if time.time() > deadline:
        # Fallback: use initial estimate
        c_fallback = np.mean(all_vertices, axis=0)
        R_fb = objective(c_fallback)
        side_fb = 2 * R_fb / np.sqrt(3)
        return inner_positions, (float(c_fallback[0]), float(c_fallback[1])), float(side_fb), 0.0
    
    # Try rotating the outer hexagon to see if we can do better
    best_side = side_length
    best_center = c_opt
    best_outer_angle = 0.0
    
    # Try a few outer hexagon rotations
    for outer_rot_deg in [0, 15, 30, 45, 60]:
        if time.time() > deadline - 10:
            break
        rot_rad = np.radians(outer_rot_deg)
        cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
        rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        
        rotated_normals = outer_normals @ rot_mat.T
        
        def obj_rot(c):
            cx, cy = c
            diffs = all_vertices - np.array([cx, cy])
            projs = diffs @ rotated_normals.T
            return np.max(projs)
        
        res_rot = minimize(obj_rot, c0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
        c_rot = res_rot.x
        R_rot = obj_rot(c_rot)
        side_rot = 2 * R_rot / np.sqrt(3)
        
        if side_rot < best_side:
            best_side = side_rot
            best_center = c_rot
            best_outer_angle = outer_rot_deg
    
    # Also try rotating the inner hexagons relative to each other
    # But for now, keep them all at angle 0
    
    return inner_positions, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_outer_angle)
# EVOLVE-BLOCK-END

