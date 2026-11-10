# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from itertools import product

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon's side length.
    
    Returns:
    inner: np.array of shape (11, 3) with [x, y, angle] for each inner hexagon.
    outer: np.array of shape (3,) with [x, y, angle] for the outer hexagon.
    side: float, the side length of the outer hexagon.
    """
    
    # 1. Define the geometry of a unit hexagon
    # Vertices of a unit hexagon centered at (0,0) with angle 0 (flat top/bottom? No, pointy top/bottom if angle 0 means vertex at 0)
    # Task definition: vertices at cx + s*cos(a + 60i), cy + s*sin(a + 60i)
    # If a=0, vertex 0 is at (s, 0). This is a "pointy-right" hexagon.
    # The "flat" edges are perpendicular to the axes at 30, 90, etc.
    
    def get_hex_vertices(cx, cy, s, a_deg):
        a = np.radians(a_deg)
        angles = a + np.radians(60 * np.arange(6))
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def check_separation(v1, v2, eps=1e-6):
        """
        Check if two convex hexagons are separated on at least one of their 12 edge normals.
        Returns True if separated (valid), False if intersecting (invalid).
        """
        # Check all 6 edges of v1
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
                return True
        
        # Check all 6 edges of v2
        for i in range(6):
            q1 = v2[i]
            q2 = v2[(i + 1) % 6]
            edge = q2 - q1
            normal = np.array([-edge[1], edge[0]])
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
                return True
        
        return False

    def is_valid_packing(verts_list):
        """Check if all pairs of hexagons are separated."""
        n = len(verts_list)
        for i in range(n):
            for j in range(i + 1, n):
                if not check_separation(verts_list[i], verts_list[j]):
                    return False
        return True

    def compute_min_enclosing_hex(all_pts, outer_angle_deg):
        """
        Compute the minimum side length S and center (cx, cy) of a regular hexagon
        with given angle that encloses all points in all_pts.
        Uses Linear Programming.
        """
        # The outer hexagon has center (cx, cy), side S, angle a.
        # The 6 inward normals of the outer hexagon are at angles:
        # a + 30 + 60k for k=0..5 ? 
        # Let's derive.
        # Vertices are at a + 60i.
        # Edge i connects vertex i and i+1.
        # Midpoint of edge i is at angle a + 60i + 30.
        # The inward normal points from the edge towards the center.
        # The vector from center to edge midpoint is in direction a + 60i + 30.
        # So the inward normal n_i is in direction a + 60i + 30.
        # The distance from center to the edge (apothem) is S * sqrt(3) / 2.
        
        # Constraint for each point p_k:
        # dot(p_k - c, n_i) <= S * sqrt(3) / 2  for all i=0..5
        # dot(p_k, n_i) - dot(c, n_i) <= S * sqrt(3) / 2
        
        # We want to minimize S.
        # Variables: c_x, c_y, S.
        # Let x = [c_x, c_y, S].
        
        a_rad = np.radians(outer_angle_deg)
        normals = []
        for k in range(6):
            theta = a_rad + np.radians(30 + 60 * k)
            normals.append(np.array([np.cos(theta), np.sin(theta)]))
        normals = np.array(normals) # shape (6, 2)
        
        # Constraints:
        # For each point p_k (M points) and each normal n_i (6 normals):
        # n_i[0]*p_k[0] + n_i[1]*p_k[1] - n_i[0]*c_x - n_i[1]*c_y - (sqrt(3)/2)*S <= 0
        
        M = len(all_pts)
        # A_ub shape: (6*M, 3)
        # b_ub shape: (6*M,)
        
        A_rows = []
        b_rows = []
        
        for k in range(M):
            px, py = all_pts[k]
            for i in range(6):
                nx, ny = normals[i]
                # nx*px + ny*py - nx*cx - ny*cy - (sqrt(3)/2)*S <= 0
                # -nx*cx - ny*cy - (sqrt(3)/2)*S <= -nx*px - ny*py
                A_rows.append([-nx, -ny, -np.sqrt(3)/2])
                b_rows.append(-nx*px - ny*py)
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        # Objective: Minimize S
        # c = [0, 0, 1]
        c_obj = np.array([0, 0, 1])
        
        # Bounds: S >= 0. c_x, c_y can be anything.
        bounds = [(None, None), (None, None), (0, None)]
        
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            return res.x[2], res.x[0], res.x[1]
        else:
            return np.inf, 0, 0

    # 2. Generate candidate configurations
    # We use a hexagonal lattice.
    # Basis vectors for unit hexagons (pointy-right, side 1):
    # Distance between centers of touching hexagons is sqrt(3).
    # Directions to neighbors: 0, 60, 120, 180, 240, 300 degrees.
    # Vector length sqrt(3).
    sqrt3 = np.sqrt(3)
    basis = np.array([
        [sqrt3, 0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5]
    ])
    
    # To ensure strict separation, we scale the lattice by a factor > 1.
    # Let's use 1.01. This creates a small gap.
    scale = 1.01
    scaled_basis = basis * scale
    
    # Candidate sets of axial coordinates (q, r) for the 11 hexagons.
    # We want compact shapes.
    # Shape 1: Center + 6 neighbors + 4 from distance 2.
    # Distance 2 positions in axial coords (q, r):
    # (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    # Also (1,1), (-1,-1) etc are distance sqrt(3) in cartesian but not lattice neighbors?
    # Axial distance: max(|q|, |r|, |q+r|).
    # Neighbors: dist 1.
    # (1,1) -> dist 1? No, |1|,|1|,|2| -> dist 2.
    # Let's list all axial coords with dist <= 2.
    
    coords = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) <= 2:
                coords.append((q, r))
                
    # We need to pick 11 from these.
    # The most compact is the "flower" of 7 (dist 1) plus 4 from dist 2.
    # Or maybe a 2x3 rectangle plus 5?
    # Let's try a few specific topologies.
    
    topologies = []
    
    # Topology 1: Center (0,0), all 6 neighbors, and 4 from dist 2 ring.
    # Dist 2 ring: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # Note: (1,1) has dist max(1,1,2)=2.
    dist2_coords = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)]
    base7 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Try all combinations of 4 from dist2_coords
    from itertools import combinations
    for combo in combinations(dist2_coords, 4):
        shape = base7 + list(combo)
        # Ensure uniqueness
        if len(set(shape)) == 11:
            topologies.append(shape)
            
    # Topology 2: A 2x3 block of hexagons (6) + 5 others?
    # A 2x3 block in axial:
    # (0,0), (1,0), (2,0)
    # (0,1), (1,1), (2,1)
    # This is a parallelogram.
    block23 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)]
    # Add 5 more from nearby.
    # This might be less compact than the flower.
    
    # Topology 3: 1x11 line? No, too long.
    
    # Let's also try random local search improvements on the best topology found so far?
    # No, stick to the grid search as per directive.
    
    # 3. Evaluate each topology
    best_side = np.inf
    best_positions = None
    best_outer_angle = 0
    best_outer_center = (0,0)
    
    # Search outer angle. The inner cluster has some symmetry, but the outer hexagon
    # might need to rotate to fit.
    # We'll search angles 0 to 60 in steps of 5, then refine.
    angles_to_try = np.arange(0, 60, 5)
    
    for shape in topologies:
        # Calculate Cartesian positions
        positions = []
        for (q, r) in shape:
            # Axial to Cartesian (pointy-right)
            # x = sqrt(3) * (q + r/2)
            # y = 1.5 * r
            # But we are using the basis vectors defined above which correspond to steps.
            # pos = q * basis[0] + r * basis[1]? 
            # Let's check basis[0] = (sqrt3, 0). basis[1] = (sqrt3/2, 1.5).
            # q=1, r=0 -> (sqrt3, 0). Correct.
            # q=0, r=1 -> (sqrt3/2, 1.5). Correct.
            pos = q * scaled_basis[0] + r * scaled_basis[1]
            positions.append(pos)
        
        positions = np.array(positions)
        
        # Compute vertices for all inner hexagons
        # Inner hexagons have angle 0 (aligned with lattice)
        verts_list = []
        for pos in positions:
            verts = get_hex_vertices(pos[0], pos[1], 1.0, 0.0)
            verts_list.append(verts)
            
        # Check separation
        if not is_valid_packing(verts_list):
            continue
            
        all_pts = np.vstack(verts_list)
        
        # Find minimal enclosing hexagon for this configuration
        # We can try multiple outer angles
        local_best_s = np.inf
        local_best_a = 0
        local_best_c = (0,0)
        
        for a_deg in angles_to_try:
            s, cx, cy = compute_min_enclosing_hex(all_pts, a_deg)
            if s < local_best_s:
                local_best_s = s
                local_best_a = a_deg
                local_best_c = (cx, cy)
                
        if local_best_s < best_side:
            best_side = local_best_s
            best_positions = positions
            best_outer_angle = local_best_a
            best_outer_center = local_best_c
            
    # Refinement: Fine-tune the best outer angle found
    if best_positions is not None:
        # Fine search around best_outer_angle
        fine_angles = np.linspace(best_outer_angle - 5, best_outer_angle + 5, 101)
        all_pts = np.vstack([get_hex_vertices(p[0], p[1], 1.0, 0.0) for p in best_positions])
        
        refined_best_s = np.inf
        refined_best_a = best_outer_angle
        refined_best_c = best_outer_center
        
        for a_deg in fine_angles:
            s, cx, cy = compute_min_enclosing_hex(all_pts, a_deg)
            if s < refined_best_s:
                refined_best_s = s
                refined_best_a = a_deg
                refined_best_c = (cx, cy)
                
        if refined_best_s < best_side:
            best_side = refined_best_s
            best_outer_angle = refined_best_a
            best_outer_center = refined_best_c

    # Construct output
    # Inner: (11, 3) [x, y, angle]
    inner_hex_data = np.zeros((11, 3))
    for i, pos in enumerate(best_positions):
        inner_hex_data[i] = [pos[0], pos[1], 0.0]
        
    # Outer: (3,) [x, y, angle]
    outer_hex_data = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    # Side length
    outer_hex_side_length = float(best_side)
    
    # Fallback if something went wrong (e.g., no valid packing found)
    if best_positions is None:
        # Construct a known valid packing manually
        # Center + 6 neighbors + 4 more
        # Use scale 1.02 to be safe
        s = 1.02 * np.sqrt(3)
        centers = [
            (0,0),
            (s, 0), (s/2, 1.5*s/sqrt3), (-s/2, 1.5*s/sqrt3), (-s, 0), (-s/2, -1.5*s/sqrt3), (s/2, -1.5*s/sqrt3),
            (2*s, 0), (0, 2*1.5*s/sqrt3), (-2*s, 0), (0, -2*1.5*s/sqrt3)
        ]
        # Wait, the manual fallback needs to be valid.
        # Let's just return a large box if the algorithm fails, though it shouldn't.
        inner_hex_data = np.array([
            [0, 0, 0],
            [2.5, 0, 0],
            [1.25, 2.17, 0],
            [-1.25, 2.17, 0],
            [-2.5, 0, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [5, 0, 0],
            [3.75, 4.34, 0],
            [-3.75, 4.34, 0],
            [-5, 0, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
