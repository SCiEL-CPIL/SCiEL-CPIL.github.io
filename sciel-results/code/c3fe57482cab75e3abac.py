import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600.0  # Safety margin from 630s limit

    # Unit hexagon vertices (angle 0)
    hex_verts_base = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])

    def get_vertices(center, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad), np.cos(angle_rad)]])
        return center + hex_verts_base @ rot.T

    def get_normals(vertices):
        # Returns unit normals for the 6 edges (outward)
        normals = []
        for i in range(6):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % 6]
            edge_vec = v2 - v1
            n = np.array([edge_vec[1], -edge_vec[0]])
            n = n / np.linalg.norm(n)
            normals.append(n)
        return np.array(normals)

    def max_penetration(v1, v2):
        # SAT for two convex polygons. Returns max penetration depth (0 if disjoint).
        n1 = get_normals(v1)
        n2 = get_normals(v2)
        all_normals = np.vstack([n1, n2])
        
        max_pen = 0.0
        for n in all_normals:
            proj1 = v1 @ n
            min1, max1 = np.min(proj1), np.max(proj1)
            proj2 = v2 @ n
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # If there is a gap, they are disjoint
            if max1 < min2 or max2 < min1:
                return 0.0
            
            overlap = np.minimum(max1, max2) - np.maximum(min1, min2)
            if overlap > max_pen:
                max_pen = overlap
        return max_pen

    def compute_side(centers, angles, theta_outer_deg):
        # Compute the minimum side length of the outer hexagon containing all inner hexagons
        # for a given outer hexagon orientation.
        # The outer hexagon is centered at (0,0).
        # The condition is that all inner vertices must be inside the outer hexagon.
        # This is equivalent to: for each of the 6 outward normals of the outer hexagon,
        # the projection of all inner vertices onto that normal must be <= the apothem of the outer hexagon.
        
        theta_outer_rad = np.deg2rad(theta_outer_deg)
        # Normals of the outer hexagon edges
        # The vertices of a regular hexagon with angle 0 are at 0, 60, 120, ...
        # The edge normals are at 30, 90, 150, ...
        # If the outer hexagon is rotated by theta, the normals are at theta + 30, theta + 90, ...
        outer_normals = []
        for i in range(6):
            a = theta_outer_rad + np.deg2rad(30 + 60 * i)
            outer_normals.append([np.cos(a), np.sin(a)])
        outer_normals = np.array(outer_normals)
        
        # Get all inner vertices
        all_verts = []
        for i in range(len(centers)):
            verts = get_vertices(centers[i], angles[i])
            all_verts.append(verts)
        all_verts_flat = np.vstack(all_verts)
        
        # Project onto outer normals
        # proj shape: (66, 6)
        proj = all_verts_flat @ outer_normals.T
        max_proj = np.max(proj, axis=0)
        
        # The max projection is the distance from center to the furthest point in that normal direction.
        # This distance must be <= apothem of the outer hexagon.
        # Apothem = side * sqrt(3) / 2
        # So side = 2 * max_proj / sqrt(3)
        # We need side >= 2 * max(max_proj) / sqrt(3)
        return (2.0 / np.sqrt(3.0)) * np.max(max_proj)

    def objective(params):
        if time.time() > deadline - 5:
            return 1e10

        # params: [x1, y1, a1, ..., x11, y11, a11, theta_outer]
        data = params[:33].reshape(11, 3)
        theta_outer_deg = params[33]
        
        centers = data[:, :2]
        angles = data[:, 2]
        
        # Compute vertices for all inner hexagons
        all_vertices = []
        for i in range(11):
            verts = get_vertices(centers[i], angles[i])
            all_vertices.append(verts)
            
        # Check overlaps
        pen_sum = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                pen = max_penetration(all_vertices[i], all_vertices[j])
                if pen > 0:
                    pen_sum += pen * pen
                    
        # Calculate side length
        side = compute_side(centers, angles, theta_outer_deg)
        
        # Penalty for overlaps
        weight = 1000.0
        return side + weight * pen_sum

    # Initial candidates
    d = np.sqrt(3)
    
    # Ring 1: 6 hexagons around center
    ring1_centers = []
    ring1_angles = []
    for i in range(6):
        angle_pos = np.deg2rad(30 + 60 * i)
        cx = d * np.cos(angle_pos)
        cy = d * np.sin(angle_pos)
        ring1_centers.append([cx, cy])
        ring1_angles.append(0.0)
        
    ring1_centers = np.array(ring1_centers)
    ring1_angles = np.array(ring1_angles)
    
    base_centers = np.vstack([np.array([[0, 0]]), ring1_centers])
    base_angles = np.concatenate([np.array([0.0]), ring1_angles])
    
    # Generate candidates
    candidates = []
    
    # Candidate 1: Symmetric ring 2 (4 hexagons in gaps)
    # Gaps are at 0, 60, 120, 180, 240, 300
    # We pick 4. Let's try 0, 90, 180, 270 (wait, 90 is not a gap in the 60-degree symmetry)
    # The gaps between ring 1 hexagons are at angles 0, 60, 120, 180, 240, 300 relative to the center?
    # No, ring 1 is at 30, 90, 150, 210, 270, 330.
    # The "gaps" (vertices of the central hexagon) are at 0, 60, 120, 180, 240, 300.
    # Let's place ring 2 at 0, 120, 240 and maybe one more? Or 0, 90, 180, 270?
    # Let's try a few configurations.
    
    # Config A: 4 hexagons at 0, 90, 180, 270 (symmetric)
    # Distance: The vertex of center hex is at (1, 0). The vertex of ring 1 hex at 30 deg is at (sqrt(3)*cos(30)+1, sqrt(3)*sin(30)) = (1.5+1, 0.866) = (2.5, 0.866)?
    # Let's just use a heuristic distance.
    # The "gap" center is roughly at distance 2.0?
    # Let's try distance 2.5.
    
    def make_candidate(r2_angles_deg, r2_dist, r2_rot=0.0):
        r2_centers = []
        r2_angles = []
        for a in r2_angles_deg:
            rad = np.deg2rad(a)
            cx = r2_dist * np.cos(rad)
            cy = r2_dist * np.sin(rad)
            r2_centers.append([cx, cy])
            r2_angles.append(r2_rot) # Keep aligned with global grid for now, optimizer will rotate
            
        r2_centers = np.array(r2_centers)
        r2_angles = np.array(r2_angles)
        
        centers = np.vstack([base_centers, r2_centers])
        angles = np.concatenate([base_angles, r2_angles])
        
        init_params = []
        for i in range(11):
            init_params.extend([centers[i][0], centers[i][1], angles[i]])
        init_params.append(0.0) # Outer angle
        return np.array(init_params)
    
    # Try different subsets of 4 angles for ring 2
    # The 6 gap directions are 0, 60, 120, 180, 240, 300.
    # We need to pick 4.
    # Symmetric choices:
    # 0, 90, 180, 270 (not aligned with gaps)
    # 0, 60, 180, 240 (aligned with gaps)
    # 0, 120, 180, 300 (aligned with gaps)
    # 60, 120, 240, 300 (aligned with gaps)
    
    gap_angles = [0, 60, 120, 180, 240, 300]
    
    # Subset 1: 0, 60, 180, 240
    candidates.append(make_candidate([0, 60, 180, 240], 2.5))
    # Subset 2: 0, 120, 180, 300
    candidates.append(make_candidate([0, 120, 180, 300], 2.5))
    # Subset 3: 60, 120, 240, 300
    candidates.append(make_candidate([60, 120, 240, 300], 2.5))
    # Subset 4: 0, 90, 180, 270 (mixed)
    candidates.append(make_candidate([0, 90, 180, 270], 2.5))
    
    # Add some random candidates to escape local minima
    np.random.seed(42)
    for _ in range(30):
        centers = base_centers.copy()
        angles = base_angles.copy()
        
        r2_centers = []
        r2_angles = []
        for i in range(4):
            angle_pos = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(2.0, 3.5)
            cx = r * np.cos(angle_pos)
            cy = r * np.sin(angle_pos)
            r2_centers.append([cx, cy])
            r2_angles.append(np.random.uniform(0, 60))
            
        r2_centers = np.array(r2_centers)
        r2_angles = np.array(r2_angles)
        
        centers = np.vstack([centers, r2_centers])
        angles = np.concatenate([angles, r2_angles])
        
        init_params = []
        for i in range(11):
            init_params.extend([centers[i][0], centers[i][1], angles[i]])
        init_params.append(np.random.uniform(-30, 30))
        
        candidates.append(np.array(init_params))

    best_result = None
    best_score = float('inf')
    
    for init in candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(objective, init, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
            
            if res.fun < best_score:
                # Check validity
                data = res.x[:33].reshape(11, 3)
                theta_outer_deg = res.x[33]
                
                centers = data[:, :2]
                angles = data[:, 2]
                
                all_vertices = []
                for i in range(11):
                    verts = get_vertices(centers[i], angles[i])
                    all_vertices.append(verts)
                    
                valid = True
                for i in range(11):
                    for j in range(i + 1, 11):
                        pen = max_penetration(all_vertices[i], all_vertices[j])
                        if pen > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Calculate final side
                    side = compute_side(centers, angles, theta_outer_deg)
                    
                    if side < best_score:
                        best_score = side
                        # Format output
                        inner_hex_data = np.column_stack([centers, angles])
                        outer_hex_data = np.array([0, 0, theta_outer_deg])
                        best_result = (inner_hex_data, outer_hex_data, best_score)
                        
        except Exception:
            pass

    if best_result is None:
        # Fallback
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    return best_result
